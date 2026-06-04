"""Daemon-level integration for profiles + agents + supervisor (#17).

Model: Profile -> Agent -> Session. Drives a live daemon configured with the
default profile, a flat single-agent profile, and a multi-agent profile;
asserts open-under-(profile,agent), agent selection, profile.list (nested
agents with running/session counts), profile.start/stop, and unknown
profile/agent errors.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import _StreamClient, short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")


def _agent_spec() -> dict:
    return {"agent_command": sys.executable, "agent_args": [FAKE_ACP]}


@pytest.fixture
async def profile_daemon():
    socket_path = short_socket_path("blemeesd-prof")
    with socket_cleanup(socket_path):
        cfg = Config(
            socket_path=str(socket_path),
            agent_command=sys.executable,
            agent_args=[FAKE_ACP],
            profiles={
                # flat → single "default" agent
                "alt": {**_agent_spec(), "model": "x"},
                # multi-agent profile
                "work": {"agents": {"a": _agent_spec(), "b": _agent_spec()}},
            },
            idle_timeout_s=60,
            max_concurrent_sessions=16,
        )
        daemon = Daemon(cfg, configure("error"))
        await daemon.start()
        serve = asyncio.create_task(daemon.serve_forever())
        try:
            yield str(socket_path)
        finally:
            daemon.request_shutdown()
            try:
                await asyncio.wait_for(serve, timeout=5.0)
            except TimeoutError:
                serve.cancel()


async def _connect(socket_path: str):
    reader, writer = await asyncio.open_unix_connection(socket_path)
    client = _StreamClient(reader, writer)
    await client.send({"type": "hello", "client": "t/1", "protocol": PROTOCOL_VERSION})
    ack = await client.recv()
    assert ack["type"] == "hello_ack"
    return client, ack


async def _drive_turn(client, sid, *, profile=None, agent=None):
    frame = {"type": "session.open", "id": "o", "session_id": sid, "options": {}}
    if profile:
        frame["profile"] = profile
    if agent:
        frame["agent"] = agent
    await client.send(frame)
    opened = await client.wait_for(lambda e: e.get("type") == "session.opened")
    await client.send({"type": "session.prompt", "session_id": sid, "prompt": "say pong"})
    await client.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
    return opened


async def test_hello_ack_lists_profiles(profile_daemon):
    client, ack = await _connect(profile_daemon)
    try:
        assert set(ack["profiles"]) == {"default", "alt", "work"}
    finally:
        await client.close()


async def test_open_under_flat_profile_uses_default_agent(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        opened = await _drive_turn(client, "s-alt", profile="alt")
        assert opened["profile"] == "alt"
        assert opened["agent"] == "default"
    finally:
        await client.close()


async def test_open_selects_agent_within_profile(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        opened_a = await _drive_turn(client, "s-a", profile="work", agent="a")
        opened_b = await _drive_turn(client, "s-b", profile="work", agent="b")
        assert opened_a["agent"] == "a"
        assert opened_b["agent"] == "b"
    finally:
        await client.close()


async def test_unknown_profile_and_agent_rejected(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        await client.send(
            {
                "type": "session.open",
                "id": "o",
                "session_id": "s1",
                "profile": "ghost",
                "options": {},
            }
        )
        e1 = await client.wait_for(lambda e: e.get("type") == "error")
        assert e1["code"] == "profile_unknown"

        await client.send(
            {
                "type": "session.open",
                "id": "o",
                "session_id": "s2",
                "profile": "work",
                "agent": "ghost",
                "options": {},
            }
        )
        e2 = await client.wait_for(lambda e: e.get("type") == "error")
        assert e2["code"] == "profile_unknown"
    finally:
        await client.close()


async def test_profile_list_nested_with_running_and_sessions(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        await _drive_turn(client, "s-a", profile="work", agent="a")

        await client.send({"type": "profile.list", "id": "pl"})
        reply = await client.wait_for(lambda e: e.get("type") == "profiles")
        rows = {r["name"]: r for r in reply["profiles"]}
        work_agents = {a["name"]: a for a in rows["work"]["agents"]}
        assert work_agents["a"]["running"] is True
        assert work_agents["a"]["sessions"] >= 1
        # Agent "b" of the same profile was never opened → its own process is idle.
        assert work_agents["b"]["running"] is False
    finally:
        await client.close()


async def test_profile_start_and_stop(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        await client.send({"type": "profile.start", "id": "ps", "name": "work"})
        started = await client.wait_for(lambda e: e.get("type") == "profile.started")
        assert started["name"] == "work" and started["agents_started"] == 2

        await client.send({"type": "profile.stop", "id": "px", "name": "work"})
        stopped = await client.wait_for(lambda e: e.get("type") == "profile.stopped")
        assert stopped["name"] == "work" and stopped["agents_stopped"] == 2
    finally:
        await client.close()
