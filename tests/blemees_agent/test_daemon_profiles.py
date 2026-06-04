"""Daemon-level integration for profiles + supervisor (#17).

Drives a live daemon configured with a default profile plus a named one,
asserting: open under a named profile, profile.list with running/session
counts, lazy start, profile.start/stop, unknown-profile error, and two
profiles' processes running concurrently.
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


@pytest.fixture
async def profile_daemon():
    socket_path = short_socket_path("blemeesd-prof")
    with socket_cleanup(socket_path):
        cfg = Config(
            socket_path=str(socket_path),
            agent_command=sys.executable,
            agent_args=[FAKE_ACP],
            profiles={
                "alt": {"agent_command": sys.executable, "agent_args": [FAKE_ACP], "model": "x"},
            },
            idle_timeout_s=60,
            max_concurrent_sessions=8,
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


async def _connect(socket_path: str) -> _StreamClient:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    client = _StreamClient(reader, writer)
    await client.send({"type": "hello", "client": "t/1", "protocol": PROTOCOL_VERSION})
    ack = await client.recv()
    assert ack["type"] == "hello_ack"
    return client, ack


async def _drive_turn(client, sid, profile=None):
    frame = {"type": "session.open", "id": "o", "session_id": sid, "options": {}}
    if profile:
        frame["profile"] = profile
    await client.send(frame)
    opened = await client.wait_for(lambda e: e.get("type") == "session.opened")
    await client.send({"type": "session.prompt", "session_id": sid, "prompt": "say pong"})
    await client.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
    return opened


async def test_hello_ack_lists_profiles(profile_daemon):
    client, ack = await _connect(profile_daemon)
    try:
        assert set(ack["profiles"]) == {"default", "alt"}
    finally:
        await client.close()


async def test_open_under_named_profile(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        opened = await _drive_turn(client, "s-alt", profile="alt")
        assert opened["profile"] == "alt"
    finally:
        await client.close()


async def test_unknown_profile_rejected(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        await client.send(
            {
                "type": "session.open",
                "id": "o",
                "session_id": "s",
                "profile": "ghost",
                "options": {},
            }
        )
        err = await client.wait_for(lambda e: e.get("type") == "error")
        assert err["code"] == "profile_unknown"
    finally:
        await client.close()


async def test_profile_list_reports_running_and_session_counts(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        # Lazily start by opening under each profile.
        await _drive_turn(client, "s-default", profile="default")
        await _drive_turn(client, "s-alt", profile="alt")

        await client.send({"type": "profile.list", "id": "pl"})
        reply = await client.wait_for(lambda e: e.get("type") == "profiles")
        rows = {r["name"]: r for r in reply["profiles"]}
        assert rows["default"]["running"] is True and rows["default"]["sessions"] >= 1
        assert rows["alt"]["running"] is True and rows["alt"]["sessions"] >= 1
        # Two distinct profiles' processes are up concurrently.
        assert rows["default"]["sessions"] >= 1 and rows["alt"]["sessions"] >= 1
    finally:
        await client.close()


async def test_profile_start_and_stop(profile_daemon):
    client, _ = await _connect(profile_daemon)
    try:
        await client.send({"type": "profile.start", "id": "ps", "name": "alt"})
        started = await client.wait_for(lambda e: e.get("type") == "profile.started")
        assert started["name"] == "alt" and started["pid"]

        await client.send({"type": "profile.stop", "id": "px", "name": "alt"})
        stopped = await client.wait_for(lambda e: e.get("type") == "profile.stopped")
        assert stopped["name"] == "alt" and stopped["was_running"] is True
    finally:
        await client.close()
