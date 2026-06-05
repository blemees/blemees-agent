"""Daemon-level permission relay + policy over the wire (#20)."""

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
async def perm_daemon():
    socket_path = short_socket_path("blemeesd-perm")
    with socket_cleanup(socket_path):
        agent = {"agent_command": sys.executable, "agent_args": [FAKE_ACP]}
        cfg = Config(
            socket_path=str(socket_path),
            agent_command=sys.executable,
            agent_args=[FAKE_ACP],
            profiles={
                # default profile uses the built-in relay+stall policy
                "allowp": {**agent, "permission_policy": {"mode": "allow"}},
                "denyp": {**agent, "permission_policy": {"mode": "deny"}},
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
    c = _StreamClient(reader, writer)
    await c.send({"type": "hello", "client": "t", "protocol": PROTOCOL_VERSION})
    assert (await c.recv())["type"] == "hello_ack"
    return c


async def _open(c, sid, profile=None):
    frame = {"type": "session.open", "id": "o", "session_id": sid, "options": {}}
    if profile:
        frame["profile"] = profile
    await c.send(frame)
    await c.wait_for(lambda e: e.get("type") == "session.opened")


def _last_text(frames) -> str:
    return "".join(
        f["update"]["content"]["text"]
        for f in frames
        if f.get("type") == "session.update"
        and f["update"].get("sessionUpdate") == "agent_message_chunk"
    )


async def test_relay_to_owner_then_allow(perm_daemon):
    c = await _connect(perm_daemon)
    try:
        await _open(c, "s-relay")  # default profile → relay
        await c.send({"type": "session.prompt", "session_id": "s-relay", "prompt": "permit please"})
        req = await c.wait_for(
            lambda e: e.get("type") == "session.request_permission", timeout=30.0
        )
        assert any(o["kind"] == "allow_once" for o in req["options"])
        await c.send(
            {
                "type": "session.permission_response",
                "session_id": "s-relay",
                "request_id": req["request_id"],
                "outcome": "selected",
                "option_id": "allow",
            }
        )
        frames = await c.wait_for(
            lambda e: e.get("type") == "session.result", collect=True, timeout=30.0
        )
    finally:
        await c.close()
    assert next(f for f in frames if f["type"] == "session.result")["stop_reason"] == "end_turn"
    assert "perm:allow" in _last_text(frames)


async def test_allow_policy_auto_no_relay(perm_daemon):
    c = await _connect(perm_daemon)
    try:
        await _open(c, "s-allow", profile="allowp")
        await c.send({"type": "session.prompt", "session_id": "s-allow", "prompt": "permit please"})
        frames = await c.wait_for(
            lambda e: e.get("type") == "session.result", collect=True, timeout=30.0
        )
    finally:
        await c.close()
    # No relay frame was sent; the agent received an allow decision.
    assert not any(f.get("type") == "session.request_permission" for f in frames)
    assert "perm:allow" in _last_text(frames)


async def test_deny_policy_auto_no_relay(perm_daemon):
    c = await _connect(perm_daemon)
    try:
        await _open(c, "s-deny", profile="denyp")
        await c.send({"type": "session.prompt", "session_id": "s-deny", "prompt": "permit please"})
        frames = await c.wait_for(
            lambda e: e.get("type") == "session.result", collect=True, timeout=30.0
        )
    finally:
        await c.close()
    assert not any(f.get("type") == "session.request_permission" for f in frames)
    assert "perm:deny" in _last_text(frames)
