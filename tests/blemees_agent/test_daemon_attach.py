"""Daemon-level owner/viewer attach, takeover, detach, replay (#19)."""

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
SID = "33333333-3333-3333-3333-333333333333"


@pytest.fixture
async def acp_daemon():
    socket_path = short_socket_path("blemeesd-attach")
    with socket_cleanup(socket_path):
        cfg = Config(
            socket_path=str(socket_path),
            agent_command=sys.executable,
            agent_args=[FAKE_ACP],
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


async def _open(c: _StreamClient, sid: str = SID) -> None:
    await c.send({"type": "session.open", "id": "o", "session_id": sid, "options": {}})
    await c.wait_for(lambda e: e.get("type") == "session.opened")


async def test_viewer_receives_fanout_and_cannot_drive(acp_daemon):
    owner = await _connect(acp_daemon)
    viewer = await _connect(acp_daemon)
    try:
        await _open(owner)
        await viewer.send({"type": "session.attach", "id": "a", "session_id": SID, "as": "viewer"})
        attached = await viewer.wait_for(lambda e: e.get("type") == "session.attached")
        assert attached["role"] == "viewer"

        # Owner drives a turn; the viewer sees the same fan-out.
        await owner.send({"type": "session.prompt", "session_id": SID, "prompt": "say pong"})
        v_result = await viewer.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
        assert v_result["stop_reason"] == "end_turn"

        # Viewer cannot drive: prompt against a session it doesn't own → unknown.
        await viewer.send({"type": "session.prompt", "session_id": SID, "prompt": "hi"})
        err = await viewer.wait_for(lambda e: e.get("type") == "error")
        assert err["code"] == "session_unknown"
    finally:
        await owner.close()
        await viewer.close()


async def test_owner_takeover_notifies_prior_owner(acp_daemon):
    first = await _connect(acp_daemon)
    second = await _connect(acp_daemon)
    try:
        await _open(first)
        await second.send({"type": "session.attach", "id": "a", "session_id": SID, "as": "owner"})
        attached = await second.wait_for(lambda e: e.get("type") == "session.attached")
        assert attached["role"] == "owner"
        # The prior owner is told it was taken.
        taken = await first.wait_for(lambda e: e.get("type") == "session.taken")
        assert taken["session_id"] == SID

        # The new owner can drive.
        await second.send({"type": "session.prompt", "session_id": SID, "prompt": "say pong"})
        result = await second.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
        assert result["stop_reason"] == "end_turn"
    finally:
        await first.close()
        await second.close()


async def test_detach_leaves_session_running_and_reattach_replays(acp_daemon):
    c1 = await _connect(acp_daemon)
    try:
        await _open(c1)
        await c1.send({"type": "session.prompt", "session_id": SID, "prompt": "say pong"})
        result = await c1.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
        last_seq = result["seq"]

        await c1.send({"type": "session.detach", "id": "d", "session_id": SID})
        detached = await c1.wait_for(lambda e: e.get("type") == "session.detached")
        assert detached["was_attached"] is True
    finally:
        await c1.close()

    # A fresh connection re-attaches as owner and replays from seq 0.
    c2 = await _connect(acp_daemon)
    try:
        await c2.send(
            {
                "type": "session.attach",
                "id": "a",
                "session_id": SID,
                "as": "owner",
                "last_seen_seq": 0,
            }
        )
        attached = await c2.wait_for(lambda e: e.get("type") == "session.attached")
        assert attached["last_seq"] >= last_seq
        # Replayed frames include the prior turn's result.
        replayed = await c2.wait_for(
            lambda e: e.get("type") == "session.result" and e.get("seq") == last_seq, timeout=5.0
        )
        assert replayed["stop_reason"] == "end_turn"
    finally:
        await c2.close()


async def test_attach_unknown_session_errors(acp_daemon):
    c = await _connect(acp_daemon)
    try:
        await c.send({"type": "session.attach", "id": "a", "session_id": "ghost", "as": "viewer"})
        err = await c.wait_for(lambda e: e.get("type") == "error")
        assert err["code"] == "session_unknown"
    finally:
        await c.close()
