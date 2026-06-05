"""Daemon-level registry-backed session.list / session.info, incl. restart (#21)."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import _StreamClient, short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")


@contextlib.asynccontextmanager
async def _daemon(state_dir: Path, socket_path: Path):
    cfg = Config(
        socket_path=str(socket_path),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        state_dir=str(state_dir),
        idle_timeout_s=60,
        max_concurrent_sessions=8,
    )
    daemon = Daemon(cfg, configure("error"))
    await daemon.start()
    serve = asyncio.create_task(daemon.serve_forever())
    try:
        yield
    finally:
        daemon.request_shutdown()
        try:
            await asyncio.wait_for(serve, timeout=5.0)
        except TimeoutError:
            serve.cancel()


async def _connect(socket_path: Path) -> _StreamClient:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    c = _StreamClient(reader, writer)
    await c.send({"type": "hello", "client": "t", "protocol": PROTOCOL_VERSION})
    assert (await c.recv())["type"] == "hello_ack"
    return c


async def _drive(c, sid):
    await c.send({"type": "session.open", "id": "o", "session_id": sid, "options": {}})
    await c.wait_for(lambda e: e.get("type") == "session.opened")
    await c.send({"type": "session.prompt", "session_id": sid, "prompt": "say pong"})
    await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


async def test_list_and_info_for_live_session(state_dir):
    sock = short_socket_path("blemeesd-reg")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await _drive(c, "s-live")
                await c.send({"type": "session.list", "id": "l"})
                reply = await c.wait_for(lambda e: e.get("type") == "sessions")
                rows = {r["session_id"]: r for r in reply["sessions"]}
                assert "s-live" in rows
                assert rows["s-live"]["attached"] is True
                assert rows["s-live"]["running"] is True
                assert rows["s-live"]["profile"] == "default"

                await c.send({"type": "session.info", "id": "i", "session_id": "s-live"})
                info = await c.wait_for(lambda e: e.get("type") == "session.info_reply")
                assert info["profile"] == "default" and info["agent"] == "default"
                assert info["attached"] is True
                assert info["needs_attention"] is False
                assert info["turns"] >= 1
            finally:
                await c.close()
        # registry.json was written.
        assert (state_dir / "registry.json").is_file()


async def test_registry_survives_restart(state_dir):
    sock1 = short_socket_path("blemeesd-reg1")
    sock2 = short_socket_path("blemeesd-reg2")
    with socket_cleanup(sock1), socket_cleanup(sock2):
        # First daemon: open + drive a session, then shut down.
        async with _daemon(state_dir, sock1):
            c = await _connect(sock1)
            try:
                await _drive(c, "s-keep")
            finally:
                await c.close()

        # Second daemon over the same state dir: the session is known (cold).
        async with _daemon(state_dir, sock2):
            c = await _connect(sock2)
            try:
                await c.send({"type": "session.list", "id": "l"})
                reply = await c.wait_for(lambda e: e.get("type") == "sessions")
                rows = {r["session_id"]: r for r in reply["sessions"]}
                assert "s-keep" in rows
                cold = rows["s-keep"]
                assert cold["attached"] is False
                assert cold["running"] is False
                assert cold["profile"] == "default"

                await c.send({"type": "session.info", "id": "i", "session_id": "s-keep"})
                info = await c.wait_for(lambda e: e.get("type") == "session.info_reply")
                assert info["attached"] is False
                assert info["profile"] == "default"
            finally:
                await c.close()


async def test_close_delete_removes_from_registry(state_dir):
    sock = short_socket_path("blemeesd-regdel")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await _drive(c, "s-del")
                await c.send(
                    {"type": "session.close", "id": "x", "session_id": "s-del", "delete": True}
                )
                await c.wait_for(lambda e: e.get("type") == "session.closed")
                await c.send({"type": "session.list", "id": "l"})
                reply = await c.wait_for(lambda e: e.get("type") == "sessions")
                assert all(r["session_id"] != "s-del" for r in reply["sessions"])
            finally:
                await c.close()
