"""Daemon-level registry-backed session.list / session.info, incl. restart (#21)."""

from __future__ import annotations

import asyncio
import contextlib
import json
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


async def test_event_log_replays_across_restart(state_dir):
    """#22: the per-session event log under <state_dir>/sessions survives a
    restart, so a reattaching client replays prior turns from disk."""
    sock1 = short_socket_path("blemeesd-log1")
    sock2 = short_socket_path("blemeesd-log2")
    with socket_cleanup(sock1), socket_cleanup(sock2):
        async with _daemon(state_dir, sock1):
            c = await _connect(sock1)
            try:
                await c.send(
                    {"type": "session.open", "id": "o", "session_id": "s-log", "options": {}}
                )
                await c.wait_for(lambda e: e.get("type") == "session.opened")
                await c.send(
                    {"type": "session.prompt", "session_id": "s-log", "prompt": "say pong"}
                )
                result = await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
                last_seq = result["seq"]
            finally:
                await c.close()
        # The event log was written under the state dir (no explicit event_log_dir).
        assert (state_dir / "sessions" / "s-log.jsonl").is_file()

        async with _daemon(state_dir, sock2):
            c = await _connect(sock2)
            try:
                await c.send(
                    {
                        "type": "session.open",
                        "id": "o",
                        "session_id": "s-log",
                        "resume": True,
                        "last_seen_seq": 0,
                        "options": {},
                    }
                )
                opened = await c.wait_for(lambda e: e.get("type") == "session.opened")
                assert opened["last_seq"] >= last_seq  # seeded from the durable log
                # The prior run's turn-end result is replayed from disk.
                replayed = await c.wait_for(
                    lambda e: e.get("type") == "session.result" and e.get("seq") == last_seq,
                    timeout=10.0,
                )
                assert replayed["stop_reason"] == "end_turn"
            finally:
                await c.close()


async def test_resume_rehydrates_loadsession_agent_across_restart(state_dir):
    """#23: a loadSession-capable session reloads its agent context across a
    restart — the prior native session id (from the persistent registry) is
    rehydrated via session/load, the session is not view-only, and a new turn
    drives to completion."""
    sock1 = short_socket_path("blemeesd-res1")
    sock2 = short_socket_path("blemeesd-res2")
    with socket_cleanup(sock1), socket_cleanup(sock2):
        # First daemon: drive a session so the agent's native id is persisted.
        async with _daemon(state_dir, sock1):
            c = await _connect(sock1)
            try:
                await _drive(c, "s-resume")
            finally:
                await c.close()
        # The agent's native session id was persisted for resume.
        rec = json.loads((state_dir / "registry.json").read_text())["sessions"][0]
        assert rec["session_id"] == "s-resume"
        assert rec["native_session_id"] == "fake-session-1"

        # Second daemon over the same state dir: resume reloads that session.
        async with _daemon(state_dir, sock2):
            c = await _connect(sock2)
            try:
                await c.send(
                    {
                        "type": "session.open",
                        "id": "o",
                        "session_id": "s-resume",
                        "resume": True,
                        "options": {},
                    }
                )
                opened = await c.wait_for(lambda e: e.get("type") == "session.opened")
                assert opened["view_only"] is False
                # Reloaded the prior native session rather than minting a new one.
                assert opened.get("native_session_id") == "fake-session-1"
                # Drivable: a fresh turn streams a result to completion.
                await c.send(
                    {"type": "session.prompt", "session_id": "s-resume", "prompt": "say pong"}
                )
                result = await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
                assert result["stop_reason"] == "end_turn"
            finally:
                await c.close()


async def test_resume_without_loadsession_is_view_only_across_restart(state_dir, monkeypatch):
    """#23: resuming against an agent that can't reload yields a view-only
    session — flagged in session.opened / session.info / session.list, and a
    prompt is rejected with the ``view_only`` error rather than starting a
    blank turn."""
    monkeypatch.setenv("BLEMEES_FAKE_NO_LOAD", "1")
    sock1 = short_socket_path("blemeesd-vo1")
    sock2 = short_socket_path("blemeesd-vo2")
    with socket_cleanup(sock1), socket_cleanup(sock2):
        async with _daemon(state_dir, sock1):
            c = await _connect(sock1)
            try:
                await _drive(c, "s-vonly")
            finally:
                await c.close()

        async with _daemon(state_dir, sock2):
            c = await _connect(sock2)
            try:
                await c.send(
                    {
                        "type": "session.open",
                        "id": "o",
                        "session_id": "s-vonly",
                        "resume": True,
                        "options": {},
                    }
                )
                opened = await c.wait_for(lambda e: e.get("type") == "session.opened")
                assert opened["view_only"] is True
                # No live ACP session was reloaded, so the opened frame must not
                # advertise a native id even though one survives in the sidecar.
                assert "native_session_id" not in opened

                # session.info and session.list both report view-only.
                await c.send({"type": "session.info", "id": "i", "session_id": "s-vonly"})
                info = await c.wait_for(lambda e: e.get("type") == "session.info_reply")
                assert info["view_only"] is True

                await c.send({"type": "session.list", "id": "l"})
                listing = await c.wait_for(lambda e: e.get("type") == "sessions")
                row = next(r for r in listing["sessions"] if r["session_id"] == "s-vonly")
                assert row["view_only"] is True

                # Driving it is rejected up front with the view_only error.
                await c.send(
                    {"type": "session.prompt", "session_id": "s-vonly", "prompt": "say pong"}
                )
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "view_only"
                assert err["session_id"] == "s-vonly"
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
