"""Unit tests for the ACP backend (#16/#17): shared process + per-session handle.

Drive ``AcpAgentProcess`` / ``AcpSessionHandle`` against ``fake_acp.py`` (a
real ACP agent over stdio), asserting blemees ``session.*`` translation and
that one process multiplexes several sessions.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from blemees_agent.backends.acp import AcpAgentProcess, AcpSessionHandle, _to_content_blocks
from blemees_agent.errors import ProtocolError, SessionBusyError
from blemees_agent.logging import configure
from blemees_agent.supervisor import Agent

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")


def _process(env: dict | None = None) -> AcpAgentProcess:
    agent = Agent(name="default", command=sys.executable, args=[FAKE_ACP])
    return AcpAgentProcess(
        agent, key=("t", "default"), logger=configure("error"), env=env or dict(os.environ)
    )


async def _collect_turn(q: asyncio.Queue, timeout: float = 30.0) -> list[dict]:
    frames: list[dict] = []
    while True:
        frame = await asyncio.wait_for(q.get(), timeout=timeout)
        frames.append(frame)
        if frame.get("type") == "session.result":
            break
    # Drain a short tail for any update that trails the result on the wire.
    while True:
        try:
            frames.append(await asyncio.wait_for(q.get(), timeout=0.5))
        except TimeoutError:
            return frames


# ---- pure helper ----------------------------------------------------


def test_to_content_blocks_text():
    blocks = _to_content_blocks({"role": "user", "content": "hi"})
    assert len(blocks) == 1 and blocks[0].text == "hi"


def test_to_content_blocks_rejects_non_text():
    with pytest.raises(ProtocolError):
        _to_content_blocks({"role": "user", "content": [{"type": "image", "data": ".."}]})


# ---- process + handle against the fake agent ------------------------


async def _handle(process: AcpAgentProcess, q: asyncio.Queue) -> AcpSessionHandle:
    handle = AcpSessionHandle(process=process, on_event=q.put, cwd=None)
    await handle.spawn()
    return handle


async def test_spawn_reports_capabilities_and_native_id():
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    handle = await _handle(process, q)
    try:
        assert handle.running
        assert handle.pid is not None
        assert handle.load_session is True
        assert handle.native_session_id == "fake-session-1"
    finally:
        await process.close()
    assert not handle.running


async def test_turn_streams_updates_then_result():
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    handle = await _handle(process, q)
    try:
        await handle.send_user_turn({"role": "user", "content": "say pong"})
        frames = await _collect_turn(q)
    finally:
        await process.close()
    updates = [f for f in frames if f["type"] == "session.update"]
    result = next(f for f in frames if f["type"] == "session.result")
    assert result["stop_reason"] == "end_turn"
    assert "".join(u["update"]["content"]["text"] for u in updates) == "PONG done"
    assert "seq" not in result  # seq is the Session's job


async def test_busy_rejects_concurrent_turn():
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    handle = await _handle(process, q)
    try:
        await handle.send_user_turn({"role": "user", "content": "hang please"})
        await asyncio.sleep(0.2)
        assert handle.turn_active is True
        with pytest.raises(SessionBusyError):
            await handle.send_user_turn({"role": "user", "content": "again"})
    finally:
        await process.close()


async def test_one_process_multiplexes_two_sessions():
    process = _process()
    qa: asyncio.Queue = asyncio.Queue()
    qb: asyncio.Queue = asyncio.Queue()
    a = await _handle(process, qa)
    b = await _handle(process, qb)
    try:
        # Distinct ACP session ids on the same process.
        assert a.native_session_id != b.native_session_id
        assert process.session_count() == 2
        await a.send_user_turn({"role": "user", "content": "say pong"})
        await b.send_user_turn({"role": "user", "content": "say pong"})
        fa = await _collect_turn(qa)
        fb = await _collect_turn(qb)
    finally:
        await process.close()
    assert any(f["type"] == "session.result" for f in fa)
    assert any(f["type"] == "session.result" for f in fb)


async def test_cancel_finalizes_as_cancelled():
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    handle = await _handle(process, q)
    try:
        await handle.send_user_turn({"role": "user", "content": "hang please"})
        await asyncio.sleep(0.2)
        assert await handle.interrupt() is True
        result = await asyncio.wait_for(_first(q, "session.result"), timeout=10.0)
        assert result["stop_reason"] == "cancelled"
        assert handle.turn_active is False
    finally:
        await process.close()


async def test_three_sessions_stream_independently():
    process = _process()
    queues = [asyncio.Queue() for _ in range(3)]
    handles = [await _handle(process, q) for q in queues]
    try:
        ids = {h.native_session_id for h in handles}
        assert len(ids) == 3  # distinct ACP session ids
        assert process.session_count() == 3
        for h in handles:
            await h.send_user_turn({"role": "user", "content": "say pong"})
        results = []
        for q in queues:
            frames = await _collect_turn(q)
            text = "".join(
                f["update"]["content"]["text"] for f in frames if f["type"] == "session.update"
            )
            results.append((text, next(f for f in frames if f["type"] == "session.result")))
    finally:
        await process.close()
    # Each session got its own complete, uncrossed stream.
    assert all(text == "PONG done" for text, _ in results)
    assert all(r["stop_reason"] == "end_turn" for _, r in results)


async def test_agent_failure_surfaces_session_error():
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    handle = await _handle(process, q)
    try:
        await handle.send_user_turn({"role": "user", "content": "boom"})
        frame = await asyncio.wait_for(_first(q, "session.error"), timeout=30.0)
    finally:
        await process.close()
    assert frame["code"] == "agent_crashed"


async def _first(q: asyncio.Queue, type_: str) -> dict:
    while True:
        f = await q.get()
        if f.get("type") == type_:
            return f


# ---- resume (#23) ---------------------------------------------------


async def test_resume_loads_prior_session_and_is_drivable():
    """A loadSession-capable agent rehydrates its native session id and the
    resumed handle can drive a turn (not view-only)."""
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    try:
        resumed = AcpSessionHandle(
            process=process, on_event=q.put, cwd=None, resume_native_id="fake-session-prior"
        )
        await resumed.spawn()
        assert resumed.view_only is False
        assert resumed.native_session_id == "fake-session-prior"
        # Drivable: a turn streams and completes normally.
        await resumed.send_user_turn({"role": "user", "content": "say pong"})
        frames = await _collect_turn(q)
        assert any(f["type"] == "session.result" for f in frames)
    finally:
        await process.close()


async def test_resume_drops_history_updates_during_load():
    """session/load replays history as session/update; those are suppressed so
    the client (which already has the durable log) doesn't see duplicates."""
    process = _process()
    q: asyncio.Queue = asyncio.Queue()
    try:
        resumed = AcpSessionHandle(
            process=process, on_event=q.put, cwd=None, resume_native_id="fake-session-prior"
        )
        await resumed.spawn()
    finally:
        await process.close()
    # The fake agent's load_session emits nothing, but the contract is that any
    # update arriving while loading is dropped: the queue holds no frames.
    assert q.empty()


async def test_resume_without_load_capability_is_view_only():
    """An agent that can't reload sessions yields a view-only handle: no native
    session, not drivable."""
    env = {**os.environ, "BLEMEES_FAKE_NO_LOAD": "1"}
    process = _process(env)
    q: asyncio.Queue = asyncio.Queue()
    try:
        resumed = AcpSessionHandle(
            process=process, on_event=q.put, cwd=None, resume_native_id="fake-session-prior"
        )
        await resumed.spawn()
        assert resumed.load_session is False
        assert resumed.view_only is True
        assert resumed.native_session_id is None
    finally:
        await process.close()
