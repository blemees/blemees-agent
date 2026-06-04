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
from blemees_agent.supervisor import Profile

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")


def _process() -> AcpAgentProcess:
    profile = Profile(name="t", command=sys.executable, args=[FAKE_ACP])
    return AcpAgentProcess(profile, logger=configure("error"), env=dict(os.environ))


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
