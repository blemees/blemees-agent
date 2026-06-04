"""Unit tests for the ACP client backend (#16) using the fake ACP agent stub.

These drive ``AcpBackend`` against ``fake_acp.py`` (a real ACP agent spoken
over stdio via the SDK), asserting the blemees ``session.*`` translation:
streamed ``session.update`` frames, a turn-ending ``session.result``,
interrupt → ``cancelled``, and agent failure → ``session.error``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from blemees_agent.backends.acp import AcpBackend, _to_content_blocks
from blemees_agent.errors import ProtocolError, SessionBusyError
from blemees_agent.logging import configure

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")


def _make_backend(queue: asyncio.Queue, *, session_id: str = "s1") -> AcpBackend:
    return AcpBackend(
        session_id=session_id,
        command=sys.executable,
        args=[FAKE_ACP],
        cwd=None,
        on_event=queue.put,
        logger=configure("error"),
    )


async def _drain_until(queue: asyncio.Queue, *types: str, timeout: float = 30.0) -> list[dict]:
    """Collect frames until one of ``types`` is seen (inclusive)."""
    frames: list[dict] = []
    while True:
        frame = await asyncio.wait_for(queue.get(), timeout=timeout)
        frames.append(frame)
        if frame.get("type") in types:
            return frames


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_to_content_blocks_text():
    blocks = _to_content_blocks({"role": "user", "content": "hello"})
    assert len(blocks) == 1 and blocks[0].text == "hello"


def test_to_content_blocks_array():
    blocks = _to_content_blocks(
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    )
    assert [b.text for b in blocks] == ["a", "b"]


def test_to_content_blocks_rejects_non_text():
    with pytest.raises(ProtocolError):
        _to_content_blocks(
            {"role": "user", "content": [{"type": "image", "data": "..", "mimeType": "image/png"}]}
        )


# ---------------------------------------------------------------------------
# End-to-end against the fake ACP agent
# ---------------------------------------------------------------------------


async def test_spawn_initializes_and_reports_capabilities():
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        assert backend.running
        assert backend.pid is not None
        # fake_acp advertises load_session and a fixed native session id.
        assert backend.load_session is True
        assert backend.native_session_id == "fake-session-1"
    finally:
        await backend.close()
    assert not backend.running


async def test_turn_streams_updates_then_result():
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        await backend.send_user_turn({"role": "user", "content": "say pong"})
        frames = await _drain_until(queue, "session.result")
    finally:
        await backend.close()

    updates = [f for f in frames if f["type"] == "session.update"]
    results = [f for f in frames if f["type"] == "session.result"]
    assert len(results) == 1
    assert results[0]["stop_reason"] == "end_turn"
    assert len(updates) == 2
    # Verbatim ACP update payload carried under "update".
    first = updates[0]["update"]
    assert first["sessionUpdate"] == "agent_message_chunk"
    assert first["content"]["text"] == "PONG"
    # Backend does NOT assign seq — that's the Session's job (#19 path).
    assert "seq" not in results[0]


async def test_second_turn_after_result_works():
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        await backend.send_user_turn({"role": "user", "content": "one"})
        await _drain_until(queue, "session.result")
        assert backend.turn_active is False
        await backend.send_user_turn({"role": "user", "content": "two"})
        frames = await _drain_until(queue, "session.result")
    finally:
        await backend.close()
    assert any(f["type"] == "session.result" for f in frames)


async def test_busy_rejects_concurrent_turn():
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        await backend.send_user_turn({"role": "user", "content": "hang please"})
        # Turn is in flight (the agent is sleeping); a second turn is rejected.
        await asyncio.sleep(0.2)
        with pytest.raises(SessionBusyError):
            await backend.send_user_turn({"role": "user", "content": "again"})
    finally:
        await backend.close()


async def test_interrupt_signals_in_flight_turn():
    # The full cancel→`cancelled` round-trip (SDK cancel propagation to the
    # agent) is exercised in #18; here we just assert interrupt() reports it
    # acted on an in-flight turn.
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        await backend.send_user_turn({"role": "user", "content": "hang please"})
        await asyncio.sleep(0.3)
        assert await backend.interrupt() is True
    finally:
        await backend.close()


async def test_interrupt_when_idle_returns_false():
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        assert await backend.interrupt() is False
    finally:
        await backend.close()


async def test_agent_failure_surfaces_session_error():
    queue: asyncio.Queue = asyncio.Queue()
    backend = _make_backend(queue)
    await backend.spawn()
    try:
        await backend.send_user_turn({"role": "user", "content": "boom"})
        frames = await _drain_until(queue, "session.error", "session.result")
    finally:
        await backend.close()
    err = next(f for f in frames if f["type"] == "session.error")
    assert err["code"] == "agent_crashed"
    assert backend.turn_active is False
