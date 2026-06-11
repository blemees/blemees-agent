"""Session attention state machine + notify triggers (#24).

Deterministic, in-process tests: construct a Session directly, capture its
emitted frames via a writer, and exercise the entry/exit edges and the three
trigger sources with a recording NotifyService stand-in.
"""

from __future__ import annotations

import pytest

from blemees_agent.protocol import OpenMessage
from blemees_agent.session import Session

pytestmark = pytest.mark.asyncio


class _RecordingNotify:
    """Stands in for NotifyService — records fire/clear without any sinks."""

    def __init__(self) -> None:
        self.fired: list[dict] = []
        self.cleared: list[str] = []

    async def fire(self, *, reason, profile, session_id, detail):
        self.fired.append(
            {"reason": reason, "profile": profile, "session_id": session_id, "detail": detail}
        )

    def clear(self, session_id):
        self.cleared.append(session_id)


def _open_msg(session: str = "s1") -> OpenMessage:
    return OpenMessage(id=None, session_id=session, options={}, resume=False)


def _session(*, connection_id=None, profile="claude") -> tuple[Session, _RecordingNotify, list]:
    notify = _RecordingNotify()
    frames: list[dict] = []

    async def writer(frame: dict) -> None:
        frames.append(frame)

    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None, connection_id=connection_id)
    sess.notify = notify
    sess.profile_name = profile
    sess._writer = writer
    return sess, notify, frames


def _types(frames) -> list[str]:
    return [f["type"] for f in frames]


async def test_enter_attention_detached_fires_and_emits():
    sess, notify, frames = _session(connection_id=None)
    await sess.enter_attention("agent_crashed", "boom")
    assert sess.needs_attention is True
    assert sess.attention_reason == "agent_crashed"
    assert "session.needs_attention" in _types(frames)
    assert notify.fired == [
        {"reason": "agent_crashed", "profile": "claude", "session_id": "s1", "detail": "boom"}
    ]


async def test_enter_attention_attached_is_noop():
    # Owner present → no needs_attention, no webhook (they see it live).
    sess, notify, frames = _session(connection_id=7)
    await sess.enter_attention("agent_crashed", "boom")
    assert sess.needs_attention is False
    assert notify.fired == []
    assert "session.needs_attention" not in _types(frames)


async def test_enter_attention_is_idempotent_on_same_reason():
    sess, notify, _ = _session(connection_id=None)
    await sess.enter_attention("agent_crashed", "boom")
    await sess.enter_attention("agent_crashed", "boom again")
    assert len(notify.fired) == 1  # one webhook per real edge


async def test_clear_attention_emits_and_clears_outstanding():
    sess, notify, frames = _session(connection_id=None)
    await sess.enter_attention("agent_crashed", "boom")
    frames.clear()
    await sess.clear_attention()
    assert sess.needs_attention is False
    assert sess.attention_reason is None
    assert _types(frames) == ["session.attention_cleared"]
    assert notify.cleared == ["s1"]


async def test_clear_attention_noop_when_not_flagged():
    sess, notify, frames = _session(connection_id=None)
    await sess.clear_attention()
    assert frames == []
    assert notify.cleared == []


async def test_crash_error_frame_triggers_attention_when_detached():
    sess, notify, _ = _session(connection_id=None)
    await sess.on_event({"type": "session.error", "code": "agent_crashed", "message": "died"})
    assert sess.needs_attention is True
    assert notify.fired[0]["reason"] == "agent_crashed"
    assert notify.fired[0]["detail"] == "died"


async def test_auth_error_frame_triggers_attention_when_detached():
    sess, notify, _ = _session(connection_id=None)
    await sess.on_event({"type": "session.error", "code": "auth_required", "message": "expired"})
    assert sess.attention_reason == "auth_required"
    assert notify.fired[0]["reason"] == "auth_required"


async def test_crash_error_frame_silent_when_attached():
    sess, notify, _ = _session(connection_id=7)
    await sess.on_event({"type": "session.error", "code": "agent_crashed", "message": "died"})
    assert sess.needs_attention is False
    assert notify.fired == []


async def test_turn_result_does_not_trigger_attention():
    # Turn-complete is deliberately NOT a trigger (§6).
    sess, notify, _ = _session(connection_id=None)
    await sess.on_event({"type": "session.result", "stop_reason": "end_turn"})
    assert sess.needs_attention is False
    assert notify.fired == []


async def test_attach_clears_attention():
    sess, notify, _ = _session(connection_id=None)
    await sess.enter_attention("agent_crashed", "boom")

    captured: list[dict] = []

    async def writer(frame: dict) -> None:
        captured.append(frame)

    await sess.attach(connection_id=9, writer=writer)
    assert sess.needs_attention is False
    assert notify.cleared == ["s1"]
    assert "session.attention_cleared" in [f["type"] for f in captured]


# ---- per-session attention policy (#51) ------------------------------


async def test_turn_complete_detached_is_silent_by_default():
    sess, notify, frames = _session(connection_id=None)
    await sess.on_event({"type": "session.result", "stop_reason": "end_turn"})
    assert sess.needs_attention is False
    assert notify.fired == []
    assert "session.needs_attention" not in _types(frames)


async def test_turn_complete_fires_when_armed_and_detached():
    sess, notify, frames = _session(connection_id=None)
    sess.attention_triggers = {"turn_complete"}
    await sess.on_event({"type": "session.result", "stop_reason": "end_turn"})
    assert sess.needs_attention is True
    assert sess.attention_reason == "turn_complete"
    assert "session.needs_attention" in _types(frames)
    assert notify.fired and notify.fired[0]["reason"] == "turn_complete"
    assert "end_turn" in notify.fired[0]["detail"]


async def test_turn_complete_armed_but_attached_is_noop():
    sess, notify, frames = _session(connection_id=7)
    sess.attention_triggers = {"turn_complete"}
    await sess.on_event({"type": "session.result", "stop_reason": "end_turn"})
    assert sess.needs_attention is False
    assert notify.fired == []


async def test_disarmed_blocked_trigger_is_silent():
    # The policy can also disarm a default trigger — an empty set means
    # nothing pushes for this session (#51).
    sess, notify, frames = _session(connection_id=None)
    sess.attention_triggers = set()
    await sess.enter_attention("agent_crashed", "boom")
    assert sess.needs_attention is False
    assert notify.fired == []
