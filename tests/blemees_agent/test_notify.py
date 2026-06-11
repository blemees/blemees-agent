"""Unit tests for the notify service (#24): NotifyService + WebhookSink."""

from __future__ import annotations

import asyncio

import pytest

from blemees_agent.logging import configure
from blemees_agent.notify import (
    AGENT_CRASHED,
    PERMISSION_PENDING,
    TEST,
    Notification,
    NotifyService,
    WebhookSink,
)

pytestmark = pytest.mark.asyncio

LOG = configure("error")


class _RecordingSink:
    def __init__(self) -> None:
        self.seen: list[Notification] = []

    async def emit(self, notification: Notification) -> None:
        self.seen.append(notification)


class _BoomSink:
    async def emit(self, notification: Notification) -> None:
        raise RuntimeError("sink down")


def _svc(*sinks, now_ms=lambda: 1000) -> NotifyService:
    return NotifyService(sinks=list(sinks), logger=LOG, now_ms=now_ms)


async def test_notification_payload_shape():
    n = Notification(reason=PERMISSION_PENDING, profile="p", session_id="s", detail="d", ts_ms=42)
    assert n.to_payload() == {
        "type": "blemees.notify",
        "reason": "permission_pending",
        "profile": "p",
        "session_id": "s",
        "title": "blemees: permission needed",
        "detail": "d",
        "ts_ms": 42,
    }


async def test_fire_records_outstanding_and_dispatches():
    sink = _RecordingSink()
    svc = _svc(sink)
    n = await svc.fire(reason=AGENT_CRASHED, profile="claude", session_id="s1", detail="boom")
    assert n.reason == "agent_crashed"
    assert [p["session_id"] for p in svc.outstanding()] == ["s1"]
    # Dispatch rides a service-owned task (decoupled from the caller, #51) —
    # drain it before observing sink emissions.
    await asyncio.gather(*svc._tasks)
    assert sink.seen[0].session_id == "s1"


async def test_clear_removes_from_outstanding():
    svc = _svc(_RecordingSink())
    await svc.fire(reason=AGENT_CRASHED, profile="p", session_id="s1", detail="d")
    assert svc.clear("s1") is True
    assert svc.outstanding() == []
    assert svc.clear("s1") is False  # idempotent


async def test_outstanding_sorted_oldest_first():
    ts = iter([300, 100, 200])
    svc = _svc(now_ms=lambda: next(ts))
    for sid in ("c", "a", "b"):
        await svc.fire(reason=AGENT_CRASHED, profile="p", session_id=sid, detail="d")
    assert [p["ts_ms"] for p in svc.outstanding()] == [100, 200, 300]


async def test_test_event_does_not_enter_outstanding():
    sink = _RecordingSink()
    svc = _svc(sink)
    n = await svc.test(profile="claude")
    assert n.reason == TEST
    assert sink.seen[0].reason == "test"
    assert svc.outstanding() == []  # a test never pollutes the queue


async def test_sink_failure_is_swallowed():
    good = _RecordingSink()
    svc = _svc(_BoomSink(), good)
    # A failing sink must not prevent other sinks or break fire().
    await svc.fire(reason=AGENT_CRASHED, profile="p", session_id="s1", detail="d")
    await asyncio.gather(*svc._tasks)
    assert good.seen[0].session_id == "s1"
    assert svc.outstanding()  # still recorded


# ---- WebhookSink ----------------------------------------------------


async def test_webhook_posts_to_resolved_url():
    posted: list[tuple[str, bytes]] = []

    sink = WebhookSink(
        resolve_url=lambda profile: {"claude": "https://hook/claude"}.get(profile),
        logger=LOG,
        post=lambda url, body: posted.append((url, body)),
    )
    await sink.emit(
        Notification(
            reason=PERMISSION_PENDING, profile="claude", session_id="s", detail="d", ts_ms=1
        )
    )
    assert len(posted) == 1
    url, body = posted[0]
    assert url == "https://hook/claude"
    import json

    assert json.loads(body)["reason"] == "permission_pending"


async def test_webhook_skips_when_no_url():
    posted: list[tuple[str, bytes]] = []
    sink = WebhookSink(
        resolve_url=lambda profile: None, logger=LOG, post=lambda u, b: posted.append((u, b))
    )
    await sink.emit(
        Notification(reason=PERMISSION_PENDING, profile="x", session_id="s", detail="d", ts_ms=1)
    )
    assert posted == []  # no URL configured → no POST


async def test_webhook_post_failure_is_swallowed():
    def boom(url, body):
        raise OSError("connection refused")

    sink = WebhookSink(resolve_url=lambda p: "https://hook", logger=LOG, post=boom)
    # Must not raise.
    await sink.emit(
        Notification(reason=AGENT_CRASHED, profile="p", session_id="s", detail="d", ts_ms=1)
    )


async def test_fire_survives_caller_cancellation():
    # The trigger may fire from a task that is cancelled immediately after
    # (the backend pump during a post-turn soft-kill, #51) — the dispatch
    # must complete anyway.
    sink = _RecordingSink()
    svc = _svc(sink)

    async def doomed():
        await svc.fire(reason=AGENT_CRASHED, profile="p", session_id="s1", detail="d")
        await asyncio.sleep(60)  # keep the caller alive so the cancel bites

    task = asyncio.create_task(doomed())
    await asyncio.sleep(0)  # let fire() schedule the dispatch task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    await asyncio.gather(*svc._tasks)
    assert sink.seen and sink.seen[0].session_id == "s1"
