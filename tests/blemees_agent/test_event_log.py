"""Unit tests for the ring buffer + durable event log and Session replay."""

from __future__ import annotations

import asyncio

from blemees_agent.event_log import DurableEventLog, RingBuffer, event_log_path
from blemees_agent.protocol import OpenMessage
from blemees_agent.session import Session, SessionTable

# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_drops_oldest():
    ring = RingBuffer(3)
    for i in range(5):
        ring.append({"seq": i + 1})
    assert [f["seq"] for f in ring.since(0)] == [3, 4, 5]
    assert ring.earliest_seq() == 3
    assert ring.latest_seq() == 5


def test_ring_buffer_since_filters_inclusive_boundary():
    ring = RingBuffer(10)
    for i in range(1, 6):
        ring.append({"seq": i})
    assert [f["seq"] for f in ring.since(2)] == [3, 4, 5]
    assert ring.since(5) == []


# ---------------------------------------------------------------------------
# DurableEventLog
# ---------------------------------------------------------------------------


def test_durable_log_roundtrips(tmp_path):
    log = DurableEventLog(tmp_path / "sess.jsonl")
    log.open()
    for i in range(1, 4):
        log.append({"seq": i, "type": "x"})
    log.close()
    assert [r["seq"] for r in log.tail(10)] == [1, 2, 3]


def test_durable_log_tail_skips_malformed(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"seq":1,"type":"x"}\nnot-json\n{"seq":2,"type":"y"}\n', encoding="utf-8")
    log = DurableEventLog(path)
    assert [r["seq"] for r in log.tail(10)] == [1, 2]


def test_durable_log_unlink_removes_file(tmp_path):
    p = tmp_path / "s.jsonl"
    log = DurableEventLog(p)
    log.open()
    log.append({"seq": 1})
    log.close()
    assert p.exists()
    log.unlink()
    assert not p.exists()


def test_event_log_path_joins():
    assert event_log_path("/a/b", "abc").name == "abc.jsonl"


# ---------------------------------------------------------------------------
# Session event dispatch
# ---------------------------------------------------------------------------


def _open_msg(session: str = "s1") -> OpenMessage:
    return OpenMessage(id=None, session_id=session, options={}, resume=False)


async def test_session_assigns_monotonic_seq_and_buffers():
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    seen: list[dict] = []

    async def writer(frame):
        seen.append(frame)

    await sess.attach(connection_id=1, writer=writer)
    await sess.on_event({"type": "stream_event", "session_id": "s1"})
    await sess.on_event({"type": "stream_event", "session_id": "s1"})
    assert [f["seq"] for f in seen] == [1, 2]
    assert [f["seq"] for f in sess.ring.since(0)] == [1, 2]


async def test_session_replays_since_last_seen_seq():
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)

    # Fill before attach: emulates events arriving while detached.
    for _ in range(5):
        await sess.on_event({"type": "stream_event", "session_id": "s1"})

    seen: list[dict] = []

    async def writer(frame):
        seen.append(frame)

    summary = await sess.attach(connection_id=2, writer=writer, last_seen_seq=3)
    assert [f["seq"] for f in seen] == [4, 5]
    assert summary["replayed"] == 2
    assert summary["gap_from"] == 0


async def test_session_emits_replay_gap_when_ring_rolled_over():
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sess.ring = RingBuffer(3)
    for _ in range(10):
        await sess.on_event({"type": "stream_event", "session_id": "s1"})

    seen: list[dict] = []

    async def writer(frame):
        seen.append(frame)

    summary = await sess.attach(connection_id=3, writer=writer, last_seen_seq=2)
    assert [f["seq"] for f in seen[:3]] == [8, 9, 10]
    assert seen[3]["type"] == "replay_gap"
    assert seen[3]["first_available_seq"] == 8
    assert seen[3]["seq"] == 11
    assert summary["gap_from"] == 3 and summary["gap_to"] == 7


async def test_session_no_replay_when_last_seen_seq_is_caught_up():
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    for _ in range(3):
        await sess.on_event({"type": "stream_event", "session_id": "s1"})
    seen: list[dict] = []

    async def writer(frame):
        seen.append(frame)

    summary = await sess.attach(connection_id=3, writer=writer, last_seen_seq=3)
    assert seen == []
    assert summary["replayed"] == 0


async def test_session_durable_log_persists_and_reloads(tmp_path):
    sess1 = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sess1.enable_durable_log(tmp_path)
    for _ in range(4):
        await sess1.on_event({"type": "stream_event", "session_id": "s1"})
    sess1.log.close()

    # Simulate a daemon restart: new Session with same id picks up the log.
    sess2 = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sess2.enable_durable_log(tmp_path)
    # Ring seeded and seq resumed.
    assert sess2.seq >= 4
    assert len(sess2.ring) == 4

    seen: list[dict] = []

    async def writer(frame):
        seen.append(frame)

    summary = await sess2.attach(connection_id=9, writer=writer, last_seen_seq=2)
    assert [f["seq"] for f in seen] == [3, 4]
    assert summary["replayed"] == 2


async def test_session_detach_writer_stops_delivery():
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    seen: list[dict] = []

    async def writer(frame):
        seen.append(frame)

    await sess.attach(connection_id=1, writer=writer)
    await sess.on_event({"type": "stream_event", "session_id": "s1"})
    sess.detach_writer()
    await sess.on_event({"type": "stream_event", "session_id": "s1"})
    assert len(seen) == 1
    # Still buffered for a future attach.
    assert len(sess.ring) == 2


async def test_session_finishing_triggers_soft_kill_on_result():
    class FakeSub:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sub = FakeSub()
    sess.backend = sub  # type: ignore[assignment]
    sess.mark_finishing()

    await sess.on_event({"type": "session.result", "session_id": "s1", "subtype": "success"})
    # Give the scheduled close task a chance to run.
    await asyncio.sleep(0.01)
    assert sub.closed is True


async def test_soft_kill_task_is_held_until_done():
    # #50: the loop keeps only weak refs to tasks, so the session must hold
    # the soft-kill close task until it finishes.
    release = asyncio.Event()

    class SlowSub:
        def __init__(self):
            self.closed = False

        async def close(self):
            await release.wait()
            self.closed = True

    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sub = SlowSub()
    sess.backend = sub  # type: ignore[assignment]
    sess.mark_finishing()

    await sess.on_event({"type": "session.result", "session_id": "s1", "subtype": "success"})
    await asyncio.sleep(0)
    assert len(sess._bg_tasks) == 1
    (task,) = sess._bg_tasks
    assert task.get_name() == "backend-soft-kill-s1"
    assert not task.done()

    release.set()
    await sess.wait_background_tasks()
    assert sub.closed is True
    assert sess._bg_tasks == set()


async def test_soft_kill_task_failure_is_logged(monkeypatch):
    import blemees_agent.session as session_mod

    warnings: list[tuple[str, dict]] = []

    class RecordingLog:
        def warning(self, event, **fields):
            warnings.append((event, fields))

    monkeypatch.setattr(session_mod, "_log", RecordingLog())

    class BrokenSub:
        async def close(self):
            raise RuntimeError("boom")

    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sess.backend = BrokenSub()  # type: ignore[assignment]
    sess.mark_finishing()

    await sess.on_event({"type": "session.result", "session_id": "s1", "subtype": "success"})
    await sess.wait_background_tasks()
    await asyncio.sleep(0)  # let the done-callback run
    assert sess._bg_tasks == set()
    assert len(warnings) == 1
    event, fields = warnings[0]
    assert event == "session.background_task_failed"
    assert fields["session_id"] == "s1"
    assert "boom" in fields["error"]


async def test_table_remove_waits_for_soft_kill_task():
    order: list[str] = []
    release = asyncio.Event()

    class SlowSub:
        async def close(self):
            if not release.is_set():
                await release.wait()
                order.append("soft-kill")
            else:
                order.append("remove-close")

    table = SessionTable(idle_timeout_s=60, max_concurrent=4)
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    await table.register(sess)
    sess.backend = SlowSub()  # type: ignore[assignment]
    sess.mark_finishing()
    await sess.on_event({"type": "session.result", "session_id": "s1", "subtype": "success"})

    asyncio.get_running_loop().call_later(0.01, release.set)
    await table.remove("s1")
    assert order == ["soft-kill", "remove-close"]
    assert sess._bg_tasks == set()
