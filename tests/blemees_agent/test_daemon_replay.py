"""Daemon-level tests for mid-turn disconnect, replay, and durable logs (blemees/3)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import pytest_asyncio

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import short_socket_path

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")

pytestmark = pytest.mark.asyncio


def _make_config(*, event_log_dir: Path | None = None, ring_buffer_size: int = 1024) -> Config:
    return Config(
        socket_path=str(short_socket_path("blemeesd-replay")),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        idle_timeout_s=60,
        max_concurrent_sessions=8,
        ring_buffer_size=ring_buffer_size,
        event_log_dir=str(event_log_dir) if event_log_dir else None,
    )


@pytest_asyncio.fixture
async def custom_daemon(tmp_path, request):
    overrides = getattr(request, "param", None) or {}
    event_log_dir = overrides.get("event_log_dir")
    if event_log_dir == "__tmp__":
        event_log_dir = tmp_path / "event_log"
    cfg = _make_config(
        event_log_dir=event_log_dir,
        ring_buffer_size=overrides.get("ring_buffer_size", 1024),
    )
    daemon = Daemon(cfg, configure("error"))
    await daemon.start()
    serve_task = asyncio.create_task(daemon.serve_forever())
    try:
        yield daemon, cfg
    finally:
        daemon.request_shutdown()
        try:
            await asyncio.wait_for(serve_task, timeout=5.0)
        except TimeoutError:
            serve_task.cancel()


class _Stream:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pump = asyncio.create_task(self._run())

    async def _run(self):
        try:
            while True:
                raw = await self.reader.readuntil(b"\n")
                await self._queue.put(json.loads(raw.rstrip(b"\r\n").decode("utf-8")))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            await self._queue.put(None)

    async def send(self, frame):
        self.writer.write((json.dumps(frame) + "\n").encode("utf-8"))
        await self.writer.drain()

    async def recv(self, timeout=5.0):
        evt = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if evt is None:
            raise ConnectionError("connection closed")
        return evt

    async def wait_for(self, pred, *, timeout=10.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("predicate never matched")
            evt = await self.recv(timeout=remaining)
            if pred(evt):
                return evt

    async def drain_seqs(self, *, until_quiet=0.4) -> list[int]:
        """Collect seq values until the stream goes quiet for ``until_quiet``."""
        seqs: list[int] = []
        while True:
            try:
                evt = await self.recv(timeout=until_quiet)
            except TimeoutError:
                return seqs
            if isinstance(evt.get("seq"), int):
                seqs.append(evt["seq"])

    async def close(self):
        self._pump.cancel()
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def _connect(socket_path: str) -> _Stream:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    s = _Stream(reader, writer)
    await s.send({"type": "hello", "client": "t/0", "protocol": PROTOCOL_VERSION})
    ack = await s.recv()
    assert ack["type"] == "hello_ack"
    return s


async def _open(s: _Stream, session_id: str, **extra) -> dict:
    frame = {"type": "session.open", "id": "r", "session_id": session_id, "options": {}, **extra}
    await s.send(frame)
    return await s.wait_for(lambda e: e["type"] == "session.opened")


async def _prompt(s: _Stream, session_id: str, text: str = "hi") -> None:
    await s.send({"type": "session.prompt", "session_id": session_id, "prompt": text})


# ---------------------------------------------------------------------------
# Events carry a monotonic seq.
# ---------------------------------------------------------------------------


async def test_outbound_events_carry_monotonic_seq(custom_daemon):
    _daemon, cfg = custom_daemon
    s = await _connect(cfg.socket_path)
    try:
        opened = await _open(s, "s1")
        assert opened["last_seq"] == 0
        await _prompt(s, "s1")
        seqs = await s.drain_seqs()
    finally:
        await s.close()
    assert seqs == sorted(seqs)
    # 2 session.update chunks + 1 session.result.
    assert len(seqs) >= 3
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))


# ---------------------------------------------------------------------------
# Reconnect with last_seen_seq replays missed frames.
# ---------------------------------------------------------------------------


async def test_reconnect_replays_from_last_seen_seq(custom_daemon):
    _daemon, cfg = custom_daemon
    s1 = await _connect(cfg.socket_path)
    await _open(s1, "rep")
    await _prompt(s1, "rep")
    first_seqs = await s1.drain_seqs()
    await s1.close()

    last_seq = max(first_seqs)
    mid_seq = max(1, last_seq - 2)

    s2 = await _connect(cfg.socket_path)
    try:
        opened = await _open(s2, "rep", resume=True, last_seen_seq=mid_seq)
        assert opened["last_seq"] >= last_seq
        replayed = await s2.drain_seqs()
        assert replayed, "expected some replayed frames"
        assert min(replayed) == mid_seq + 1
        assert max(replayed) == last_seq
    finally:
        await s2.close()


@pytest.mark.parametrize("custom_daemon", [{"ring_buffer_size": 1}], indirect=True)
async def test_reconnect_emits_replay_gap_when_buffer_rolled_over(custom_daemon):
    _daemon, cfg = custom_daemon
    s1 = await _connect(cfg.socket_path)
    await _open(s1, "gap")
    await _prompt(s1, "gap")
    await s1.wait_for(lambda e: e.get("type") == "session.result")
    await s1.drain_seqs()
    await s1.close()

    # Ring holds only the last frame, so seq 1 is gone from memory.
    s2 = await _connect(cfg.socket_path)
    try:
        await _open(s2, "gap", resume=True, last_seen_seq=1)
        gap = await s2.wait_for(lambda e: e.get("type") == "replay_gap")
        assert gap["since_seq"] == 1
        assert gap["first_available_seq"] > 2
    finally:
        await s2.close()


async def test_mid_turn_disconnect_preserves_events_for_replay(custom_daemon):
    _daemon, cfg = custom_daemon
    s1 = await _connect(cfg.socket_path)
    await _open(s1, "mid")
    await _prompt(s1, "mid")
    await s1.wait_for(lambda e: e.get("type") == "session.update")
    await s1.close()  # drop mid-turn

    await asyncio.sleep(0.5)  # let the turn finish and buffer

    s2 = await _connect(cfg.socket_path)
    try:
        await _open(s2, "mid", resume=True, last_seen_seq=0)
        saw_result = False
        while True:
            try:
                evt = await s2.recv(timeout=0.4)
            except TimeoutError:
                break
            if evt.get("type") == "session.result" and evt.get("session_id") == "mid":
                saw_result = True
        assert saw_result, "result must have been buffered while disconnected"
    finally:
        await s2.close()


# ---------------------------------------------------------------------------
# Durable log survives daemon restart.
# ---------------------------------------------------------------------------


async def test_event_log_survives_restart(tmp_path):
    log_dir = tmp_path / "event_log"
    cfg = _make_config(event_log_dir=log_dir)
    logger = configure("error")

    d1 = Daemon(cfg, logger)
    await d1.start()
    t1 = asyncio.create_task(d1.serve_forever())
    try:
        s = await _connect(cfg.socket_path)
        await _open(s, "dur")
        await _prompt(s, "dur")
        await s.wait_for(lambda e: e.get("type") == "session.result")
        await s.drain_seqs()
        await s.close()
    finally:
        d1.request_shutdown()
        await asyncio.wait_for(t1, timeout=5.0)

    log_file = log_dir / "dur.jsonl"
    assert log_file.is_file()
    seqs = [json.loads(line)["seq"] for line in log_file.read_text().splitlines() if line.strip()]
    assert seqs == sorted(seqs)

    cfg2 = _make_config(event_log_dir=log_dir)
    cfg2.socket_path = str(short_socket_path("blemeesd-replay2"))
    d2 = Daemon(cfg2, logger)
    await d2.start()
    t2 = asyncio.create_task(d2.serve_forever())
    try:
        s = await _connect(cfg2.socket_path)
        opened = await _open(s, "dur", resume=True, last_seen_seq=0)
        assert opened["last_seq"] >= max(seqs)
        replayed = await s.drain_seqs()
        assert len(replayed) >= len(seqs)
        await s.close()
    finally:
        d2.request_shutdown()
        await asyncio.wait_for(t2, timeout=5.0)
