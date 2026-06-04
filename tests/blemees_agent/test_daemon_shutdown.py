"""Daemon shutdown behaviour: graceful (in-flight turns run to completion
within the grace window) vs force-kill when the grace period expires."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import short_socket_path

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")

pytestmark = pytest.mark.asyncio


def _config(*, grace_s: int) -> Config:
    return Config(
        socket_path=str(short_socket_path("blemeesd-shutdown")),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        idle_timeout_s=60,
        max_concurrent_sessions=8,
        shutdown_grace_s=grace_s,
    )


class _Stream:
    def __init__(self, r, w):
        self.reader = r
        self.writer = w
        self._q: asyncio.Queue = asyncio.Queue()
        self._pump = asyncio.create_task(self._run())

    async def _run(self):
        try:
            while True:
                raw = await self.reader.readuntil(b"\n")
                await self._q.put(json.loads(raw.rstrip(b"\r\n").decode("utf-8")))
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            await self._q.put(None)

    async def send(self, frame):
        self.writer.write((json.dumps(frame) + "\n").encode())
        await self.writer.drain()

    async def recv(self, timeout=5.0):
        evt = await asyncio.wait_for(self._q.get(), timeout=timeout)
        if evt is None:
            raise ConnectionError("closed")
        return evt

    async def wait_for(self, pred, *, timeout=10.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            evt = await self.recv(timeout=remaining)
            if pred(evt):
                return evt

    async def close(self):
        self._pump.cancel()
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def _connect(path: str) -> _Stream:
    r, w = await asyncio.open_unix_connection(path)
    s = _Stream(r, w)
    await s.send({"type": "hello", "client": "t/0", "protocol": PROTOCOL_VERSION})
    await s.recv()
    return s


async def _start_daemon(cfg: Config) -> tuple[Daemon, asyncio.Task]:
    daemon = Daemon(cfg, configure("error"))
    await daemon.start()
    return daemon, asyncio.create_task(daemon.serve_forever())


async def _open(s: _Stream, session_id: str) -> None:
    await s.send({"type": "session.open", "id": "r1", "session_id": session_id, "options": {}})
    await s.wait_for(lambda e: e.get("type") == "session.opened")


async def test_shutdown_waits_for_in_flight_turn():
    cfg = _config(grace_s=5)
    daemon, task = await _start_daemon(cfg)
    try:
        s = await _connect(cfg.socket_path)
        try:
            await _open(s, "g")
            await s.send({"type": "session.prompt", "session_id": "g", "prompt": "finish please"})
            await s.wait_for(lambda e: e.get("type") == "session.update")
            t0 = time.monotonic()
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=10.0)
            elapsed = time.monotonic() - t0
        finally:
            await s.close()
        # The turn needs ~0.5s to finish; graceful shutdown waits for it but
        # well under the 5s grace budget.
        assert 0.3 <= elapsed <= 3.0, f"elapsed={elapsed}"
    finally:
        if not task.done():
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=5.0)


async def test_shutdown_force_kills_when_grace_expires():
    cfg = _config(grace_s=1)
    daemon, task = await _start_daemon(cfg)
    try:
        s = await _connect(cfg.socket_path)
        try:
            await _open(s, "slow")
            await s.send({"type": "session.prompt", "session_id": "slow", "prompt": "hang please"})
            await s.wait_for(lambda e: e.get("type") == "session.update")
            t0 = time.monotonic()
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=10.0)
            elapsed = time.monotonic() - t0
        finally:
            await s.close()
        assert 0.9 <= elapsed <= 4.0, f"elapsed={elapsed}"
    finally:
        if not task.done():
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=5.0)


async def test_shutdown_grace_zero_kills_immediately():
    cfg = _config(grace_s=0)
    daemon, task = await _start_daemon(cfg)
    try:
        s = await _connect(cfg.socket_path)
        try:
            await _open(s, "z")
            await s.send({"type": "session.prompt", "session_id": "z", "prompt": "hang please"})
            await s.wait_for(lambda e: e.get("type") == "session.update")
            t0 = time.monotonic()
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=5.0)
            elapsed = time.monotonic() - t0
        finally:
            await s.close()
        assert elapsed <= 2.0, f"elapsed={elapsed}"
    finally:
        if not task.done():
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=5.0)


async def test_shutdown_skips_wait_for_idle_sessions():
    cfg = _config(grace_s=30)
    daemon, task = await _start_daemon(cfg)
    try:
        s = await _connect(cfg.socket_path)
        try:
            await _open(s, "idle")  # no prompt → session idle
            t0 = time.monotonic()
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=5.0)
            elapsed = time.monotonic() - t0
        finally:
            await s.close()
        assert elapsed <= 2.0, f"elapsed={elapsed}"
    finally:
        if not task.done():
            daemon.request_shutdown()
            await asyncio.wait_for(task, timeout=5.0)
