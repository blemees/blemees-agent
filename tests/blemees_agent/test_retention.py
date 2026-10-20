"""Opt-in retention sweep for closed sessions (``session_retention_days``, #50)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION, config as config_mod
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure
from blemees_agent.registry import Registry
from blemees_agent.retention import sweep_expired_sessions
from blemees_agent.session import SessionTable

from .conftest import _StreamClient, short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")
DAY_MS = 24 * 60 * 60 * 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _backdate(path: Path, ms: int) -> None:
    os.utime(path, (ms / 1000, ms / 1000), follow_symlinks=False)


def _seed(state_dir: Path, records: list[dict]) -> Path:
    """Write ``registry.json`` + per-session files the way the daemon lays them out.

    Each file's mtime is set to the record's last activity, as it would be
    for a session the daemon last wrote at that time.
    """
    log_dir = state_dir / "sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        sid = rec["session_id"]
        last_ms = rec.get("last_active_at_ms") or rec.get("created_at_ms")
        files = {f"{sid}.jsonl": '{"seq":1}\n', f"{sid}.usage.json": "{}"}
        for name, text in files.items():
            path = log_dir / name
            path.write_text(text)
            if last_ms:
                _backdate(path, last_ms)
    payload = {"version": 1, "sessions": records, "profiles": []}
    (state_dir / "registry.json").write_text(json.dumps(payload))
    return log_dir


def _rec(sid: str, age_days: float) -> dict:
    ts = _now_ms() - int(age_days * DAY_MS)
    return {
        "session_id": sid,
        "profile": "default",
        "agent": "default",
        "created_at_ms": ts,
        "last_active_at_ms": ts,
        "turns": 1,
    }


def _registry_ids(state_dir: Path) -> set[str]:
    data = json.loads((state_dir / "registry.json").read_text())
    return {r["session_id"] for r in data["sessions"]}


def _sweep(state_dir: Path, days: int, sessions: SessionTable | None = None) -> list[str]:
    registry = Registry(state_dir / "registry.json")
    registry.load()
    log_dir = state_dir / "sessions"
    return sweep_expired_sessions(
        registry=registry,
        sessions=sessions
        or SessionTable(idle_timeout_s=60, max_concurrent=8, event_log_dir=str(log_dir)),
        log_dir=log_dir,
        retention_days=days,
        logger=configure("error"),
    )


@contextlib.asynccontextmanager
async def _daemon(state_dir: Path, socket_path: Path, **overrides):
    cfg = Config(
        socket_path=str(socket_path),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        state_dir=str(state_dir),
        idle_timeout_s=60,
        max_concurrent_sessions=8,
        **overrides,
    )
    daemon = Daemon(cfg, configure("error"))
    await daemon.start()
    serve = asyncio.create_task(daemon.serve_forever())
    try:
        yield daemon
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


# -- config ---------------------------------------------------------------


def test_default_is_disabled():
    assert Config(socket_path="/tmp/x").session_retention_days == 0


def test_env_and_toml(tmp_path, monkeypatch):
    toml = tmp_path / "agentd.toml"
    toml.write_text("session_retention_days = 14\n")
    monkeypatch.delenv("BLEMEES_AGENTD_SESSION_RETENTION_DAYS", raising=False)
    cfg, _ = config_mod.load(["--config", str(toml)])
    assert cfg.session_retention_days == 14
    monkeypatch.setenv("BLEMEES_AGENTD_SESSION_RETENTION_DAYS", "3")
    cfg, _ = config_mod.load(["--config", str(toml)])
    assert cfg.session_retention_days == 3


# -- (a) default config: nothing deleted -----------------------------------


async def test_default_config_deletes_nothing(tmp_path):
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("ancient", 3650)])
    sock = short_socket_path("blemeesd-ret")
    with socket_cleanup(sock):
        async with _daemon(state, sock):
            pass
    assert _registry_ids(state) == {"ancient"}
    assert (log_dir / "ancient.jsonl").is_file()
    assert (log_dir / "ancient.usage.json").is_file()


@pytest.mark.parametrize("days", [0, -5])
def test_zero_or_negative_disables(tmp_path, days):
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("ancient", 3650)])
    assert _sweep(state, days) == []
    assert _registry_ids(state) == {"ancient"}
    assert (log_dir / "ancient.jsonl").is_file()


async def test_negative_config_is_disabled_in_daemon(tmp_path):
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("ancient", 3650)])
    sock = short_socket_path("blemeesd-ret")
    with socket_cleanup(sock):
        async with _daemon(state, sock, session_retention_days=-1):
            pass
    assert (log_dir / "ancient.jsonl").is_file()


# -- (b) N > 0: old closed session removed, recent one kept ----------------


async def test_startup_sweep_removes_old_keeps_recent(tmp_path):
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("old", 10), _rec("recent", 1)])
    sock = short_socket_path("blemeesd-ret")
    with socket_cleanup(sock):
        async with _daemon(state, sock, session_retention_days=7):
            # The sweep ran in start(): the record is gone from the live
            # registry and from disk before any client connects.
            assert _registry_ids(state) == {"recent"}
            c = await _connect(sock)
            try:
                await c.send({"type": "session.list", "id": "l"})
                reply = await c.wait_for(lambda e: e.get("type") == "sessions")
                assert {r["session_id"] for r in reply["sessions"]} == {"recent"}
            finally:
                await c.close()
    assert not (log_dir / "old.jsonl").exists()
    assert not (log_dir / "old.usage.json").exists()
    assert (log_dir / "recent.jsonl").is_file()
    assert (log_dir / "recent.usage.json").is_file()


def test_falls_back_to_file_mtime_without_timestamps(tmp_path):
    state = tmp_path / "state"
    log_dir = _seed(state, [{"session_id": "bare-old"}, {"session_id": "bare-new"}])
    old = time.time() - 30 * 86400
    for name in ("bare-old.jsonl", "bare-old.usage.json"):
        os.utime(log_dir / name, (old, old))
    assert _sweep(state, 7) == ["bare-old"]
    assert _registry_ids(state) == {"bare-new"}


def test_uses_newest_of_record_and_file_mtime(tmp_path):
    """Old record timestamps but a fresh event log -> kept (max rule)."""
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("fresh-log", 30), _rec("all-old", 30)])
    # A frame was written to fresh-log's event log just now.
    (log_dir / "fresh-log.jsonl").write_text('{"seq":1}\n{"seq":2}\n')
    assert _sweep(state, 7) == ["all-old"]
    assert _registry_ids(state) == {"fresh-log"}
    assert (log_dir / "fresh-log.jsonl").is_file()
    assert (log_dir / "fresh-log.usage.json").is_file()


async def test_resumed_old_session_survives_restart_sweep(tmp_path):
    """Resume a 30-day-old record, run a turn, restart with retention on: kept."""
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("old", 30)])
    sock = short_socket_path("blemeesd-ret")
    with socket_cleanup(sock):
        async with _daemon(state, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "session.open",
                        "id": "o",
                        "session_id": "old",
                        "resume": True,
                        "options": {},
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "session.opened")
                await c.send({"type": "session.prompt", "session_id": "old", "prompt": "hi"})
                await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
            finally:
                await c.close()
    # The record itself was refreshed (open / close / shutdown touch it).
    rec = json.loads((state / "registry.json").read_text())["sessions"][0]
    assert rec["last_active_at_ms"] > _now_ms() - DAY_MS
    with socket_cleanup(sock):
        async with _daemon(state, sock, session_retention_days=7):
            assert _registry_ids(state) == {"old"}
    assert _registry_ids(state) == {"old"}
    assert (log_dir / "old.jsonl").is_file()
    assert (log_dir / "old.usage.json").is_file()


# -- (c) a live session is never swept -------------------------------------


async def test_live_session_is_not_swept(tmp_path):
    state = tmp_path / "state"
    sock = short_socket_path("blemeesd-ret")
    with socket_cleanup(sock):
        async with _daemon(state, sock, session_retention_days=7) as daemon:
            c = await _connect(sock)
            try:
                await c.send(
                    {"type": "session.open", "id": "o", "session_id": "live", "options": {}}
                )
                await c.wait_for(lambda e: e.get("type") == "session.opened")
                # Pretend the live session's last activity is ancient.
                rec = daemon._registry._records["live"]
                rec["last_active_at_ms"] = rec["created_at_ms"] = _now_ms() - 365 * DAY_MS
                assert daemon._sweep_retention(7) == []
                assert daemon._registry.get("live") is not None
                assert (state / "sessions" / "live.jsonl").is_file()
                # Still drivable.
                await c.send({"type": "session.prompt", "session_id": "live", "prompt": "hi"})
                await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
            finally:
                await c.close()


# -- (d) files outside the owned dirs are untouched ------------------------


def test_files_outside_owned_dirs_untouched(tmp_path):
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("old", 30), _rec("linked", 30)])
    # Agent-owned transcript with the same session id.
    agent_file = tmp_path / "home" / ".claude" / "projects" / "x" / "old.jsonl"
    agent_file.parent.mkdir(parents=True)
    agent_file.write_text("agent data")
    # A symlinked event log pointing out of the dir: unlink the link only.
    target = tmp_path / "elsewhere.jsonl"
    target.write_text("keep me")
    (log_dir / "linked.jsonl").unlink()
    (log_dir / "linked.jsonl").symlink_to(target)
    _backdate(log_dir / "linked.jsonl", _now_ms() - 30 * DAY_MS)
    # Unrelated files in the log dir (no registry record).
    (log_dir / "orphan.jsonl").write_text("x")
    (log_dir / "notes.txt").write_text("x")
    # A hand-edited record whose id would escape the log dir.
    escape = state / "escape.jsonl"
    escape.write_text("keep me")
    reg = json.loads((state / "registry.json").read_text())
    reg["sessions"].append(_rec("../escape", 30))
    (state / "registry.json").write_text(json.dumps(reg))

    removed = _sweep(state, 7)

    assert sorted(removed) == ["linked", "old"]
    assert agent_file.read_text() == "agent data"
    assert target.read_text() == "keep me"
    assert not (log_dir / "linked.jsonl").is_symlink()
    assert (log_dir / "orphan.jsonl").is_file()
    assert (log_dir / "notes.txt").is_file()
    assert escape.read_text() == "keep me"
    # The unsafe record is kept (logged as a failure), not silently dropped.
    assert _registry_ids(state) == {"../escape"}


# -- (e) one failure does not stop the sweep -------------------------------


def test_failure_on_one_entry_does_not_stop_others(tmp_path):
    state = tmp_path / "state"
    log_dir = _seed(state, [_rec("a", 30), _rec("broken", 30), _rec("c", 30)])
    # A non-empty directory where the sidecar should be: unlink() raises.
    (log_dir / "broken.usage.json").unlink()
    (log_dir / "broken.usage.json").mkdir()
    (log_dir / "broken.usage.json" / "x").write_text("x")
    _backdate(log_dir / "broken.usage.json", _now_ms() - 30 * DAY_MS)

    removed = _sweep(state, 7)

    assert sorted(removed) == ["a", "c"]
    assert _registry_ids(state) == {"broken"}
    for sid in ("a", "c"):
        assert not (log_dir / f"{sid}.jsonl").exists()
        assert not (log_dir / f"{sid}.usage.json").exists()


# -- close{delete:true} still uses the same file removal -------------------


async def test_close_delete_removes_log_and_sidecar(tmp_path):
    state = tmp_path / "state"
    log_dir = state / "sessions"
    sock = short_socket_path("blemeesd-ret")
    with socket_cleanup(sock):
        async with _daemon(state, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {"type": "session.open", "id": "o", "session_id": "gone", "options": {}}
                )
                await c.wait_for(lambda e: e.get("type") == "session.opened")
                await c.send({"type": "session.prompt", "session_id": "gone", "prompt": "hi"})
                await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)
                assert (log_dir / "gone.jsonl").is_file()
                assert (log_dir / "gone.usage.json").is_file()
                await c.send(
                    {"type": "session.close", "id": "x", "session_id": "gone", "delete": True}
                )
                await c.wait_for(lambda e: e.get("type") == "session.closed")
            finally:
                await c.close()
    assert not (log_dir / "gone.jsonl").exists()
    assert not (log_dir / "gone.usage.json").exists()
