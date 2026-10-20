"""Opt-in retention sweep for closed sessions (``session_retention_days``).

Without a sweep, every session the daemon ever opened keeps a record in
``registry.json`` and a ``<session>.jsonl`` event log (plus a
``<session>.usage.json`` sidecar) under the event-log dir, until a client
sends ``session.close{delete:true}``.

With ``session_retention_days = N`` (N > 0) the daemon removes, for each
registry record whose last activity is older than N days:

* the record itself (from ``registry.json`` in the state dir), and
* that session's event log + usage sidecar in the event-log dir — through
  :func:`~blemees_agent.event_log.delete_session_files`, the same helper
  ``close{delete:true}`` uses.

It never removes a session that is live in this daemon (any session in the
``SessionTable``: attached, detached-but-kept, or running a turn), never
touches agent-owned files (``~/.claude``, ``~/.codex``, …) or anything
outside the event-log dir, never follows a symlink (a symlinked log is
unlinked itself), and never touches over-wire profiles. ``N <= 0`` (the
default) disables the sweep.

Last activity is the record's ``last_active_at_ms``: the daemon sets it on
every open and refreshes it every 30 s while the session is live, so it is
the time the session was last in use. A record without it falls back to
``created_at_ms``, then to the newest mtime of the session's own files. A
record with none of these is kept.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .event_log import delete_session_files, session_file_paths
from .logging import StructuredLogger
from .registry import Registry
from .session import SessionTable

_DAY_MS = 24 * 60 * 60 * 1000
# How often a running daemon repeats the sweep (it also runs at startup).
SWEEP_INTERVAL_S = 24 * 60 * 60


def _last_active_ms(rec: dict[str, Any], log_dir: Path | None) -> int | None:
    for key in ("last_active_at_ms", "created_at_ms"):
        value = rec.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    if log_dir is None:
        return None
    mtimes: list[float] = []
    for path in session_file_paths(log_dir, rec["session_id"]):
        try:
            mtimes.append(path.lstat().st_mtime)
        except OSError:
            continue
    return int(max(mtimes) * 1000) if mtimes else None


def sweep_expired_sessions(
    *,
    registry: Registry,
    sessions: SessionTable,
    log_dir: str | Path | None,
    retention_days: int,
    logger: StructuredLogger,
    now_ms: int | None = None,
) -> list[str]:
    """Delete closed sessions idle for more than ``retention_days``.

    Synchronous on purpose: no ``await`` between the liveness check and the
    delete, so a session cannot be opened in between. Returns the ids
    removed. An error on one session is logged and the sweep goes on.
    """
    if retention_days <= 0:
        return []
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - retention_days * _DAY_MS
    base = Path(log_dir) if log_dir else None
    removed: list[str] = []
    failed = 0
    for rec in registry.all():
        sid = rec["session_id"]
        if sessions.try_get(sid) is not None:
            continue  # live in this daemon — never swept
        try:
            last_ms = _last_active_ms(rec, base)
            if last_ms is None or last_ms >= cutoff_ms:
                continue
            files = delete_session_files(base, sid) if base is not None else []
            registry.remove(sid)
        except Exception as exc:  # one bad entry must not stop the sweep
            failed += 1
            logger.warning("session.retention_delete_failed", session_id=sid, error=str(exc))
            continue
        removed.append(sid)
        logger.info(
            "session.retention_deleted",
            session_id=sid,
            last_active_at_ms=last_ms,
            files=[str(p) for p in files],
        )
    if removed:
        registry.save()
    logger.info(
        "session.retention_sweep",
        retention_days=retention_days,
        deleted=len(removed),
        failed=failed,
    )
    return removed
