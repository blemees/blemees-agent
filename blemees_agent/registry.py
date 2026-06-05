"""Persistent session registry (blemees/3, #21).

The daemon-owned source of truth for *what sessions exist* — survives daemon
restarts so ``session.list`` / ``session.info`` can report sessions from a
prior run (and the daemon can later respawn/resume them, #22/#23). Replaces
the old practice of scanning agent-specific transcript directories
(``~/.claude``/``~/.codex``), which broke the any-ACP-agent generality.

One JSON file (``registry.json``) under the daemon's state dir, written
atomically (temp + rename). Each record carries the metadata
``session.list`` / ``session.info`` need; live runtime state (attached,
running, turn_active, seq, usage) is overlaid from the in-memory
``SessionTable`` at read time.

With no state dir configured the registry is purely in-memory (no
persistence) — the daemon still works, it just doesn't survive a restart.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# Fields persisted per session. Live-only fields (attached/running/seq/usage)
# are NOT stored — they're overlaid from the SessionTable on read.
_RECORD_FIELDS = (
    "session_id",
    "profile",
    "agent",
    # The agent's own session id, persisted so a restart can resume it via
    # ACP session/load (#23).
    "native_session_id",
    "cwd",
    "model",
    "mode",
    "created_at_ms",
    "last_active_at_ms",
    "turns",
    "view_only",
)


def _now_ms() -> int:
    return int(time.time() * 1000)


class Registry:
    """In-memory session records, optionally persisted to ``registry.json``."""

    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path else None
        self._records: dict[str, dict[str, Any]] = {}
        # Over-wire profiles (#25): the raw profile spec keyed by name, persisted
        # alongside sessions so dynamically-created profiles survive a restart.
        self._profiles: dict[str, dict[str, Any]] = {}

    @property
    def persistent(self) -> bool:
        return self._path is not None

    def load(self) -> None:
        if self._path is None or not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        sessions = data.get("sessions") if isinstance(data, dict) else None
        if isinstance(sessions, list):
            for rec in sessions:
                if isinstance(rec, dict) and isinstance(rec.get("session_id"), str):
                    self._records[rec["session_id"]] = {
                        k: rec.get(k) for k in _RECORD_FIELDS if k in rec
                    }
        profiles = data.get("profiles") if isinstance(data, dict) else None
        if isinstance(profiles, list):
            for spec in profiles:
                if isinstance(spec, dict) and isinstance(spec.get("name"), str):
                    self._profiles[spec["name"]] = dict(spec)

    def save(self) -> None:
        if self._path is None:
            return
        payload = {
            "version": 1,
            "sessions": list(self._records.values()),
            "profiles": list(self._profiles.values()),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass  # best-effort; in-memory records remain authoritative

    def upsert(self, session_id: str, **fields: Any) -> dict[str, Any]:
        rec = self._records.setdefault(session_id, {"session_id": session_id})
        for key, value in fields.items():
            if key in _RECORD_FIELDS and value is not None:
                rec[key] = value
        rec.setdefault("created_at_ms", _now_ms())
        rec["last_active_at_ms"] = (
            fields.get("last_active_at_ms") or rec.get("last_active_at_ms") or rec["created_at_ms"]
        )
        return rec

    def touch(self, session_id: str, *, turns: int | None = None, model: str | None = None) -> None:
        rec = self._records.get(session_id)
        if rec is None:
            return
        rec["last_active_at_ms"] = _now_ms()
        if turns is not None:
            rec["turns"] = turns
        if model is not None:
            rec["model"] = model

    def remove(self, session_id: str) -> bool:
        return self._records.pop(session_id, None) is not None

    def get(self, session_id: str) -> dict[str, Any] | None:
        rec = self._records.get(session_id)
        return dict(rec) if rec is not None else None

    def all(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records.values()]

    # -- over-wire profiles (#25) ---------------------------------------

    def upsert_profile(self, name: str, spec: dict[str, Any]) -> None:
        self._profiles[name] = {**spec, "name": name}

    def remove_profile(self, name: str) -> bool:
        return self._profiles.pop(name, None) is not None

    def get_profile_spec(self, name: str) -> dict[str, Any] | None:
        spec = self._profiles.get(name)
        return dict(spec) if spec is not None else None

    def all_profiles(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._profiles.values()]
