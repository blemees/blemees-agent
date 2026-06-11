"""Notify service (#24, Phase 5).

The daemon models ``needs_attention`` as a per-session state entered when a
session needs its owner and none is attached (see :mod:`session` for the
state machine and the entry/exit edges). On *entry* the daemon fires one
structured :class:`Notification`; sinks consume it. The outstanding set is
kept here so an attaching client can read the queue immediately
(surfaced via ``status``).

The primary built-in sink is an outbound **webhook** (:class:`WebhookSink`):
an HTTP ``POST`` of the JSON payload to a per-profile URL with a global
fallback, so the user routes it to ntfy / Pushover / Slack / Discord / a
custom service. Sinks are best-effort — a failing or slow sink never blocks
or breaks the trigger that fired it.

Design: ``docs/acp-migration-spec.md`` §6.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

# Notification reasons. The first three are the spec's documented triggers;
# ``test`` is emitted only by the ``notify.test`` verb.
PERMISSION_PENDING = "permission_pending"
AUTH_REQUIRED = "auth_required"
AGENT_CRASHED = "agent_crashed"
TURN_COMPLETE = "turn_complete"
TEST = "test"

# The per-profile attention policy (#51). "Blocked" triggers — the session
# cannot make progress without its owner — are the default; turn_complete
# (a turn finished while detached) is opt-in per profile so push stays
# scarce unless a profile asks for it (2026-06-10 roadmap decision).
BLOCKED_TRIGGERS = frozenset({PERMISSION_PENDING, AUTH_REQUIRED, AGENT_CRASHED})
KNOWN_TRIGGERS = BLOCKED_TRIGGERS | {TURN_COMPLETE}

_TITLES = {
    PERMISSION_PENDING: "blemees: permission needed",
    AUTH_REQUIRED: "blemees: authentication required",
    AGENT_CRASHED: "blemees: agent crashed",
    TURN_COMPLETE: "blemees: turn complete",
    TEST: "blemees: test notification",
}


def _default_now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(slots=True)
class Notification:
    """One attention event. ``to_payload`` is the documented webhook body."""

    reason: str
    profile: str
    session_id: str
    detail: str
    ts_ms: int
    title: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            self.title = _TITLES.get(self.reason, "blemees: attention needed")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "blemees.notify",
            "reason": self.reason,
            "profile": self.profile,
            "session_id": self.session_id,
            "title": self.title,
            "detail": self.detail,
            "ts_ms": self.ts_ms,
        }


class Sink(Protocol):
    """Consumes notifications. Implementations must be best-effort."""

    async def emit(self, notification: Notification) -> None:
        raise NotImplementedError


class WebhookSink:
    """POST the notification payload to a per-profile URL (global fallback).

    ``resolve_url`` maps a profile name → its webhook URL (or ``None``); a
    ``None`` result means "no webhook for this profile" and the POST is
    skipped. The blocking HTTP call runs in a worker thread under a timeout so
    a hung endpoint can't stall the daemon's event loop.
    """

    def __init__(
        self,
        resolve_url: Callable[[str], str | None],
        logger: Any,
        *,
        timeout_s: float = 5.0,
        post: Callable[[str, bytes], None] | None = None,
    ) -> None:
        self._resolve_url = resolve_url
        self._log = logger
        self._timeout_s = timeout_s
        self._post = post or self._http_post

    async def emit(self, notification: Notification) -> None:
        url = self._resolve_url(notification.profile)
        if not url:
            return
        body = json.dumps(notification.to_payload()).encode("utf-8")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._post, url, body), timeout=self._timeout_s
            )
        except Exception as exc:  # best-effort: log and move on
            self._log.warning(
                "notify.webhook_failed",
                reason=notification.reason,
                session_id=notification.session_id,
                error=str(exc),
            )

    def _http_post(self, url: str, body: bytes) -> None:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "blemees-agentd"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310 (operator-configured URL)
            resp.read()


@dataclass(slots=True)
class NotifyService:
    """Owns the outstanding-attention set and dispatches to sinks.

    Callers (the session state machine) are responsible for firing on the
    *entry* edge of ``needs_attention`` and calling :meth:`clear` on exit, so
    the service trusts each :meth:`fire` to be a real new event — it does not
    dedupe.
    """

    sinks: list[Sink] = field(default_factory=list)
    logger: Any = None
    now_ms: Callable[[], int] = _default_now_ms
    _outstanding: dict[str, Notification] = field(default_factory=dict, init=False)
    # Dispatch tasks are service-owned: a trigger may fire from a task that
    # is about to be cancelled (e.g. the backend pump during a post-turn
    # soft-kill), and an inline await would die with it — silently eating
    # the webhook (#51). Refs held per the asyncio GC rules.
    _tasks: set = field(default_factory=set, init=False)

    async def fire(
        self, *, reason: str, profile: str, session_id: str, detail: str
    ) -> Notification:
        """Record an entered-attention event and dispatch it to all sinks."""
        notification = Notification(
            reason=reason,
            profile=profile,
            session_id=session_id,
            detail=detail,
            ts_ms=self.now_ms(),
        )
        self._outstanding[session_id] = notification
        if not self.sinks:
            return notification
        # Fire-and-forget on a service-owned task — sinks are best-effort and
        # must never block or die with the trigger that fired them.
        task = asyncio.create_task(self._dispatch(notification))
        self._tasks.add(task)
        task.add_done_callback(self._reap)
        return notification

    def _reap(self, task: asyncio.Task) -> None:
        """Drop a finished dispatch task and surface unexpected crashes
        instead of relying on the never-retrieved warning."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None and self.logger is not None:
            self.logger.warning("notify.dispatch_crashed", error=repr(exc))

    async def test(
        self, *, profile: str, session_id: str = "notify-test", detail: str = "test event"
    ) -> Notification:
        """Fire a synthetic event through the sinks without entering the queue."""
        notification = Notification(
            reason=TEST,
            profile=profile,
            session_id=session_id,
            detail=detail,
            ts_ms=self.now_ms(),
        )
        await self._dispatch(notification)
        return notification

    def clear(self, session_id: str) -> bool:
        """Drop a session from the outstanding set (attention resolved)."""
        return self._outstanding.pop(session_id, None) is not None

    def outstanding(self) -> list[dict[str, Any]]:
        """The outstanding attention set as payloads, oldest first."""
        return [n.to_payload() for n in sorted(self._outstanding.values(), key=lambda n: n.ts_ms)]

    async def _dispatch(self, notification: Notification) -> None:
        if not self.sinks:
            return
        results = await asyncio.gather(
            *(sink.emit(notification) for sink in self.sinks), return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception) and self.logger is not None:
                self.logger.warning("notify.sink_error", error=str(result))
