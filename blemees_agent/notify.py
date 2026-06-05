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
TEST = "test"

_TITLES = {
    PERMISSION_PENDING: "blemees: permission needed",
    AUTH_REQUIRED: "blemees: authentication required",
    AGENT_CRASHED: "blemees: agent crashed",
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
        await self._dispatch(notification)
        return notification

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
