"""Per-session agent backend Protocol.

The daemon holds one backend instance per session and drives it through
the narrow :class:`AgentBackend` Protocol below, so ``session.py`` and the
dispatcher stay backend-agnostic. Event delivery goes through the
per-session :data:`EventCallback`; the backend never touches ``Session``
directly.

The only implementation is
:class:`blemees_agent.backends.acp.AcpSessionHandle` (blemees/3): a
per-session view onto a profile's shared ACP agent subprocess. It
forwards user turns as ACP ``session/prompt``, emits ``session.*`` frames
(verbatim ACP ``session/update`` payloads), handles cancel, and surfaces
spawn / auth failures as typed errors.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

# The callback handed to a backend that takes one outbound ``session.*``
# frame and pushes it into the per-session event pipeline.
EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


@runtime_checkable
class AgentBackend(Protocol):
    """Protocol every concrete backend implements.

    Implementations must be awaitable-friendly: every method is async
    *except* the property-shaped accessors, which are pure.

    Event delivery uses the supplied `on_event` callback; the backend
    never touches `Session` directly. That keeps the dispatcher free
    to swap backends per session without leaking backend-specific
    plumbing into `session.py`.
    """

    backend: str
    """Backend name (``"acp"`` for the ACP backend)."""

    pid: int | None
    """The child's PID once spawned, or `None` before/after."""

    turn_active: bool
    """True while a user turn is in flight (between the prompt and its
    ``session.result``)."""

    @property
    def running(self) -> bool:
        """True iff the child has been spawned and has not yet exited."""
        ...

    async def spawn(self) -> None:
        """Start the upstream child and begin consuming its output.

        Raises `SpawnFailedError` if the binary is missing or the OS
        refuses the launch.
        """
        ...

    async def send_user_turn(self, message: dict[str, Any]) -> None:
        """Deliver one user turn to the child.

        `message` is the validated user-turn payload from the client
        frame. The backend translates it into its native input shape
        (ACP ``session/prompt``).

        Raises `SessionBusyError` if a turn is already in flight, and
        `SpawnFailedError` if the underlying transport fails.
        """
        ...

    async def interrupt(self) -> bool:
        """Cancel the in-flight turn, if any.

        Returns `True` if a kill / cancel was issued, `False` if no
        turn was in flight (the caller replies
        ``session.cancelled{was_idle:true}``).
        """
        ...

    async def close(self) -> None:
        """Tear down the child and release its readers."""
        ...

    async def wait_for_exit(self, timeout: float) -> bool:
        """Return `True` if the child has exited within `timeout` seconds."""
        ...


__all__ = [
    "AgentBackend",
    "EventCallback",
]
