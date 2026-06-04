"""ACP client backend (blemees/3, #16).

The daemon plays the **client** role of the
[Agent Client Protocol](https://agentclientprotocol.com): it spawns an
ACP *agent* subprocess (e.g. ``claude-agent-acp``, ``codex-acp``,
``gemini --experimental-acp``, ``cursor-agent acp``) over stdio via the
official ``agent-client-protocol`` SDK and translates the agent's
streamed ``session/update`` notifications into blemees ``session.*``
frames.

This is the tracer-bullet slice (#16): one hard-spawned agent, one ACP
session per backend instance, the empty-client-capability surface the
#15 spike validated (no ``fs`` / ``terminal`` — the agent self-serves
IO). Profiles (#17), multiplexing (#18), owner/viewer (#19) and the real
permission policy (#20) layer on top of this.

Frames emitted via ``on_event`` (the backend never assigns ``seq`` — the
owning :class:`~blemees_agent.session.Session` does):

* ``session.update``  — wraps a verbatim ACP ``SessionNotification.update``.
* ``session.result``  — turn end, carries ``stop_reason`` and optional ``usage``.
* ``session.error``   — backend/transport failure mid-turn.
* ``session.stderr``  — a line from the agent's stderr.

The backend conforms to :class:`~blemees_agent.backends.AgentBackend` so
the existing session/dispatch machinery drives it unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import uuid4

import acp
from acp import PROTOCOL_VERSION as ACP_PROTOCOL_VERSION, connect_to_agent, text_block
from acp.schema import AllowedOutcome, ClientCapabilities, RequestPermissionResponse

from ..errors import ProtocolError, SessionBusyError, SpawnFailedError
from . import EventCallback, build_spawn_env


def _to_content_blocks(message: dict[str, Any]) -> list[Any]:
    """Translate a blemees ``agent.user``-style message into ACP content blocks.

    #16 supports text only; image/audio/resource blocks (gated by the
    agent's ``promptCapabilities``) are a later addition. Non-text blocks
    raise :class:`ProtocolError` so the daemon can surface
    ``invalid_message`` rather than silently dropping content.
    """
    content = message.get("content")
    if isinstance(content, str):
        return [text_block(content)]
    if isinstance(content, list):
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append(text_block(str(block.get("text", ""))))
            else:
                kind = block.get("type") if isinstance(block, dict) else type(block).__name__
                raise ProtocolError(f"unsupported content block for ACP: {kind!r}")
        return blocks
    raise ProtocolError("message.content must be a string or array")


class _DaemonAcpClient(acp.Client):
    """The ACP client surface the daemon presents to an agent.

    Advertises **no** filesystem/terminal capabilities (the agent does its
    own IO — validated by the #15 spike). #16 auto-approves every
    permission request; the per-profile relay/stall policy is #20.
    """

    def __init__(self, backend: AcpBackend) -> None:
        self._backend = backend

    async def session_update(self, session_id: str, update: Any, **_kw: Any) -> None:
        await self._backend._emit(
            {
                "type": "session.update",
                # by_alias=True keeps the verbatim ACP wire shape (camelCase
                # discriminators like `sessionUpdate`, `toolCallId`).
                "update": update.model_dump(mode="json", by_alias=True, exclude_none=True),
            }
        )

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **_kw: Any
    ) -> RequestPermissionResponse:
        chosen = next((o for o in options if o.kind.startswith("allow")), options[0])
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=chosen.option_id, outcome="selected")
        )


class AcpBackend:
    """One ACP agent subprocess, driven over stdio via the ACP SDK."""

    backend = "acp"

    def __init__(
        self,
        *,
        session_id: str,
        command: str,
        args: list[str],
        cwd: str | None,
        on_event: EventCallback,
        logger: Any,
        env: dict[str, str] | None = None,
        model: str | None = None,
        alias: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._command = command
        self._args = list(args)
        self._cwd = cwd
        self._on_event = on_event
        self._log = logger
        self._env = env if env is not None else build_spawn_env(session_id, cwd, alias)
        self._model = model

        self._proc: asyncio.subprocess.Process | None = None
        self._conn: Any = None  # acp.ClientSideConnection once spawned
        self._stderr_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None

        # Surfaced to the daemon: the agent's own session id and whether it
        # supports session/load (drives the durable-resume path, #23).
        self.native_session_id: str | None = None
        self.load_session: bool = False
        self.turn_active: bool = False

    # -- AgentBackend protocol -----------------------------------------

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    async def spawn(self) -> None:
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._cwd or None,
                env=self._env,
            )
        except (OSError, ValueError) as exc:
            raise SpawnFailedError(f"failed to spawn ACP agent {self._command!r}: {exc}") from exc

        # ClientSideConnection wants (input_stream=writer→agent stdin,
        # output_stream=reader←agent stdout). asyncio gives us exactly those.
        assert self._proc.stdin is not None and self._proc.stdout is not None
        # input_stream = agent stdin (we write), output_stream = agent stdout (we read).
        self._conn = connect_to_agent(_DaemonAcpClient(self), self._proc.stdin, self._proc.stdout)
        self._stderr_task = asyncio.create_task(
            self._pump_stderr(), name=f"acp-stderr-{self._session_id}"
        )

        try:
            init = await self._conn.initialize(
                protocol_version=ACP_PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
            )
            caps = getattr(init, "agent_capabilities", None)
            self.load_session = bool(getattr(caps, "load_session", False))
            ns = await self._conn.new_session(cwd=self._cwd or ".", mcp_servers=[])
            self.native_session_id = ns.session_id
        except Exception as exc:  # noqa: BLE001 — any init failure is a spawn failure
            await self.close()
            raise SpawnFailedError(f"ACP initialize/new_session failed: {exc}") from exc

        self._log.info(
            "acp.spawned",
            session_id=self._session_id,
            command=self._command,
            native_session_id=self.native_session_id,
            load_session=self.load_session,
        )

    async def send_user_turn(self, message: dict[str, Any]) -> None:
        if self.turn_active:
            raise SessionBusyError("a turn is already in flight")
        if self._conn is None or self.native_session_id is None:
            raise SpawnFailedError("ACP session not initialised")
        blocks = _to_content_blocks(message)  # may raise ProtocolError
        self.turn_active = True
        self._turn_task = asyncio.create_task(
            self._run_turn(blocks), name=f"acp-turn-{self._session_id}"
        )

    async def _run_turn(self, blocks: list[Any]) -> None:
        assert self._conn is not None and self.native_session_id is not None
        try:
            resp = await self._conn.prompt(
                prompt=blocks,
                session_id=self.native_session_id,
                message_id=str(uuid4()),
            )
            frame: dict[str, Any] = {"type": "session.result", "stop_reason": resp.stop_reason}
            usage = getattr(resp, "usage", None)
            if usage is not None:
                # Finding A (#15 spike): ACP carries usage; surface it when present.
                frame["usage"] = usage.model_dump(mode="json", by_alias=True, exclude_none=True)
            await self._emit(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface transport/agent errors as a frame
            await self._emit(
                {"type": "session.error", "code": "agent_crashed", "message": str(exc)}
            )
        finally:
            self.turn_active = False

    async def interrupt(self) -> bool:
        if not self.turn_active or self._conn is None or self.native_session_id is None:
            return False
        # ACP session/cancel is a notification; the in-flight prompt() returns
        # with stop_reason "cancelled", which _run_turn turns into session.result.
        await self._conn.cancel(session_id=self.native_session_id)
        return True

    async def close(self) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._conn = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                with contextlib.suppress(asyncio.CancelledError):
                    await proc.wait()

    async def wait_for_exit(self, timeout: float) -> bool:
        if self._proc is None:
            return True
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    # -- internals ------------------------------------------------------

    async def _emit(self, frame: dict[str, Any]) -> None:
        await self._on_event(frame)

    async def _pump_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                await self._emit(
                    {"type": "session.stderr", "line": line.decode("utf-8", "replace").rstrip("\n")}
                )
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 — never let stderr pumping crash the backend
            return
