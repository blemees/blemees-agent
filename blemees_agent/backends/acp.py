"""ACP client backend (blemees/3).

The daemon plays the **client** role of the
[Agent Client Protocol](https://agentclientprotocol.com): it spawns ACP
*agent* subprocesses over stdio via the official ``agent-client-protocol``
SDK and translates each agent's streamed ``session/update`` notifications
into blemees ``session.*`` frames.

Two layers (#17 profiles + supervisor):

* :class:`AcpAgentProcess` — one supervised agent subprocess per profile.
  ACP multiplexes many sessions over one stdio connection (the protocol's
  native model), so a single process hosts N sessions, demuxed by the
  agent's ``sessionId``. Owns the subprocess, the ACP connection, the
  ``initialize`` handshake, and per-session turn state.
* :class:`AcpSessionHandle` — the per-(daemon-)session view that conforms to
  :class:`~blemees_agent.backends.AgentBackend`, so the daemon's session /
  dispatch machinery drives it unchanged. Delegates to its profile's
  shared :class:`AcpAgentProcess`, scoped to one ACP ``sessionId``.

Frames emitted via ``on_event`` (the backend never assigns ``seq`` — the
owning :class:`~blemees_agent.session.Session` does): ``session.update``
(verbatim ACP), ``session.result`` (``stop_reason`` + optional ``usage``),
``session.error``, ``session.stderr``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import acp
from acp import PROTOCOL_VERSION as ACP_PROTOCOL_VERSION, connect_to_agent, text_block
from acp.connection import InMemoryMessageQueue
from acp.schema import (
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    McpServerStdio,
    RequestPermissionResponse,
)

from ..errors import AuthRequiredError, ProtocolError, SessionBusyError, SpawnFailedError
from . import EventCallback

if TYPE_CHECKING:
    from ..supervisor import Agent

# JSON-RPC error code the ACP SDK uses for "authentication required"
# (``acp.RequestError.auth_required``); see #24.
_ACP_AUTH_REQUIRED_CODE = -32000

# StreamReader buffer for the agent's stdout. ACP frames are newline-delimited
# JSON, and a single frame (a big tool result, file read, diff, or base64
# image) routinely exceeds asyncio's 64 KiB default — which makes
# ``readline()`` raise ``LimitOverrunError`` and kills the receive loop,
# surfacing as a spurious ``agent_crashed``. 16 MiB gives ample headroom.
_STDIO_BUFFER_LIMIT = 16 * 1024 * 1024


def _translate_request_error(exc: Exception) -> Exception:
    """Map an ACP auth rejection to :class:`AuthRequiredError`, else passthrough.

    The agent signals "the user must authenticate" by returning JSON-RPC
    ``-32000``; surfacing it distinctly lets the notify service route it as
    ``auth_required`` rather than a generic spawn failure (#24, §6).
    """
    if isinstance(exc, acp.RequestError) and exc.code == _ACP_AUTH_REQUIRED_CODE:
        return AuthRequiredError(f"ACP agent requires authentication: {exc}")
    return exc


def _to_content_blocks(message: dict[str, Any]) -> list[Any]:
    """Translate a blemees ``{"role":"user","content":...}`` message into ACP blocks.

    Text only for now; non-text blocks raise :class:`ProtocolError` so the
    daemon surfaces ``invalid_message`` rather than silently dropping content.
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


def _mcp_servers(agent: Agent) -> list[Any]:
    """Build ACP McpServer entries from a profile's mcp_servers config.

    #17 supports stdio servers (`{name, command, args, env}`); HTTP/SSE
    transports are a later addition.
    """
    out: list[Any] = []
    for spec in agent.mcp_servers:
        if not isinstance(spec, dict) or "command" not in spec:
            continue
        env = spec.get("env") or {}
        env_entries = [{"name": k, "value": str(v)} for k, v in env.items()]
        out.append(
            McpServerStdio(
                name=spec.get("name", spec["command"]),
                command=spec["command"],
                args=list(spec.get("args", [])),
                env=env_entries,
            )
        )
    return out


class _SessionState:
    """Per-ACP-session bookkeeping inside a shared process."""

    def __init__(self, on_event: EventCallback, permission_cb: Any = None) -> None:
        self.on_event = on_event
        # async (options: list[dict], tool_call: dict) -> decision dict, or None
        # to auto-allow (the #16/#17 fallback). Supplied by the daemon (#20).
        self.permission_cb = permission_cb
        self.turn_active = False
        self.turn_task: asyncio.Task | None = None
        self.cancelled = False  # user interrupted this turn → finalize as cancelled
        self.notified_crash = False  # agent_crashed already emitted for this turn
        self.loading = False  # session/load is replaying history → drop those updates


class _ProcessClient(acp.Client):
    """ACP client surface for a shared agent process; routes by sessionId.

    Advertises no fs/terminal capabilities (the agent self-serves IO, per the
    #15 spike). #17 auto-approves permission requests; the per-profile
    relay/stall policy is #20.
    """

    def __init__(self, process: AcpAgentProcess) -> None:
        self._p = process

    async def session_update(self, session_id: str, update: Any, **_kw: Any) -> None:
        st = self._p.sessions.get(session_id)
        if st is None or st.cancelled or st.notified_crash or st.loading:
            # Drop updates for a cancelled/crashed turn, or the history the
            # agent replays during session/load (the client already has it
            # from the durable event log, #22).
            return
        await st.on_event(
            {
                "type": "session.update",
                "update": update.model_dump(mode="json", by_alias=True, exclude_none=True),
            }
        )

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **_kw: Any
    ) -> RequestPermissionResponse:
        st = self._p.sessions.get(session_id)
        if st is None or st.permission_cb is None:
            # Fallback (no policy wired): auto-allow.
            chosen = next((o for o in options if o.kind.startswith("allow")), options[0])
            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=chosen.option_id, outcome="selected")
            )
        opts = [{"option_id": o.option_id, "name": o.name, "kind": o.kind} for o in options]
        tc = (
            tool_call.model_dump(mode="json", by_alias=True, exclude_none=True)
            if hasattr(tool_call, "model_dump")
            else tool_call
        )
        decision = await st.permission_cb(opts, tc)
        if decision.get("outcome") == "selected" and decision.get("option_id"):
            return RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=decision["option_id"], outcome="selected")
            )
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


class AcpAgentProcess:
    """One supervised ACP agent subprocess for a profile; multiplexes sessions."""

    def __init__(
        self, agent: Agent, *, key: tuple[str, str], logger: Any, env: dict[str, str]
    ) -> None:
        self.agent = agent
        self.key_tuple = key
        self._label = f"{key[0]}/{key[1]}"
        self._log = logger
        self._env = env
        self._proc: asyncio.subprocess.Process | None = None
        self._conn: Any = None
        self._rpc_queue: Any = None
        self._stderr_task: asyncio.Task | None = None
        self._exit_task: asyncio.Task | None = None
        self._closing = False  # set on intentional close() so the exit watcher is silent
        self._start_lock = asyncio.Lock()
        self.sessions: dict[str, _SessionState] = {}
        self.load_session: bool = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def session_count(self) -> int:
        return len(self.sessions)

    def is_turn_active(self, acp_session_id: str) -> bool:
        st = self.sessions.get(acp_session_id)
        return bool(st and st.turn_active)

    async def ensure_started(self) -> None:
        async with self._start_lock:
            if self.running:
                return
            # (Re)spawn: a prior crash left stale ACP session ids; drop them.
            self._closing = False
            self.sessions.clear()
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    self.agent.command,
                    *self.agent.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.agent.agent_home or None,
                    env=self._env,
                    limit=_STDIO_BUFFER_LIMIT,
                )
            except (OSError, ValueError) as exc:
                raise SpawnFailedError(
                    f"failed to spawn ACP agent {self.agent.command!r}: {exc}"
                ) from exc

            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._rpc_queue = InMemoryMessageQueue()
            self._conn = connect_to_agent(
                _ProcessClient(self), self._proc.stdin, self._proc.stdout, queue=self._rpc_queue
            )
            self._stderr_task = asyncio.create_task(
                self._pump_stderr(), name=f"acp-stderr-{self._label}"
            )
            try:
                init = await self._conn.initialize(
                    protocol_version=ACP_PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                )
                caps = getattr(init, "agent_capabilities", None)
                self.load_session = bool(getattr(caps, "load_session", False))
            except Exception as exc:  # noqa: BLE001 — init failure is a spawn failure
                await self.close()
                translated = _translate_request_error(exc)
                if isinstance(translated, AuthRequiredError):
                    raise translated from exc
                raise SpawnFailedError(f"ACP initialize failed: {exc}") from exc
            self._exit_task = asyncio.create_task(
                self._watch_exit(self._proc), name=f"acp-exit-{self._label}"
            )
            self._log.info(
                "acp.process_started",
                agent=self._label,
                command=self.agent.command,
                load_session=self.load_session,
            )

    async def _watch_exit(self, proc: asyncio.subprocess.Process) -> None:
        """Detect unexpected agent-process death and recover the sessions.

        On crash: emit ``session.error{agent_crashed}`` to every live session,
        cancel in-flight turns, and drop the (now-dead) ACP session ids. The
        daemon respawns this process on the next ``session.prompt`` (a fresh
        ACP session per daemon session; conversational resume is #23).
        """
        with contextlib.suppress(asyncio.CancelledError):
            await proc.wait()
        if self._closing or proc is not self._proc:
            return  # intentional close, or already replaced by a respawn
        self._log.warning("acp.process_crashed", agent=self._label, returncode=proc.returncode)
        crashed = list(self.sessions.items())
        self.sessions.clear()
        for _sid, st in crashed:
            if st.turn_task is not None and not st.turn_task.done():
                st.turn_task.cancel()
            if not st.notified_crash:
                st.notified_crash = True
                with contextlib.suppress(Exception):
                    await st.on_event(
                        {
                            "type": "session.error",
                            "code": "agent_crashed",
                            "message": "ACP agent process exited unexpectedly",
                        }
                    )

    async def new_session(
        self, *, cwd: str | None, on_event: EventCallback, permission_cb: Any = None
    ) -> str:
        await self.ensure_started()
        try:
            ns = await self._conn.new_session(
                cwd=cwd or self.agent.agent_home or ".",
                mcp_servers=_mcp_servers(self.agent),
            )
        except acp.RequestError as exc:
            raise _translate_request_error(exc) from exc
        sid = ns.session_id
        self.sessions[sid] = _SessionState(on_event, permission_cb)
        await self._apply_selection(sid, ns)
        return sid

    async def resume_session(
        self,
        *,
        native_session_id: str,
        cwd: str | None,
        on_event: EventCallback,
        permission_cb: Any,
    ) -> str:
        """Rehydrate a prior agent session via ACP ``session/load`` (#23).

        The agent replays the conversation as ``session/update`` notifications;
        we suppress them (the client already has the history from the durable
        event log) and keep only the model-side context warm so the next turn
        continues the conversation.
        """
        await self.ensure_started()
        st = _SessionState(on_event, permission_cb)
        st.loading = True
        self.sessions[native_session_id] = st
        try:
            await self._conn.load_session(
                cwd=cwd or self.agent.agent_home or ".",
                session_id=native_session_id,
                mcp_servers=_mcp_servers(self.agent),
            )
        except acp.RequestError as exc:
            self.sessions.pop(native_session_id, None)
            raise _translate_request_error(exc) from exc
        finally:
            st.loading = False
        return native_session_id

    async def _apply_selection(self, sid: str, ns: Any) -> None:
        """Apply the profile's model/mode after session/new (finding B, #15).

        Heterogeneous across agents: model/mode live in SessionModelState /
        SessionModeState (set_session_model/mode) or config_options
        (set_config_option). Best-effort — warn, never fail, on a miss.
        """
        if self.agent.model:
            await self._select(sid, ns, want=self.agent.model, category="model")
        if self.agent.mode:
            await self._select(sid, ns, want=self.agent.mode, category="mode")

    async def _select(self, sid: str, ns: Any, *, want: str, category: str) -> None:
        try:
            if category == "model":
                models = getattr(ns, "models", None)
                avail = [m.model_id for m in getattr(models, "available_models", []) or []]
                if want in avail:
                    await self._conn.set_session_model(model_id=want, session_id=sid)
                    return
            elif category == "mode":
                modes = getattr(ns, "modes", None)
                avail = [m.id for m in getattr(modes, "available_modes", []) or []]
                if want in avail:
                    await self._conn.set_session_mode(mode_id=want, session_id=sid)
                    return
            # Fall back to a config_options select in the right category.
            for opt in getattr(ns, "config_options", None) or []:
                if getattr(opt, "category", None) == category:
                    values = [o.value for o in getattr(opt, "options", []) or []]
                    if want in values:
                        await self._conn.set_config_option(
                            config_id=opt.id, session_id=sid, value=want
                        )
                        return
            self._log.warning(
                "acp.selection_unavailable",
                agent=self._label,
                category=category,
                requested=want,
            )
        except Exception as exc:  # noqa: BLE001 — selection is best-effort
            self._log.warning(
                "acp.selection_failed", agent=self._label, category=category, error=str(exc)
            )

    async def prompt(self, acp_session_id: str, blocks: list[Any]) -> None:
        st = self.sessions.get(acp_session_id)
        if st is None:
            raise SpawnFailedError("ACP session not initialised")
        if st.turn_active:
            raise SessionBusyError("a turn is already in flight")
        st.turn_active = True
        st.cancelled = False
        st.notified_crash = False
        st.turn_task = asyncio.create_task(
            self._run_turn(acp_session_id, blocks), name=f"acp-turn-{acp_session_id}"
        )

    async def _run_turn(self, acp_session_id: str, blocks: list[Any]) -> None:
        st = self.sessions.get(acp_session_id)
        if st is None:
            return
        try:
            resp = await self._conn.prompt(
                prompt=blocks, session_id=acp_session_id, message_id=str(uuid4())
            )
            await self._drain_notifications()
            frame: dict[str, Any] = {"type": "session.result", "stop_reason": resp.stop_reason}
            usage = getattr(resp, "usage", None)
            if usage is not None:
                frame["usage"] = usage.model_dump(mode="json", by_alias=True, exclude_none=True)
            await st.on_event(frame)
        except asyncio.CancelledError:
            # User interrupt → finalize the turn as cancelled (the agent may
            # never respond to session/cancel). Process teardown/crash → stay
            # silent; close()/the exit watcher own those frames.
            if st.cancelled:
                await st.on_event({"type": "session.result", "stop_reason": "cancelled"})
                return
            raise
        except Exception as exc:  # noqa: BLE001 — surface transport/agent errors as a frame
            if not st.notified_crash:
                st.notified_crash = True
                # An auth rejection mid-turn is distinct from a crash so the
                # notify service can route it as ``auth_required`` (#24).
                code = (
                    "auth_required"
                    if isinstance(exc, acp.RequestError) and exc.code == _ACP_AUTH_REQUIRED_CODE
                    else "agent_crashed"
                )
                await st.on_event({"type": "session.error", "code": code, "message": str(exc)})
        finally:
            st.turn_active = False

    async def cancel(self, acp_session_id: str) -> bool:
        st = self.sessions.get(acp_session_id)
        if st is None or not st.turn_active:
            return False
        st.cancelled = True
        # Best-effort notify the agent; we finalize locally regardless since
        # agents don't reliably respond to session/cancel.
        if self._conn is not None:
            with contextlib.suppress(Exception):
                await self._conn.cancel(session_id=acp_session_id)
        if st.turn_task is not None and not st.turn_task.done():
            st.turn_task.cancel()
        return True

    async def end_session(self, acp_session_id: str) -> None:
        st = self.sessions.pop(acp_session_id, None)
        if st is None:
            return
        if st.turn_task is not None and not st.turn_task.done():
            st.turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await st.turn_task

    async def close(self) -> None:
        self._closing = True
        if self._exit_task is not None:
            self._exit_task.cancel()
            self._exit_task = None
        for st in list(self.sessions.values()):
            if st.turn_task is not None and not st.turn_task.done():
                st.turn_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await st.turn_task
        self.sessions.clear()
        if self._conn is not None:
            with contextlib.suppress(Exception):
                await self._conn.close()
            self._conn = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                with contextlib.suppress(asyncio.CancelledError):
                    await proc.wait()

    async def _drain_notifications(self) -> None:
        if self._rpc_queue is None:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._rpc_queue.join(), timeout=5.0)

    async def _pump_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace").rstrip("\n")
                # stderr is process-wide (no sessionId); fan out to all sessions.
                for st in list(self.sessions.values()):
                    await st.on_event({"type": "session.stderr", "line": text})
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 — never let stderr pumping crash the process
            return


class AcpSessionHandle:
    """Per-session view onto a profile's shared :class:`AcpAgentProcess`.

    Conforms to :class:`~blemees_agent.backends.AgentBackend` so the daemon's
    existing session/dispatch code drives it without change.
    """

    backend = "acp"

    def __init__(
        self,
        *,
        process: AcpAgentProcess,
        on_event: EventCallback,
        cwd: str | None,
        on_close: Any = None,
        permission_cb: Any = None,
        resume_native_id: str | None = None,
    ) -> None:
        self._process = process
        self._on_event = on_event
        self._cwd = cwd
        self._on_close = on_close  # supervisor callback for idle-reap accounting
        self._permission_cb = permission_cb
        self._resume_native_id = resume_native_id  # resume this agent session, if set (#23)
        self.native_session_id: str | None = None
        self.load_session: bool = False
        # True when resume was requested but the agent can't session/load:
        # the session is viewable (replayed history) but not drivable.
        self.view_only: bool = False
        self.model: str | None = process.agent.model

    @property
    def running(self) -> bool:
        return self._process.running and self.native_session_id is not None

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def turn_active(self) -> bool:
        return bool(self.native_session_id and self._process.is_turn_active(self.native_session_id))

    async def spawn(self) -> None:
        if self._resume_native_id is not None:
            await self._process.ensure_started()
            self.load_session = self._process.load_session
            if self._process.load_session:
                # Rehydrate the agent's conversation (#23).
                self.native_session_id = await self._process.resume_session(
                    native_session_id=self._resume_native_id,
                    cwd=self._cwd,
                    on_event=self._on_event,
                    permission_cb=self._permission_cb,
                )
            else:
                # Agent can't reload — viewable (from the event log) but not
                # drivable. No ACP session is created.
                self.view_only = True
                self.native_session_id = None
            return
        self.native_session_id = await self._process.new_session(
            cwd=self._cwd, on_event=self._on_event, permission_cb=self._permission_cb
        )
        self.load_session = self._process.load_session

    async def send_user_turn(self, message: dict[str, Any]) -> None:
        if self.native_session_id is None:
            raise SpawnFailedError("ACP session not initialised")
        blocks = _to_content_blocks(message)  # may raise ProtocolError
        await self._process.prompt(self.native_session_id, blocks)

    async def interrupt(self) -> bool:
        if self.native_session_id is None:
            return False
        return await self._process.cancel(self.native_session_id)

    async def close(self) -> None:
        if self.native_session_id is not None:
            await self._process.end_session(self.native_session_id)
        if self._on_close is not None:
            await self._on_close(self._process)

    async def wait_for_exit(self, timeout: float) -> bool:
        """Wait until this session's in-flight turn finishes (not process exit).

        The agent process is shared and long-lived, so "exit" for a session
        means its turn completed. Used by the daemon's shutdown grace.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self.turn_active and loop.time() < deadline:
            await asyncio.sleep(0.05)
        return not self.turn_active
