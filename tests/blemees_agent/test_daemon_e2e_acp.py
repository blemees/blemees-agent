"""End-to-end tests against real, authenticated ACP agents (#26, Phase 7).

Each agent is gated by its own marker and auto-skipped unless the binary is on
PATH (an unauthenticated agent will surface auth errors at run time, so run
these only against agents you've logged into). They drive the *real* daemon —
no fake_acp — exercising a single turn, multi-turn memory within a live
session, owner/viewer attach fan-out, and interrupt.

Run a single agent's suite, e.g.::

    pytest -m requires_claude_acp tests/blemees_agent/test_daemon_e2e_acp.py

These never run in CI (the ACP agent binaries aren't installed there); they're
for local verification against the actual agents.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import uuid
from dataclasses import dataclass, field

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import _StreamClient, short_socket_path, socket_cleanup

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class AgentSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    # Alternative binary names to accept (e.g. cursor ships as both
    # `cursor-agent` and `agent`); the first one on PATH wins.
    aliases: tuple[str, ...] = ()

    def resolved_command(self) -> str | None:
        for candidate in (self.command, *self.aliases):
            if shutil.which(candidate):
                return candidate
        return None


# The ACP agents the migration validated (#15 spike). `command` is the binary
# the daemon spawns; `args` puts it in ACP mode.
AGENTS = [
    AgentSpec("claude_acp", "claude-agent-acp"),
    AgentSpec("codex_acp", "codex-acp", ["acp"]),
    AgentSpec("gemini_acp", "gemini", ["--experimental-acp"]),
    AgentSpec("cursor_acp", "cursor-agent", ["acp"], aliases=("agent",)),
]

# Opt-in: these hit real agents (slow, cost tokens, need a logged-in session),
# so the default `pytest` run always skips them. Set BLEMEES_E2E=1 to run the
# matrix against whichever agents are installed + authenticated.
_E2E_ENABLED = os.environ.get("BLEMEES_E2E") == "1"


def _skip_reason(spec: AgentSpec) -> str | None:
    if not _E2E_ENABLED:
        return "set BLEMEES_E2E=1 to run real-agent e2e tests"
    if spec.resolved_command() is None:
        names = " / ".join((spec.command, *spec.aliases))
        return f"none of [{names}] on PATH"
    return None


# One parametrize entry per agent, carrying its marker and the skip gate.
_AGENT_PARAMS = [
    pytest.param(
        spec,
        id=spec.name,
        marks=[
            getattr(pytest.mark, f"requires_{spec.name}"),
            pytest.mark.skipif(_skip_reason(spec) is not None, reason=_skip_reason(spec) or ""),
        ],
    )
    for spec in AGENTS
]


@contextlib.asynccontextmanager
async def _daemon(spec: AgentSpec):
    sock = short_socket_path(f"blemeesd-e2e-{spec.name}")
    with socket_cleanup(sock):
        cfg = Config(
            socket_path=str(sock),
            agent_command=spec.resolved_command() or spec.command,
            agent_args=list(spec.args),
            idle_timeout_s=60,
        )
        daemon = Daemon(cfg, configure("error"))
        await daemon.start()
        serve = asyncio.create_task(daemon.serve_forever())
        try:
            yield str(sock)
        finally:
            daemon.request_shutdown()
            try:
                await asyncio.wait_for(serve, timeout=10.0)
            except TimeoutError:
                serve.cancel()


async def _connect(sock: str) -> _StreamClient:
    reader, writer = await asyncio.open_unix_connection(sock)
    c = _StreamClient(reader, writer)
    await c.send({"type": "hello", "client": "e2e/0", "protocol": PROTOCOL_VERSION})
    assert (await c.recv())["type"] == "hello_ack"
    return c


async def _wait_or_skip(c: _StreamClient, pred, *, what: str, timeout: float) -> dict:
    """wait_for, but an unresponsive agent → skip (it isn't authenticated /
    available here), matching this suite's 'installed + authenticated' gate."""
    try:
        return await c.wait_for(pred, timeout=timeout)
    except TimeoutError:
        pytest.skip(f"agent unresponsive waiting for {what} ({timeout}s) — authenticated?")
        raise  # unreachable (pytest.skip raises); explicit for static analysis


async def _open(c: _StreamClient, sid: str) -> dict:
    await c.send({"type": "session.open", "id": "o", "session_id": sid, "options": {}})
    return await _wait_or_skip(
        c, lambda e: e.get("type") == "session.opened", what="session.opened", timeout=30.0
    )


def _chunk_text(frames: list[dict]) -> str:
    """Concatenate agent_message_chunk text from a turn's session.update frames."""
    out = []
    for f in frames:
        if f.get("type") != "session.update":
            continue
        upd = f.get("update", {})
        if upd.get("sessionUpdate") == "agent_message_chunk":
            content = upd.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text":
                out.append(content.get("text", ""))
    return "".join(out)


async def _drive(c: _StreamClient, sid: str, prompt: str, *, timeout: float = 60.0) -> list[dict]:
    """Send a prompt and collect frames through session.result."""
    await c.send({"type": "session.prompt", "session_id": sid, "prompt": prompt})
    frames: list[dict] = []
    while True:
        try:
            evt = await c.recv(timeout=timeout)
        except TimeoutError:
            pytest.skip(f"agent unresponsive within {timeout}s — authenticated?")
        frames.append(evt)
        if evt.get("type") == "session.result":
            return frames
        if evt.get("type") in ("session.error", "error"):
            pytest.skip(f"agent surfaced {evt.get('code')}: {evt.get('message')} (auth?)")


@pytest.mark.parametrize("spec", _AGENT_PARAMS)
async def test_turn_then_result(spec):
    async with _daemon(spec) as sock:
        c = await _connect(sock)
        sid = str(uuid.uuid4())
        try:
            await _open(c, sid)
            frames = await _drive(c, sid, "Reply with exactly the word: pong")
            result = next(f for f in frames if f["type"] == "session.result")
            assert result["stop_reason"] == "end_turn"
            assert _chunk_text(frames).strip() != ""
        finally:
            await c.close()


@pytest.mark.parametrize("spec", _AGENT_PARAMS)
async def test_multi_turn_memory_within_session(spec):
    async with _daemon(spec) as sock:
        c = await _connect(sock)
        sid = str(uuid.uuid4())
        try:
            await _open(c, sid)
            await _drive(c, sid, "Remember the number forty-two. Reply with just: OK")
            frames = await _drive(c, sid, "What number did I ask you to remember?")
            # Agent-agnostic: accept digits or words, any casing/spacing.
            answer = _chunk_text(frames).lower().replace(" ", "").replace("-", "")
            assert "42" in answer or "fortytwo" in answer, f"memory not recalled: {answer!r}"
        finally:
            await c.close()


@pytest.mark.parametrize("spec", _AGENT_PARAMS)
async def test_owner_and_viewer_fan_out(spec):
    async with _daemon(spec) as sock:
        owner = await _connect(sock)
        viewer = await _connect(sock)
        sid = str(uuid.uuid4())
        try:
            await _open(owner, sid)
            await viewer.send(
                {"type": "session.attach", "id": "a", "session_id": sid, "as": "viewer"}
            )
            await _wait_or_skip(
                viewer,
                lambda e: e.get("type") == "session.attached",
                what="session.attached",
                timeout=10.0,
            )
            await owner.send({"type": "session.prompt", "session_id": sid, "prompt": "Say pong."})
            # The viewer sees the streamed result too.
            await _wait_or_skip(
                viewer,
                lambda e: e.get("type") == "session.result" and e.get("session_id") == sid,
                what="viewer fan-out result",
                timeout=60.0,
            )
        finally:
            await owner.close()
            await viewer.close()


@pytest.mark.parametrize("spec", _AGENT_PARAMS)
async def test_interrupt_cancels_turn(spec):
    async with _daemon(spec) as sock:
        c = await _connect(sock)
        sid = str(uuid.uuid4())
        try:
            await _open(c, sid)
            await c.send(
                {
                    "type": "session.prompt",
                    "session_id": sid,
                    "prompt": "Count slowly from 1 to 100, one number per line.",
                }
            )
            # Let the turn get going, then interrupt.
            await _wait_or_skip(
                c, lambda e: e.get("type") == "session.update", what="first update", timeout=30.0
            )
            await c.send({"type": "session.cancel", "session_id": sid})
            result = await _wait_or_skip(
                c,
                lambda e: e.get("type") == "session.result",
                what="cancelled result",
                timeout=30.0,
            )
            assert result["stop_reason"] in ("cancelled", "end_turn")
        finally:
            await c.close()
