#!/usr/bin/env python3
"""A scriptable mock ACP *agent* for backend tests (#16).

Speaks the Agent Client Protocol over stdio via the ``agent-client-protocol``
SDK so ``AcpBackend`` can be exercised without a real agent binary. Behaviour
is driven by the prompt text so a single stub covers several scenarios:

* default          → stream two text chunks, stop ``end_turn``.
* contains "hang"  → emit one chunk, then sleep until cancelled (interrupt /
                     never-finishing-turn for shutdown force-kill tests).
* contains "finish"→ emit one chunk, sleep ~0.5s, finish ``end_turn`` (graceful
                     shutdown: turn completes within the grace window).
* contains "boom"  → raise, so the turn surfaces as an agent error.
* contains "die"   → hard-exit the process mid-turn (crash/recovery tests).

Run as: ``python fake_acp.py`` (stdio).
"""

from __future__ import annotations

import asyncio
import os

import acp
from acp import run_agent, update_agent_message_text
from acp.schema import (
    AgentCapabilities,
    InitializeResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ToolCallUpdate,
)


class FakeAgent(acp.Agent):
    def __init__(self) -> None:
        self._conn: acp.AgentSideConnection | None = None
        self._session_seq = 0

    def on_connect(self, conn: acp.AgentSideConnection) -> None:
        self._conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kw):
        # ``BLEMEES_FAKE_NO_LOAD`` simulates an agent that can't reload prior
        # sessions, so the daemon falls back to view-only on resume (#23).
        can_load = os.environ.get("BLEMEES_FAKE_NO_LOAD") != "1"
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(load_session=can_load),
        )

    async def new_session(self, cwd, additional_directories=None, mcp_servers=None, **kw):
        # ``BLEMEES_FAKE_AUTH_REQUIRED`` makes session/new reject with the ACP
        # auth error so the daemon surfaces ``auth_required`` at open (#24).
        if os.environ.get("BLEMEES_FAKE_AUTH_REQUIRED") == "1":
            raise acp.RequestError.auth_required({"detail": "log in first"})
        # Unique id per session so one process can multiplex several.
        self._session_seq += 1
        return NewSessionResponse(session_id=f"fake-session-{self._session_seq}")

    async def load_session(
        self, cwd, session_id, additional_directories=None, mcp_servers=None, **kw
    ):
        return None

    async def prompt(self, prompt, session_id, message_id=None, **kw):
        text = " ".join(
            b.text
            for b in prompt
            if getattr(b, "type", None) == "text" and getattr(b, "text", None)
        ).lower()

        if "boom" in text:
            raise RuntimeError("synthetic agent failure")

        if "needauth" in text:
            # Reject mid-turn with the ACP auth error (#24): the backend maps
            # this to a session.error{code: auth_required} frame.
            raise acp.RequestError.auth_required({"detail": "session expired"})

        if "die" in text:
            os._exit(1)  # hard crash mid-turn, no response

        assert self._conn is not None

        if "stall" in text:
            # Emit a chunk so the client can confirm the turn is in flight and
            # detach, then (after a beat) request permission — which now has no
            # owner and stalls, exercising the detached needs_attention path.
            await self._conn.session_update(session_id, update_agent_message_text("working"))
            await asyncio.sleep(0.3)
            resp = await self._conn.request_permission(
                session_id=session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id="tc-stall", title="Run a command", status="pending"
                ),
                options=[
                    PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
                    PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
                ],
            )
            decided = getattr(resp.outcome, "option_id", None) or "cancelled"
            await self._conn.session_update(
                session_id, update_agent_message_text(f"perm:{decided}")
            )
            return PromptResponse(stop_reason="end_turn")

        if "permit" in text:
            resp = await self._conn.request_permission(
                session_id=session_id,
                tool_call=ToolCallUpdate(
                    tool_call_id="tc1", title="Run a command", status="pending"
                ),
                options=[
                    PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
                    PermissionOption(
                        option_id="allow_always", name="Always allow", kind="allow_always"
                    ),
                    PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
                ],
            )
            decided = getattr(resp.outcome, "option_id", None) or "cancelled"
            await self._conn.session_update(
                session_id, update_agent_message_text(f"perm:{decided}")
            )
            return PromptResponse(stop_reason="end_turn")

        if "hang" in text:
            # Emit one chunk so the client can observe the turn is in flight,
            # then block until cancelled / killed.
            await self._conn.session_update(session_id, update_agent_message_text("working"))
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                return PromptResponse(stop_reason="cancelled")
            return PromptResponse(stop_reason="end_turn")

        if "finish" in text:
            await self._conn.session_update(session_id, update_agent_message_text("working"))
            await asyncio.sleep(0.5)
            await self._conn.session_update(session_id, update_agent_message_text(" done"))
            return PromptResponse(stop_reason="end_turn")

        await self._conn.session_update(session_id, update_agent_message_text("PONG"))
        await self._conn.session_update(session_id, update_agent_message_text(" done"))
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kw):
        return None


def main() -> None:
    asyncio.run(run_agent(FakeAgent()))


if __name__ == "__main__":
    main()
