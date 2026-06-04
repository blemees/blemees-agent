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

Run as: ``python fake_acp.py`` (stdio).
"""

from __future__ import annotations

import asyncio

import acp
from acp import run_agent, update_agent_message_text
from acp.schema import (
    AgentCapabilities,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
)


class FakeAgent(acp.Agent):
    def __init__(self) -> None:
        self._conn: acp.AgentSideConnection | None = None

    def on_connect(self, conn: acp.AgentSideConnection) -> None:
        self._conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kw):
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(load_session=True),
        )

    async def new_session(self, cwd, additional_directories=None, mcp_servers=None, **kw):
        return NewSessionResponse(session_id="fake-session-1")

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

        assert self._conn is not None

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
