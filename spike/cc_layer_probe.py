#!/usr/bin/env -S uv run --python 3.11 --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["agent-client-protocol"]
# ///
"""
Spike: does the Claude Code layer survive the trip through claude-agent-acp?

Checks, against a scratch project at /tmp/spike-acp:
  1. CLAUDE.md       — project memory loaded? (codeword probe)
  2. skills          — user skills advertised in AvailableCommandsUpdate and
                       invocable via a /slash prompt? (spike-acp-probe fixture)
  3. plugins         — plugin-namespaced commands advertised?
  4. hooks           — SessionStart/Stop/UserPromptSubmit marker files touched?
                       (fixture must exist: /tmp/spike-acp/.claude/settings.json)

Throwaway; prints a JSON report. Gates the full-switch milestone and the
emotions plugin's survival (see project roadmap memory, 2026-06-10 grill).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import acp
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import AllowedOutcome, ClientCapabilities, RequestPermissionResponse

CWD = "/tmp/spike-acp"
AGENT = os.environ.get("SPIKE_AGENT", "claude-agent-acp")
KNOWN_USER_SKILLS = ["spike-acp-probe", "grill-me", "caveman", "tdd"]


def _dump(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj


class ProbeClient(acp.Client):
    def __init__(self) -> None:
        self.assistant_text: list[str] = []
        self.available_commands: list[str] = []
        self.permission_requests = 0

    async def request_permission(self, options, session_id, tool_call, **kw):
        self.permission_requests += 1
        chosen = next((o for o in options if o.kind.startswith("allow")), options[0])
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=chosen.option_id, outcome="selected")
        )

    async def session_update(self, session_id, update, **kw):
        name = type(update).__name__
        if name == "AgentMessageChunk":
            text = getattr(getattr(update, "content", None), "text", None)
            if text:
                self.assistant_text.append(text)
        elif name == "AvailableCommandsUpdate":
            cmds = getattr(update, "available_commands", None) or []
            self.available_commands = [getattr(c, "name", "?") for c in cmds]


async def main() -> None:
    report: dict = {"agent": AGENT, "cwd": CWD}
    markers = ["MARKER-sessionstart", "MARKER-promptsubmit", "MARKER-stop"]
    for m in markers:
        Path(CWD, m).unlink(missing_ok=True)
    hooks_fixture = Path(CWD, ".claude", "settings.json").exists()
    report["hooks_fixture_present"] = hooks_fixture

    client = ProbeClient()
    async with spawn_agent_process(client, AGENT, env=dict(os.environ)) as (conn, _proc):
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities()
        )
        ns = await conn.new_session(cwd=CWD, mcp_servers=[])
        sid = ns.session_id
        await asyncio.sleep(2)  # let AvailableCommandsUpdate land

        cmds = client.available_commands
        report["available_commands_total"] = len(cmds)
        report["user_skills_advertised"] = {k: k in cmds for k in KNOWN_USER_SKILLS}
        report["plugin_commands_sample"] = [c for c in cmds if ":" in c][:8]

        # Leg 1 — CLAUDE.md
        client.assistant_text.clear()
        await conn.prompt(
            prompt=[
                text_block(
                    "What is the project codeword? Reply with exactly the codeword, nothing else."
                )
            ],
            session_id=sid,
            message_id=str(uuid4()),
        )
        answer = "".join(client.assistant_text).strip()
        report["claude_md"] = {"answer": answer[:80], "loaded": "XYZZY-42" in answer}

        # Leg 2 — skill invocation via slash prompt
        client.assistant_text.clear()
        await conn.prompt(
            prompt=[text_block("/spike-acp-probe")],
            session_id=sid,
            message_id=str(uuid4()),
        )
        answer = "".join(client.assistant_text).strip()
        report["skill_invocation"] = {"answer": answer[:120], "fired": "SKILL-FIRED-OK" in answer}

        # Leg 3 — hooks (fixture permitting): markers after session+turns
        await asyncio.sleep(2)
        report["hooks"] = {m: Path(CWD, m).exists() for m in markers}

    print(json.dumps(report, indent=2))


asyncio.run(main())
