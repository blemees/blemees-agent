#!/usr/bin/env -S uv run --python 3.11 --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["agent-client-protocol"]
# ///
"""
Spike probe for blemees-agent#15 — validate ACP agents under an empty
client-capability set, acting as the ACP *client* (the role blemees-agentd
will play). Throwaway: prints a JSON report; does not touch the daemon code.

Run:  uv run spike/acp_probe.py            # all agents
      uv run spike/acp_probe.py codex-acp  # one agent by name
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from uuid import uuid4

import acp
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import AllowedOutcome, ClientCapabilities, RequestPermissionResponse

AGENTS = [
    {"name": "claude-agent-acp", "cmd": "/Users/juanheyns/.n/bin/claude-agent-acp", "args": []},
    {"name": "codex-acp", "cmd": "/opt/homebrew/bin/codex-acp", "args": []},
    {"name": "gemini", "cmd": "gemini", "args": ["--experimental-acp"]},
    {"name": "cursor-agent", "cmd": "cursor-agent", "args": ["acp"]},
]

PER_AGENT_TIMEOUT = 180.0


def _dump(obj):
    if obj is None:
        return None
    if isinstance(obj, list):
        return [_dump(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj


class ProbeClient(acp.Client):
    """Auto-approves every permission request; records what streams back."""

    def __init__(self) -> None:
        self.update_counts: dict[str, int] = {}
        self.update_samples: dict[str, dict] = {}
        self.permission_requests = 0
        self.assistant_text: list[str] = []

    async def request_permission(self, options, session_id, tool_call, **kw):
        self.permission_requests += 1
        chosen = next((o for o in options if o.kind.startswith("allow")), options[0])
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=chosen.option_id, outcome="selected")
        )

    async def session_update(self, session_id, update, **kw):
        name = type(update).__name__
        self.update_counts[name] = self.update_counts.get(name, 0) + 1
        if name not in self.update_samples:
            self.update_samples[name] = _dump(update)
        # best-effort capture of assistant text
        if name == "AgentMessageChunk":
            content = getattr(update, "content", None)
            text = getattr(content, "text", None)
            if text:
                self.assistant_text.append(text)


async def _probe(agent: dict) -> dict:
    res: dict = {"name": agent["name"], "cmd": agent["cmd"]}
    client = ProbeClient()
    async with spawn_agent_process(
        client, agent["cmd"], *agent["args"], env=dict(os.environ)
    ) as (conn, _proc):
        # 1) initialize with EMPTY client capabilities (no fs, no terminal)
        init = await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
        )
        caps = init.agent_capabilities
        sc = getattr(caps, "session_capabilities", None)
        res["initialize"] = {
            "protocol_version": init.protocol_version,
            "load_session": getattr(caps, "load_session", None),
            "session_capabilities": _dump(sc),
            "prompt_capabilities": _dump(getattr(caps, "prompt_capabilities", None)),
            "auth_methods": [type(a).__name__ for a in (init.auth_methods or [])],
        }

        # 2) new session
        cwd = tempfile.mkdtemp(prefix=f"acp-spike-{agent['name']}-")
        ns = await conn.new_session(cwd=cwd, mcp_servers=[])
        sid = ns.session_id
        res["new_session"] = {
            "session_id_kind": "uuid-like" if "-" in sid else "opaque",
            "modes": _dump(getattr(ns, "modes", None)),
            "models": _dump(getattr(ns, "models", None)),
            "config_options": _dump(getattr(ns, "config_options", None)),
        }

        # 3) turn 1 — liveness + does empty-caps work + is usage populated?
        client.assistant_text.clear()
        p1 = await conn.prompt(
            prompt=[text_block("Reply with exactly the single word PONG and nothing else.")],
            session_id=sid,
            message_id=str(uuid4()),
        )
        res["turn1"] = {
            "stop_reason": p1.stop_reason,
            "usage": _dump(getattr(p1, "usage", None)),
            "assistant_text": "".join(client.assistant_text)[:200],
        }

        # 4) turn 2 — in-process multi-turn memory
        client.assistant_text.clear()
        p2 = await conn.prompt(
            prompt=[text_block("What single word did I just ask you to say? Reply with only that word.")],
            session_id=sid,
            message_id=str(uuid4()),
        )
        res["turn2_memory"] = {
            "stop_reason": p2.stop_reason,
            "assistant_text": "".join(client.assistant_text)[:200],
        }

        # 5) turn 3 — provoke a tool/permission request with empty caps
        client.assistant_text.clear()
        before = client.permission_requests
        p3 = await conn.prompt(
            prompt=[text_block(
                "Use your tools to create a file named spike.txt containing the word hi "
                "in the current working directory, then tell me done."
            )],
            session_id=sid,
            message_id=str(uuid4()),
        )
        res["turn3_tooluse"] = {
            "stop_reason": p3.stop_reason,
            "permission_requests": client.permission_requests - before,
            "assistant_text": "".join(client.assistant_text)[:200],
            "file_created": os.path.exists(os.path.join(cwd, "spike.txt")),
        }

        res["updates_seen"] = client.update_counts
        res["update_samples"] = client.update_samples
    return res


async def _run_one(agent: dict) -> dict:
    t0 = time.time()
    try:
        res = await asyncio.wait_for(_probe(agent), timeout=PER_AGENT_TIMEOUT)
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 — spike: capture everything
        res = {"name": agent["name"], "ok": False, "error": f"{type(e).__name__}: {e}"}
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


async def main() -> None:
    only = set(sys.argv[1:])
    agents = [a for a in AGENTS if not only or a["name"] in only]
    results = []
    for agent in agents:
        print(f"--- probing {agent['name']} ---", file=sys.stderr, flush=True)
        results.append(await _run_one(agent))
    print(json.dumps({"protocol_version": PROTOCOL_VERSION, "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
