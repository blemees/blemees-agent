# ACP spike findings (blemees-agent#15)

**Date:** 2026-06-04 · **SDK:** `agent-client-protocol==0.10.1` · **ACP protocol_version:** `1`
**Probe:** [`acp_probe.py`](acp_probe.py) — acts as the ACP *client* (the role the daemon will play), spawns each agent over stdio, `initialize` with **empty** client capabilities, `new_session`, three prompt turns (liveness, multi-turn memory, tool-use/permission), auto-approving every permission request.

## Agents exercised (all real, ambient-authenticated)

| Agent | Invocation |
|---|---|
| claude-agent-acp 0.40.0 | `claude-agent-acp` (stdio) |
| codex-acp | `codex-acp` (stdio) |
| gemini 0.38.2 | `gemini --experimental-acp` |
| cursor-agent 2026.06.02 | `cursor-agent acp` |

## Result: **all four pass end-to-end.** ✅

Every agent completed all three turns, preserved multi-turn memory, and created the file via its own tools. No agent errored or hung under empty client capabilities.

| | claude-agent-acp | codex-acp | gemini | cursor-agent |
|---|---|---|---|---|
| turn1 stop_reason | end_turn | end_turn | end_turn | end_turn |
| multi-turn memory | ✅ | ✅ | ✅ | ✅ |
| tool-use file created | ✅ | ✅ | ✅ | ✅ |
| permission requests (default mode) | 0 | 1 | 1 | 0 |
| `load_session` | **true** | **true** | **true** | **true** |
| usage reported | yes (prompt+stream) | yes (stream) | no | no |
| native session caps | close, fork, list, resume | close, list | — | list |

---

## Load-bearing assumptions — VALIDATED

1. **Empty client capabilities work.** All four ran with `ClientCapabilities()` (no `fs`, no `terminal`) and did their own file IO. → spec §4 ("advertise no fs/terminal") **confirmed**; no fs/terminal server needed.
2. **Permission requests can be auto-answered.** codex/gemini sent `session/request_permission`; returning `AllowedOutcome(option_id=…)` unblocked them and the file was written. → relay/auto-answer design (#20) **confirmed**.
3. **`loadSession` is universal here.** All four advertise `load_session: true`. → the view-only degradation path (#23) is a rare edge case, not the norm. Conversational resume across daemon restart is broadly available.

---

## Assumption-breaking findings — SPEC CHANGES NEEDED

### A. ACP **does** carry token usage (reverses #21)

Earlier research said "no usage in ACP." False at protocol_version 1 / SDK 0.10:
- `PromptResponse.usage` is a real field (`input/output/cached_read/cached_write/thought/total_tokens`). **claude-agent-acp populates it.**
- There is a streaming `UsageUpdate` `session/update` variant. **claude-agent-acp (11×) and codex-acp (4×) emit it.**
- gemini and cursor-agent report no usage.

**Recommendation:** do **not** drop usage from `session.info`. Make it **optional/opportunistic** — accumulate from `UsageUpdate` and/or `PromptResponse.usage` when the agent provides it; omit when absent. (Still never scrape agent logs.) → amend spec §3.1 + §9.3 and issue **#21**.

### B. Model / mode / effort are **not** `new_session` params (reverses §2/§3)

`new_session` has no `model` argument. Selection is post-session and **heterogeneous**:
- **`SessionModeState` + `set_session_mode`** — gemini (`default/autoEdit/yolo/plan`), codex (`read-only/auto/full-access`), cursor (`agent/plan/ask`), claude (`auto/default/acceptEdits/plan/dontAsk/bypassPermissions`).
- **`SessionModelState` + `set_session_model`** — gemini (8 models), codex (3), cursor (28!). claude exposes **no** `models` here…
- **`config_options` + `set_config_option`** — claude & codex & cursor expose a `select`-typed option set covering `mode`, `model`, and `effort`/`reasoning_effort`. claude's *model* lives **only** in `config_options`, not `SessionModelState`.

**Recommendation:** a profile's `model`/`mode`/`effort` are applied **after `new_session`** by calling whichever mechanism the agent advertises — `set_session_model` / `set_session_mode` and/or `set_config_option`. The daemon must support all three and map the profile's desired model/mode onto the agent's advertised option ids (best-effort; warn if unavailable). → amend spec §2 (profile→session wiring) + §3 (step 3) and issues **#17** and **#3** (TUI mode/model picker should read these from the session, not assume).

### C. Agents advertise native session management (augments durability/listing)

`session_capabilities` exposes `list` (claude/codex/cursor), `resume` + `fork` + `close` (claude). These are **beyond** the ACP I researched. The daemon-owned registry (#21/#22) stays the uniform source of truth (gemini has none), but where an agent supports `list`/`resume` natively, it could augment recovery. Note for **#21–#23**; no change to the chosen design (registry remains authoritative).

### D. `session/update` class names (minor)

The SDK emits `ToolCallStart` / `ToolCallProgress` (discriminators `tool_call` / `tool_call_update`), `AgentMessageChunk`, `AgentThoughtChunk`, `AvailableCommandsUpdate`, `UsageUpdate`, `SessionInfoUpdate`, `CurrentModeUpdate`. Use these exact names in spec §3.1 and the TUI reducer (#2).

---

## Auth

All four authenticated via **ambient creds** (no `authenticate` call needed). Advertised `authMethods`: claude `[]` (pure ambient), codex `[AuthMethodAgent, EnvVarAuthMethod×2]`, gemini `[AuthMethodAgent×4]`, cursor `[AuthMethodAgent]`. → ambient-auth approach (spec §3 step 2) confirmed; `authenticate` support is a nice-to-have, not required for these agents.

---

## Decision: proceed to #16

No agent requires fs/terminal; the empty-caps + auto-permission design holds for all four. The fs/terminal fallback contingency (plan risk row) is **not triggered**. Two spec amendments (A: keep optional usage; B: model/mode via set_session_model/mode/config_option) should land before/with #16.

## Dependency

`agent-client-protocol>=0.10` added to `pyproject.toml` (`dependencies`), ending the stdlib-only stance as planned. Python ≥3.11 (already required) satisfies the SDK's ≥3.10.
