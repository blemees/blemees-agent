---
title: ACP migration spec
nav_order: 3
permalink: /acp-spec/
---

# blemees-agentd — ACP supervisor + proxy (`blemees/3`)

**Status:** Design (defined 2026-06-04; core assumptions validated by the
#15 spike against claude-agent-acp / codex-acp / gemini / cursor-agent —
SDK `agent-client-protocol 0.10.1`, ACP `protocol_version 1`). Supersedes the native-translation
architecture of `blemees/2` (`docs/spec.md`).
**Protocol:** `blemees/3` — clean break, no compatibility shims.
**Language:** Python 3.11+. Takes its first runtime dependencies:
[`agent-client-protocol`](https://pypi.org/project/agent-client-protocol/)
(the ACP Python SDK) and its transitive `pydantic`.
**Target OS:** Linux, macOS.

> This document is the authoritative design for the ACP rearchitecture. It
> describes the target state, not an increment. The phased path to get
> there is in [`acp-migration-plan.md`](acp-migration-plan.md).

---

## 0. What changes, in one paragraph

`blemees-agentd` stops being a *translator* of two hard-coded backends
(`claude -p` stream-json, `codex mcp-server` JSON-RPC) and becomes a
**supervisor + semantic proxy** for any agent that speaks the
[Agent Client Protocol](https://agentclientprotocol.com). The daemon plays
the ACP **Client** role toward agent subprocesses; the conversation flows
in ACP's vocabulary (`session/update`, content blocks, tool calls) instead
of the retired `agent.*` vocabulary. `blemees-tui` stays the only
northbound client and keeps talking a blemees-native control envelope —
which now *carries* ACP payloads rather than translated ones. The daemon's
real value-add (multiplexing, viewing, replay, resume, supervision) is
preserved and extended with **profiles**, a **needs-attention/notify**
service, and **full durability across daemon restarts**.

The two retired modules — `backends/translate_claude.py` and
`backends/translate_codex.py` — and the entire `agent.*` event vocabulary
are deleted.

---

## 1. Roles and topology

```
┌─ blemees-tui ─────────┐        Unix socket          ┌─ blemees-agentd ───────────────────────────┐
│  (only northbound     │  blemees/3 control+data     │  Supervisor + Semantic Proxy               │
│   client, ever)       │ ───────────────────────────▶│                                            │
│  - ACP Pydantic models│        (newline JSON)       │  ┌─ Profile "claude-sonnet" ─────────────┐ │
│    shared from SDK     │◀─────────────────────────── │  │  one ACP agent process (stdio)        │ │
└───────────────────────┘                             │  │  claude-agent-acp                      │ │
                                                       │  │   ├─ ACP session s1  (multiplexed)    │ │
   Daemon is ACP **Client** toward each agent.         │  │   └─ ACP session s2                   │ │
   Daemon is **server** of the blemees/3 protocol      │  └────────────────────────────────────────┘ │
   toward the TUI.                                      │  ┌─ Profile "codex-ro" ──────────────────┐ │
                                                       │  │  codex-acp  ── ACP session s3         │ │
                                                       │  └────────────────────────────────────────┘ │
                                                       │  Registry (persistent) · Event logs · Notify │
                                                       └────────────────────────────────────────────┘
```

- **One agent process per profile.** ACP multiplexes many sessions over one
  stdio connection (demuxed by `sessionId`); the daemon uses that natively.
- **Daemon is the ACP Client.** It runs the `initialize` handshake, issues
  `session/new` / `session/prompt` / `session/cancel` / `session/load` /
  `session/set_mode`, and *answers* the agent's client-bound requests
  (`session/request_permission`; and `fs/*` / `terminal/*` are declined by
  capability — see §4).
- **TUI is decoupled from agents.** Agent processes are children of the
  daemon, not the TUI. Restarting/killing the TUI never drops a session.

---

## 2. Profiles, agents, sessions

The model is **Profile → Agent → Session**:

- A **profile** is a named container of one or more **agents** (plus
  profile-level cross-cutting config — permission policy §5, notify §6).
- An **agent** is an independently-configured ACP agent (its own binary, CLI
  args, model/mode, cwd, MCP servers, env). Two agents in one profile may be
  the *same vendor* with different configs (e.g. `claude` on sonnet and
  `claude-opus` on opus). The agent is the **unit of process supervision**:
  at most one running ACP process per agent.
- A **session** is an ACP session multiplexed inside an agent's process.

### 2.1 Agent fields

| Field | Purpose |
|---|---|
| `name` | Agent id within its profile (kebab-case). |
| `agent_command` / `agent_args` / `env` | How to spawn the ACP agent (e.g. `claude-agent-acp`, `codex-acp`, `gemini --experimental-acp`). |
| `model` / `mode` | Desired model and permission mode. **Not** `session/new` params — applied *after* the session opens via whichever mechanism the agent advertises (§3, finding B): `set_session_model` (`SessionModelState`), `set_session_mode` (`SessionModeState`), and/or `set_config_option` (`config_options`). Mapped best-effort; a miss warns, never fails. |
| `cwd` | Default working directory for new sessions. |
| `mcp_servers` | MCP server configs injected verbatim into every `session/new.mcpServers` / `session/load.mcpServers` for this agent. |

Profile-level fields (`permission_policy` §5, `notify` §6) apply across the
profile's agents and are added with those features.

### 2.2 Config shape

```toml
# Flat sugar: a profile with fields directly under it is one "default" agent.
[profiles.solo]
agent_command = "claude-agent-acp"
model = "sonnet"

# Multi-agent profile.
[profiles.work.agents.claude]
agent_command = "claude-agent-acp"
model = "sonnet"
[profiles.work.agents.codex]
agent_command = "codex-acp"
args = ["acp"]
```

A built-in `default` profile with a `default` agent is always synthesised
from the daemon's `agent_command` / `agent_args`, so `session.open` works
with no config. `session.open {profile?, agent?}` defaults to
`default`/`default` (or the profile's sole / `default`-named agent).

### 2.3 Lifecycle

- **Lazy start (default).** An agent's process spawns on the first
  `session.open` against it (or an explicit `profile.start`, which starts all
  of a profile's agents).
- **Idle reap.** When an agent's last session closes, its process is reaped
  after the idle timeout.
- **Definition.** Profiles come from the config file's `[profiles.*]` tables
  (#17) and, later, over-wire CRUD (#25).

### 2.4 MCP and ACP

The daemon never speaks MCP. An agent's `mcp_servers` (stdio entries for now;
HTTP/SSE later) are passed to the *agent* in `session/new.mcpServers`; the
agent connects to them as their MCP client and exposes their tools to the
model.

---

## 3. ACP southbound integration

The daemon uses the ACP Python SDK's `Client` base class over each agent's
stdio. Per process:

1. **`initialize`** — send `protocolVersion`, `clientCapabilities`,
   `clientInfo`. We advertise **no** `fs` and **no** `terminal` (§4). Record
   the agent's `agentCapabilities` (esp. `loadSession`) and `authMethods`.
2. **Auth** — if `authMethods` is non-empty and the agent reports it needs
   auth, call `authenticate`; otherwise rely on ambient creds in the
   process env (the profile's `agent.env` carries `ANTHROPIC_*` /
   `CLAUDE_CONFIG_DIR` / `OPENAI_API_KEY` / `GEMINI_API_KEY` etc.). An
   `auth_required` error surfaces as a `needs_attention` item + notify (§6).
3. **`session/new`** `{cwd, mcpServers}` per profile → store the returned
   `sessionId` (the agent's id) against the blemees session record. Record
   `modes` (`SessionModeState`), `models` (`SessionModelState`), and
   `config_options`. **`session/new` takes no `model` argument** (finding B).
3a. **Apply profile selection** — to honour the profile's `model`/`mode`/
   `effort`, call the mechanism the agent advertised: `set_session_model`,
   `set_session_mode`, and/or `set_config_option`. This is heterogeneous —
   e.g. gemini/codex/cursor expose models via `SessionModelState`, while
   claude-agent-acp exposes its model **only** through `config_options`.
   Map best-effort onto the agent's option ids; warn (don't fail) when a
   requested model/mode isn't offered.
4. **Turns** — `session/prompt {sessionId, prompt: ContentBlock[]}`; stream
   `session/update` notifications back; the turn ends when the agent
   responds with `{stopReason}`.
5. **Resume** — on reopen after a daemon restart, if the agent advertised
   `loadSession`, call `session/load {sessionId, cwd, mcpServers}`; the
   agent replays history as `session/update`s, which the daemon discards for
   model-context purposes (its own event log already drives view-replay,
   §7.3) and then becomes live. If `loadSession` is unsupported, the session
   is **view-only** after restart and flagged as such.

### 3.1 ACP `session/update` variants (carried verbatim)

SDK class names (discriminator in parens): `AgentMessageChunk`
(`agent_message_chunk`), `AgentThoughtChunk` (`agent_thought_chunk`),
`UserMessageChunk` (`user_message_chunk`), `ToolCallStart` (`tool_call`:
`toolCallId, title, kind, status, content, locations, rawInput, rawOutput`),
`ToolCallProgress` (`tool_call_update`: `toolCallId` + changed fields),
`AgentPlanUpdate` (`plan`: `entries[]`), `AvailableCommandsUpdate`,
`CurrentModeUpdate`, `UsageUpdate`, `SessionInfoUpdate`. Content blocks are
MCP-shaped: `text`, `image`, `audio`, `resource`, `resource_link`.
`stopReason` ∈ `end_turn | max_tokens | max_turn_requests | refusal | cancelled`.

> **Token usage (opportunistic).** Contrary to the original design
> assumption, ACP at `protocol_version 1` **does** carry usage: a
> `UsageUpdate` `session/update` variant and a `PromptResponse.usage`
> field (`input/output/cached_read/cached_write/thought/total_tokens`).
> The #15 spike confirmed claude-agent-acp populates both and codex-acp
> streams `UsageUpdate` (gemini/cursor report none). The daemon therefore
> **surfaces usage when the agent provides it** (accumulated from
> `UsageUpdate` and/or `PromptResponse.usage`) and omits it otherwise. It
> still never scrapes agent logs. See §9.3.

---

## 4. Client capabilities the daemon advertises

`clientCapabilities = {}` — neither `fs.readTextFile`, `fs.writeTextFile`,
nor `terminal`. Consequences:

- The agent does its **own** filesystem and terminal IO (exactly as headless
  `claude -p` does today). There are no editor buffers in a daemon to
  respect.
- The only client-bound request the daemon must service is
  `session/request_permission` (§5). `fs/*` and `terminal/*` are never
  called by a spec-conformant agent because the capabilities are absent.

This keeps the daemon's ACP surface tiny: it sends requests and answers
exactly one kind of callback.

---

## 5. Permissions and policy

`session/request_permission {sessionId, toolCall, options[]}` arrives at the
daemon (the ACP Client). The **profile's `permission_policy`** decides the
answer:

- `allow` — auto-select the first `allow_*` option. (Equivalent to today's
  `bypassPermissions`.)
- `deny` — auto-select the first `reject_*` option.
- `relay` (**default**) — forward the request to the session's **owner**
  (§8). The owner answers; the daemon relays the chosen `optionId` back.
  - When **no owner is attached**, behaviour is the profile's
    `detached` setting: `stall` (default — hold the turn open, mark the
    session `needs_attention`, fire a notification), `allow`, or `deny`.
- A per-tool / per-`kind` map may override the default for specific tool
  kinds (e.g. `{execute: relay, read: allow}`).

`allow_always` / `reject_always` selections are remembered for that session
(and optionally promoted to the profile policy by the owner).

Permission relay is **symmetric** with the SDK roles: the agent's
`request_permission` to the daemon becomes a `session.request_permission`
frame to the owner TUI; the owner's `session.permission_response` becomes
the daemon's ACP response. A pending permission on one session does **not**
block other sessions in the same process (async JSON-RPC).

---

## 6. Notify service

The daemon models **`needs_attention`** as a per-session state, entered when
a session needs the owner and none is attached. On *entry*, it fires a
notification; the outstanding set is exposed so an attaching TUI sees the
queue immediately.

**Triggers** (configurable per profile; defaults on):
- a relayed `session/request_permission` with no attached owner (the
  `detached: stall` case);
- `auth_required` / auth failure;
- agent process crash / spawn failure.

(`turn complete while detached` is deliberately **not** a default trigger.)

**Sinks.** The daemon emits one structured notification event; sinks consume
it. The primary built-in sink is a **pluggable outbound webhook**: an
HTTP `POST` of a JSON payload to a configured URL (per-profile, with a
global fallback), so the user routes it to ntfy / Pushover / Slack /
Discord / a custom service. Additional sinks (local desktop notification,
socket subscriber) may be added later behind the same event.

Webhook payload (shape):
```json
{
  "type": "blemees.notify",
  "reason": "permission_pending | auth_required | agent_crashed",
  "profile": "claude-sonnet",
  "session_id": "…",
  "title": "blemees: permission needed",
  "detail": "Tool 'Bash' wants to run in /home/u/proj",
  "ts_ms": 1769000000000
}
```

---

## 7. Durability, registry, replay

Durability promise: **survive daemon restarts** (the strongest tier).

### 7.1 Registry (source of truth)

The daemon owns a persistent registry — profiles and the sessions it has
created (with `profile`, agent `sessionId`, `cwd`, model, current mode,
created/last-active timestamps, `needs_attention`). It is the authoritative
answer to "what sessions exist." **The daemon never scans agent-specific
transcript directories** (`~/.claude/projects`, `~/.codex/sessions`) — that
would be agent-specific and break generality.

### 7.2 State directory (always-on)

`$XDG_STATE_HOME/blemees/agentd/` (default `~/.local/state/...`):
- `registry.json` (atomic-rename writes) — profiles + sessions.
- `sessions/<session_id>.jsonl` — append-only ACP `session/update` log.

Persistence is no longer opt-in (it was `BLEMEES_AGENTD_EVENT_LOG_DIR` in
`blemees/2`); full durability requires it.

### 7.3 Replay and resume

- Every frame the daemon emits for a session carries a monotonic `seq`
  (per session, from 1) and is appended to the session log.
- **View replay** (always works): a (re)attaching client passes
  `last_seen_seq`; the daemon replays buffered/logged frames with
  `seq > last_seen_seq`, then live. A `replay_gap` frame is emitted if the
  in-memory ring rolled past the requested seq and the durable log can't
  cover it.
- **Conversational resume** (model remembers): on daemon restart the daemon
  respawns each profile's process and calls `session/load` per session
  **iff** the agent advertises `loadSession`. Otherwise the session is
  re-listed as **view-only** (history visible from the log; new turns start
  a fresh ACP session under the same profile, with a clear signal to the
  TUI). This mirrors the old Codex cross-process-resume caveat, now
  generalised to "depends on the agent's `loadSession`."

---

## 8. Owner / viewer model

Preserved from `blemees/2`'s owner/watcher split, restated for ACP:

- **Owner** — the one connection that may drive a session: `session.prompt`,
  `session.cancel`, `session.set_mode`, `session.close`, and answer
  `session.request_permission`.
- **Viewer** — any number of connections that `session.attach` read-only.
  They receive the fan-out of `session.update` / `session.result` /
  `session.stderr` / `session.error` / `replay_gap` / `needs_attention`
  (with replay on attach), but cannot drive and do not receive permission
  requests.
- **Takeover** — a connection may `session.attach` as owner to a session
  already owned; the prior owner gets `session.taken {by_peer_pid?}` and
  drops to detached. The daemon does not arbitrate ping-pong.

---

## 9. Wire protocol (`blemees/3`)

Transport and framing unchanged from `blemees/2`: `AF_UNIX` stream socket,
UTF-8 newline-delimited JSON, one object per line, 16 MiB max line, `0600`
perms, socket-path resolution order identical (`$BLEMEES_AGENTD_SOCKET` →
`$XDG_RUNTIME_DIR/blemees/agentd.sock` → `/tmp/blemees-agentd-<uid>.sock`).

Two planes, both `blemees-agentd.*`-namespaced; the data plane embeds ACP
objects.

### 9.1 Handshake
```json
→ {"type":"hello","client":"blemees-tui/0.2","protocol":"blemees/3"}
← {"type":"hello_ack","daemon":"blemees-agentd/0.2","protocol":"blemees/3","pid":123,
   "agents":{"claude-agent-acp":"…","codex-acp":"…","gemini":"…"},
   "profiles":["claude-sonnet","codex-ro"]}
```
`agents` = ACP agent binaries detected on `$PATH` (best-effort). Mismatched
`protocol` → `error{protocol_mismatch}` and close.

### 9.2 Profiles
```json
→ {"type":"profile.list","id":"p1"}
← {"type":"profiles","id":"p1","profiles":[ { …profile fields, "running":true, "sessions":2 } ]}

→ {"type":"profile.create","id":"p2","profile":{ "name":"claude-sonnet","agent":{…},"model":"sonnet","permission_policy":{"mode":"relay","detached":"stall"},"mcp_servers":[…],"notify":{…} }}
← {"type":"profile.created","id":"p2","name":"claude-sonnet"}

→ {"type":"profile.start","id":"p3","name":"claude-sonnet"}   // explicit, else lazy on first open
→ {"type":"profile.stop","id":"p4","name":"claude-sonnet"}
→ {"type":"profile.update","id":"p5","name":"…","profile":{…}}
→ {"type":"profile.delete","id":"p6","name":"…"}
```

### 9.3 Sessions
```json
→ {"type":"session.open","id":"s1","profile":"claude-sonnet","session_id":"<uuid>","resume":false,"last_seen_seq":0,"cwd":"/home/u/proj"}
← {"type":"session.opened","id":"s1","session_id":"<uuid>","profile":"claude-sonnet","modes":{…},"last_seq":0,"view_only":false}

→ {"type":"session.prompt","session_id":"…","prompt":[{"type":"text","text":"Hello"}]}   // ACP ContentBlock[]
→ {"type":"session.cancel","session_id":"…"}
→ {"type":"session.set_mode","session_id":"…","mode_id":"acceptEdits"}      // → ACP set_session_mode
→ {"type":"session.set_model","session_id":"…","model_id":"sonnet"}         // → ACP set_session_model
→ {"type":"session.set_config_option","session_id":"…","option_id":"effort","value":"high"}  // → ACP set_config_option
→ {"type":"session.close","id":"sc","session_id":"…","delete":false}
← {"type":"session.closed","id":"sc","session_id":"…"}

→ {"type":"session.list","id":"sl","profile":null,"live":null}     // filters compose; registry-backed
← {"type":"sessions","id":"sl","sessions":[ {"session_id":"…","profile":"…","attached":true,"owner_pid":123,"model":"…","title":"…","cwd":"…","mode":"…","needs_attention":false,"view_only":false,"last_seq":47,"turn_active":false,"started_at_ms":…,"last_active_at_ms":…} ]}

→ {"type":"session.info","id":"si","session_id":"…"}
← {"type":"session.info_reply","id":"si","session_id":"…","profile":"…","model":"…","cwd":"…","mode":"…","turns":5,"attached":true,"running":true,"view_only":false,"needs_attention":false,"last_turn_at_ms":…,"last_seq":42,"usage":{"input_tokens":3824,"output_tokens":5,"cached_write_tokens":20022,"total_tokens":23851}}
```
> `session.info.usage` is **optional** — present only when the agent reports
> usage (accumulated from ACP `UsageUpdate` / `PromptResponse.usage`; see
> §3.1), omitted otherwise. This reverses the earlier "drop usage" plan
> after the #15 spike found usage is in-protocol at `protocol_version 1`
> (claude-agent-acp and codex-acp report it; gemini/cursor don't). The daemon
> never scrapes agent logs for it.

### 9.4 Attach / view
```json
→ {"type":"session.attach","id":"a1","session_id":"…","as":"owner","last_seen_seq":0}   // or "as":"viewer"
← {"type":"session.attached","id":"a1","session_id":"…","role":"owner","last_seq":42,"needs_attention":true}
→ {"type":"session.detach","id":"a2","session_id":"…"}
← {"type":"session.taken","session_id":"…","by_peer_pid":456}   // pushed to prior owner
```

### 9.5 Data plane (daemon → client)
```json
{"type":"session.update","session_id":"…","seq":7,"profile":"…","update":{ /* verbatim ACP SessionNotification.update */ }}
{"type":"session.result","session_id":"…","seq":12,"stop_reason":"end_turn"}
{"type":"session.request_permission","session_id":"…","seq":9,"request_id":"r1","tool_call":{…},"options":[{"optionId":"a","name":"Allow","kind":"allow_once"},{"optionId":"d","name":"Reject","kind":"reject_once"}]}
{"type":"session.stderr","session_id":"…","line":"…"}                  // rate-limited
{"type":"session.error","session_id":"…","code":"agent_crashed","message":"…"}
{"type":"session.needs_attention","session_id":"…","reason":"permission_pending"}
{"type":"session.attention_cleared","session_id":"…"}
{"type":"replay_gap","session_id":"…","since_seq":42,"first_available_seq":71}
```
Owner → daemon, answering a permission request:
```json
{"type":"session.permission_response","session_id":"…","request_id":"r1","outcome":"selected","option_id":"a"}
{"type":"session.permission_response","session_id":"…","request_id":"r1","outcome":"cancelled"}
```

### 9.6 Liveness / status
```json
→ {"type":"ping","id":"…","data":"…"}      ← {"type":"pong","id":"…","data":"…"}
→ {"type":"status","id":"…"}
← {"type":"status_reply","id":"…","daemon":"…","protocol":"blemees/3","uptime_s":…,
   "agents":{…},"profiles":[{"name":"…","running":true,"sessions":2}],
   "sessions":{"total":5,"attached":4,"needs_attention":1,"by_profile":{…}},
   "config":{"state_dir":"…","ring_buffer_size":1024,"profile_idle_s":900,…}}
```

### 9.7 Errors
`error{code,message,session_id?,id?}`. Codes carried over from `blemees/2`
plus the ACP-era changes:

| Code | Meaning | Fatal? |
|---|---|---|
| `protocol_mismatch` | Bad protocol version. | Yes |
| `invalid_message` / `unknown_message` | Malformed / unknown frame. | No |
| `profile_unknown` | No such profile. | No |
| `agent_unavailable` | Profile's agent binary missing on `$PATH`. | No |
| `spawn_failed` | Agent process failed to launch / `initialize` failed. | No |
| `session_unknown` / `session_exists` / `session_busy` | Session table errors. | No |
| `agent_crashed` | Agent stdio closed / non-zero exit mid-turn (was `backend_crashed`). | No |
| `auth_required` | Agent reports auth needed (ACP `auth_required` or login lapsed). | No |
| `auth_failed` | Authentication attempt failed. | No |
| `view_only` | Drive attempted on a session whose agent lacks `loadSession` after restart. | No |
| `oversize_message` / `slow_consumer` / `daemon_shutdown` | Connection-fatal. | Yes |
| `internal` | Unexpected. | No |

---

## 10. Security

Unchanged posture from `blemees/2 §7`: `0600` socket, no auth beyond socket
perms, no remote listener (SSH-forward the socket), per-user daemon,
`SO_PEERCRED` logged. Additions:
- Profile `agent.env` may carry credentials/tokens — never logged at INFO+,
  redacted at DEBUG, never written to the registry in plaintext beyond what
  the user put in the config file.
- The notify webhook URL and payloads are treated as secret-adjacent;
  `detail` text is the only conversation-derived field and is capped/scrubbed
  (no prompt bodies).

---

## 11. TUI upgrade (`blemees-tui`)

The TUI remains the only northbound client and shares the ACP SDK's Pydantic
models for typed rendering. Touch points:

- `connection.py` — rewrite from the `blemees/2` client to `blemees/3`
  (profiles, attach/owner-viewer, `session.*`, permission relay).
- `reducer.py` / `state.py` — consume ACP `session/update` variants
  (`agent_message_chunk`, `agent_thought_chunk`, `tool_call(+update)`,
  `plan`, `available_commands_update`, `current_mode_update`) instead of
  `agent.*`.
- `widgets/modals/new_session.py` — becomes **profile-aware**: pick an
  existing profile (or create/edit one) instead of choosing
  `backend`+`options`.
- `widgets/modals/attach.py` — owner vs viewer, with takeover; show role and
  `needs_attention` badges.
- `widgets/chat_pane.py` — render ACP content blocks + tool-call cards
  (status pending→in_progress→completed/failed) + the **inline permission
  card** (see below).
- `widgets/todo_panel.py` — already a TODO panel; bind to ACP `plan`
  entries.
- `widgets/completion.py` — slash commands from ACP `available_commands`.
- `discover.py` — list from the daemon registry, not disk.

### 11.1 Permission UX (net-new)

**Inline card + attention-queue badge.** A relayed
`session.request_permission` renders as an inline card in that session's
transcript: tool title/kind/target, and four buttons mapped to the ACP
`PermissionOption.kind`s (`allow_once`, `allow_always`, `reject_once`,
`reject_always`). Answering posts `session.permission_response`.
Background/detached sessions in `needs_attention` show a count badge in the
sidebar; selecting one focuses the session and scrolls to its pending card.
This composes with the daemon's `needs_attention` model and never hijacks
focus, so it scales to many concurrent sessions and to reattaching to
headless work.

---

## 12. Versioning

`blemees/3`. Clean break: the daemon supports a single protocol version;
no `blemees/2` shim. The `agent.*` vocabulary, `options.<backend>.*`,
`agent.open`/`agent.user`/`agent.result`, and the two native translators are
removed. Daemon stays `0.x` (breaking changes allowed pre-1.0).
