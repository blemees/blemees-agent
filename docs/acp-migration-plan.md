---
title: ACP migration plan
nav_order: 4
permalink: /acp-plan/
---

# ACP migration — phased implementation plan

Companion to [`acp-migration-spec.md`](acp-migration-spec.md). Spans two
repos: **`blemees-agent`** (the daemon rewrite) and **`blemees-tui`** (the
client upgrade). Each phase ends in a state that runs and is testable.

Legend: 🟦 `blemees-agent` · 🟪 `blemees-tui` · ⬛ shared.

---

## Phase 0 — Spike: prove the ACP client path (🟦)

De-risk the core assumption before committing.

- Add `agent-client-protocol` (SDK) as a dependency; drop the stdlib-only
  constraint in `pyproject.toml`.
- Throwaway script: spawn `claude-agent-acp` over stdio via the SDK `Client`,
  `initialize` with **empty** client capabilities, `session/new`,
  `session/prompt "what is 2+2"`, stream `session/update`, observe
  `stopReason`. Repeat for `codex-acp` and `gemini --experimental-acp`.
- **Verify the load-bearing facts:** (a) agents work with no `fs`/`terminal`
  capability; (b) `loadSession` is advertised by claude-agent-acp / codex-acp;
  (c) permission requests arrive and can be auto-answered; (d) which agents
  honour `model` at `session/new`.

**Exit:** a transcript per agent + a short findings note. Kills or confirms
the design's assumptions. If an agent *requires* `fs` capability, revisit §4.

---

## Phase 1 — ACP client backend + session multiplexing (🟦)

The new southbound core, behind the existing daemon scaffolding.

- `backends/acp.py` — wrap the SDK `Client`: spawn/supervise one process,
  `initialize`, `session/new`/`load`, `prompt`, `cancel`, `set_mode`, and the
  `request_permission` handler hook. Demux `session/update` by `sessionId`.
- Delete `backends/translate_claude.py`, `backends/translate_codex.py`,
  `backends/claude.py`, `backends/codex.py` and their tests.
- Mock ACP agent stub for tests (`fake_acp.py`): scriptable `initialize` /
  `session/*` / `session/update` over stdio.

**Exit:** unit + mock tests green for one process hosting ≥3 multiplexed
sessions, full turn → `stopReason`, crash mid-turn → `agent_crashed`.

---

## Phase 2 — Profiles + supervisor (🟦)

- `profiles.py` — profile model, config-file loader, over-wire CRUD.
- Supervisor: one process per profile, lazy start, idle reap, restart on
  crash, `agent_unavailable` when the binary is missing.
- Wire `session.open` → choose profile → ensure process → `session/new` with
  the profile's `model`/`cwd`/`mcp_servers`.

**Exit:** `profile.create/list/start/stop`, two profiles running different
agents concurrently, MCP servers injected and reachable by the agent.

---

## Phase 3 — `blemees/3` wire protocol + owner/viewer (🟦)

- `protocol.py` — `blemees/3` frame codec (control + data planes, §9). New
  schemas under `schemas/`; delete the `agent.*` schemas.
- `daemon.py` — dispatch the new verbs; rebuild the owner/viewer model
  (attach/detach/takeover, viewer fan-out) on top of the session table.
- `session.request_permission` relay ↔ `session.permission_response`.

**Exit:** the `blemees-agentctl` REPL (updated for `blemees/3`) can open under
a profile, prompt, stream ACP updates, attach a second viewer, take over.

---

## Phase 4 — Durability: registry + replay + resume (🟦)

- Always-on state dir; `registry.json` (atomic) + per-session ACP event log.
- `seq`/ring-buffer/`replay_gap` repurposed to ACP frames.
- Daemon-restart path: reload registry → respawn profile processes →
  `session/load` where `loadSession`; else mark `view_only`.
- `session.list` / `session.info` served from the registry (no disk scan);
  drop usage fields.

**Exit:** kill+restart the daemon → sessions relist, viewers replay,
load-capable sessions continue, view-only sessions clearly flagged.

---

## Phase 5 — Notify service (🟦)

- `needs_attention` state machine + the three default triggers.
- Pluggable sink interface; ship the **webhook** sink (per-profile + global
  fallback). Expose the outstanding queue via `status` / `session.list`.
- `notify.test` verb to fire a test event.

**Exit:** detached permission/auth/crash fires a webhook; reattaching shows
the queue; turn-complete does *not* fire.

---

## Phase 6 — TUI upgrade (🟪)

Depends on Phases 3–5. Share the SDK Pydantic models.

1. `connection.py` → `blemees/3` (profiles, attach/owner-viewer, permission
   relay, reconnect+replay via `last_seen_seq`).
2. `reducer.py`/`state.py` → consume ACP `session/update` variants.
3. `chat_pane.py` → render content blocks + tool-call cards + **inline
   permission card** (the four `PermissionOption.kind` buttons).
4. `modals/new_session.py` → profile picker + create/edit.
5. `modals/attach.py` → owner/viewer + takeover + `needs_attention` badges.
6. `todo_panel.py` → ACP `plan`; `completion.py` → `available_commands`.
7. `discover.py` → registry-backed listing.

**Exit:** end-to-end — create a profile, open a session, drive a turn with
tool calls + a plan, answer a permission inline; kill the TUI mid-turn,
relaunch, reattach, replay; view a second session as a viewer.

---

## Phase 7 — Polish, docs, e2e (⬛)

- e2e marks per agent (`requires_claude_acp`, `requires_codex_acp`,
  `requires_gemini_acp`); latency bench (warm prompt → first
  `agent_message_chunk`).
- Update `README`/`spec.md` to point at the ACP design; refresh
  `blemees-agentctl` help; packaging/version bump.
- `dev-install.sh` already installs both editable — no change needed.

---

## Cross-cutting risks

| Risk | Mitigation |
|---|---|
| Agent requires `fs`/`terminal` capability after all | Phase 0 verifies; fallback = advertise + daemon implements a minimal fs/terminal server (scope spike). |
| `loadSession` absent/flaky for an agent | View-only degradation is designed in (§7.3); flagged in `session.info.view_only`. |
| No token usage breaks TUI affordances | Accepted; `session.info` drops usage; TUI hides context-window UI. |
| SDK churn (pydantic models track ACP releases) | Pin the SDK; bump deliberately; mock stub insulates unit tests. |
| Permission relay deadlock vs multiplexing | Per-session pending state, async; verified in Phase 3 with a stalled session beside a live one. |
