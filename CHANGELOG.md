# Changelog

All notable changes to `blemees-agent` are recorded here. The project follows
[semantic versioning](https://semver.org) (pre-1.0: minor versions may carry
breaking wire changes, signalled by the protocol identifier).

## 0.11.0

**The ACP migration — `blemees/3`.** The daemon is rearchitected from a bespoke
per-backend translator (Claude Code / Codex, `blemees/2`, the `agent.*`
vocabulary) into an **ACP supervisor + semantic proxy**: it is now a client of
the [Agent Client Protocol](https://agentclientprotocol.com), spawning ACP
agent subprocesses and multiplexing sessions onto them. Clean break — there are
no `blemees/2` compatibility shims.

### Added
- **ACP client backend + supervisor** — one supervised ACP process per agent;
  ACP-native session multiplexing. Model is **Profile → Agent → Session**: a
  profile holds one or more independently-configured agents; sessions multiplex
  inside an agent's process. (#15, #16, #17)
- **Owner/viewer attach** — `session.attach`/`session.detach` with takeover and
  read-only viewers; reconnect + replay via `last_seen_seq`. (#19)
- **Permission relay** — per-profile policy (`relay`/`allow`/`deny` +
  `detached` behaviour); relayed `session/request_permission` to the owner,
  stall when detached. (#20)
- **Durability** — daemon-owned persistent session registry and an always-on
  per-session durable event log under the state dir; survive restart. (#21, #22)
- **Conversational resume** — on restart, rehydrate a session via ACP
  `session/load`; agents lacking `loadSession` degrade to `view_only`. (#23)
- **Notify service** — per-session `needs_attention` state with a pluggable
  outbound **webhook** sink (per-profile + global fallback); triggers for
  detached permission stall, `auth_required`, and agent crash; outstanding set
  surfaced in `status`; `notify.test` verb. (#24)
- **Over-wire profile CRUD** — `profile.create`/`update`/`delete`, persisted to
  the registry and coexisting with config-file profiles; `agent_unavailable`
  for a missing binary at create/start time. (#25)
- **Wire-protocol JSON Schemas** — re-authored Draft 2020-12 schemas for the
  full `blemees/3` frame set, shipped in the wheel under
  `blemees_agent/schemas/{inbound,outbound}/` and the machine-readable
  contract again. (#30)
- **e2e matrix + latency bench** — opt-in (`BLEMEES_E2E=1`) end-to-end tests
  per ACP agent (`requires_claude_acp`/`requires_codex_acp`/`requires_gemini_acp`,
  skipped unless installed + authenticated) covering turn, multi-turn memory,
  attach/viewer, and interrupt; `python -m blemees_agent.bench` measures warm
  prompt → first `agent_message_chunk`. (#26)

### Changed
- Wire protocol is now `blemees/3`; the data plane carries verbatim ACP
  payloads (`session/update` content blocks). The console client
  `blemees-agentctl` and its `open` verb are profile-aware.
- First (and only) runtime dependency: `agent-client-protocol` (the stdlib-only
  stance is retired).

### Removed
- The `blemees/2` `agent.*` event vocabulary and the
  `translate_claude.py` / `translate_codex.py` translators.
