"""Profiles, agents, and the per-agent process supervisor (blemees/3, #17).

The model is **Profile → Agent → Session**:

* A **Profile** is a named container of one or more **Agents** (plus, later,
  profile-level permission policy / notify config — #20 / #24).
* An **Agent** is an independently-configured ACP agent (its own binary, CLI
  args, model/mode, agent_home, MCP servers, env). Two agents in a profile may even
  be the same vendor with different configs. Each agent is the unit of
  process supervision: the daemon runs at most one
  :class:`~blemees_agent.backends.acp.AcpAgentProcess` per agent, lazily
  started on first ``session.open`` and idle-reaped when its last session
  closes.
* A **Session** is an ACP session multiplexed inside an agent's process.

Profiles come from the config file's ``[profiles.<p>.agents.<a>]`` tables; a
profile written flat (fields directly under ``[profiles.<p>]``, no ``agents``
table) is sugar for a single agent named ``default``. A built-in ``default``
profile with a ``default`` agent is always synthesised from the daemon's
``agent_command`` / ``agent_args`` so ``session.open`` works with no config.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import time
from typing import TYPE_CHECKING, Any

from .backends.acp import AcpAgentProcess, AcpSessionHandle
from .config import Config
from .errors import (
    AgentUnavailableError,
    ProfileExistsError,
    ProfileProtectedError,
    ProfileUnknownError,
    ProtocolError,
)
from .notify import BLOCKED_TRIGGERS, KNOWN_TRIGGERS
from .protocol import is_valid_name

if TYPE_CHECKING:
    from .registry import Registry

DEFAULT_PROFILE = "default"
DEFAULT_AGENT = "default"

# Placeholder that ``profile.get`` puts in place of a secret value (#43):
# agent ``env`` values; MCP-server ``env`` and ``headers`` values (in list
# form every entry key except ``name``); every ``notify`` value except
# ``format`` (a deny-list, so a new notify knob is never leaked; the webhook
# URL is secret-adjacent, spec §10). Everything else (``args``, ``url``, ...)
# is returned as-is. Sent back in a ``profile.update`` it means "keep the
# stored value" (or refuse if that can't be matched for certain); in
# ``create`` it is dropped — see :func:`_restore_redacted`. So the literal
# string can't be stored.
REDACTED = "<redacted>"


@dataclasses.dataclass(slots=True)
class Agent:
    """One independently-configured ACP agent (the process-supervision unit)."""

    name: str
    command: str
    args: list[str] = dataclasses.field(default_factory=list)
    model: str | None = None
    mode: str | None = None
    # The agent process's home directory — where the ACP subprocess is
    # spawned. Distinct from a session's ``cwd`` (the per-session working
    # directory, supplied on ``session.open``): one agent_home can serve
    # sessions in many different cwds.
    agent_home: str | None = None
    mcp_servers: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    env: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class Profile:
    """A named container of agents (plus cross-agent policy)."""

    name: str
    agents: dict[str, Agent] = dataclasses.field(default_factory=dict)
    # Permission policy applied to this profile's sessions (#20):
    # {"mode": relay|allow|deny, "detached": stall|allow|deny}.
    permission_policy: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"mode": "relay", "detached": "stall"}
    )
    # Notify config for this profile's sessions (#24, §6). Currently
    # ``{"webhook_url": str}``; absent falls back to the daemon-global URL.
    notify: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Attention policy (#51): which triggers arm needs_attention + webhook
    # for this profile's sessions. Default = the blocked set; turn_complete
    # is opt-in (``[profiles.<p>.attention] triggers = [...]``).
    attention_triggers: set[str] = dataclasses.field(default_factory=lambda: set(BLOCKED_TRIGGERS))

    @property
    def default_agent(self) -> Agent:
        """The agent used when ``session.open`` names a profile but no agent."""
        if DEFAULT_AGENT in self.agents:
            return self.agents[DEFAULT_AGENT]
        # else the sole / first agent
        return next(iter(self.agents.values()))


def _agent_from_spec(name: str, spec: dict[str, Any], fallback_command: str) -> Agent:
    return Agent(
        name=name,
        command=spec.get("agent_command") or spec.get("command") or fallback_command,
        args=list(spec.get("agent_args") or spec.get("args") or []),
        model=spec.get("model"),
        mode=spec.get("mode"),
        agent_home=spec.get("agent_home"),
        mcp_servers=list(spec.get("mcp_servers") or []),
        env=dict(spec.get("env") or {}),
    )


def _profile_from_spec(
    name: str, pspec: dict[str, Any], fallback_command: str, log: Any | None = None
) -> Profile | None:
    """Build one :class:`Profile` from a raw spec dict (config table or the
    over-wire ``profile`` object, #25).

    Agent shape, most specific first: an ``agents`` table (multi-agent), a
    single ``agent`` object (becomes the ``default`` agent), or flat sugar
    where the profile body itself is the single ``default`` agent's spec.
    """
    agents_spec = pspec.get("agents")
    if isinstance(agents_spec, dict) and agents_spec:
        # Agent names share the profile-name charset (#54); the wire path is
        # rejected at parse time, so this filter only bites config tables —
        # warn per skipped key so typos are diagnosable.
        agents = {}
        for aname, aspec in agents_spec.items():
            if not isinstance(aspec, dict):
                continue
            if not is_valid_name(str(aname)):
                if log is not None:
                    log.warning(
                        "profile.invalid_agent_name_skipped", profile=name, agent=str(aname)
                    )
                continue
            agents[aname] = _agent_from_spec(aname, aspec, fallback_command)
    elif isinstance(pspec.get("agent"), dict):
        agents = {DEFAULT_AGENT: _agent_from_spec(DEFAULT_AGENT, pspec["agent"], fallback_command)}
    else:
        agents = {DEFAULT_AGENT: _agent_from_spec(DEFAULT_AGENT, pspec, fallback_command)}
    if not agents:
        return None
    policy = pspec.get("permission_policy")
    notify = pspec.get("notify")
    return Profile(
        name=name,
        agents=agents,
        permission_policy=(
            dict(policy) if isinstance(policy, dict) else {"mode": "relay", "detached": "stall"}
        ),
        notify=dict(notify) if isinstance(notify, dict) else {},
        attention_triggers=_attention_triggers_from_spec(name, pspec, log),
    )


def _agent_specs(pspec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The (agent name, raw agent spec) pairs of a raw profile spec, using the
    same shape precedence as :func:`_profile_from_spec`."""
    agents_spec = pspec.get("agents")
    if isinstance(agents_spec, dict) and agents_spec:
        return [(str(a), s) for a, s in agents_spec.items() if isinstance(s, dict)]
    if isinstance(pspec.get("agent"), dict):
        return [(DEFAULT_AGENT, pspec["agent"])]
    return [(DEFAULT_AGENT, pspec)]


def _hashable(key: Any) -> Any:
    # Keys come from client JSON; a list/dict "name" must not crash matching.
    return key if key is None or isinstance(key, str) else repr(key)


def _is_pair_list(value: Any) -> bool:
    # List form of env / headers: [{"name": ..., "value": ...}, ...].
    return isinstance(value, list) and all(isinstance(e, dict) for e in value)


def _redact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    # Every key except ``name`` may carry the secret (``value``, ``val``, …).
    return {k: (v if k == "name" else REDACTED) for k, v in entry.items()}


def _redact_secret_field(value: Any) -> Any:
    """Redacted view of a secret-bearing field (env / headers, #43).

    Dict form keeps its keys; list form keeps each entry's ``name`` and
    redacts every other key; any other (unrecognised) shape is replaced
    whole, so an unexpected shape fails safe instead of leaking."""
    if isinstance(value, dict):
        return {k: REDACTED for k in value}
    if _is_pair_list(value):
        return [_redact_entry(e) for e in value]
    if value is None:
        return None
    return REDACTED


def _entry_has_placeholder(entry: dict[str, Any]) -> bool:
    return any(k != "name" and v == REDACTED for k, v in entry.items())


def _placeholder_path(value: Any) -> str | None:
    """Path suffix of the first placeholder in a secret field, else None."""
    if value == REDACTED:
        return ""
    if isinstance(value, dict):
        for k, v in value.items():
            if v == REDACTED:
                return f".{k}"
    elif _is_pair_list(value):
        for j, entry in enumerate(value):
            if _entry_has_placeholder(entry):
                return f"[{j}]"
    return None


def _unresolvable(path: str) -> ProtocolError:
    return ProtocolError(f"cannot resolve {REDACTED} for {path}; re-enter the value")


def _drop_placeholders(container: dict[str, Any], field: str) -> None:
    """``create`` (nothing stored): every placeholder is dropped (#43)."""
    value = container.get(field)
    if value == REDACTED:
        del container[field]
    elif isinstance(value, dict):
        container[field] = {k: v for k, v in value.items() if v != REDACTED}
    elif _is_pair_list(value):
        container[field] = [e for e in value if not _entry_has_placeholder(e)]


def _unique_by(items: list[dict[str, Any]], key: Any) -> dict[Any, dict[str, Any]]:
    """``key(item)`` → item, for keys that occur exactly once (None skipped)."""
    seen: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        k = key(item)
        if k is not None:
            seen.setdefault(k, []).append(item)
    return {k: group[0] for k, group in seen.items() if len(group) == 1}


def _entry_name(entry: dict[str, Any]) -> Any:
    return _hashable(entry["name"]) if entry.get("name") is not None else None


def _restore_field(container: dict[str, Any], field: str, old: Any, path: str) -> None:
    """Resolve ``REDACTED`` placeholders in one secret-bearing field against
    its stored value (``update``, #43).

    Dict form is keyed: a placeholder keeps the stored value of that key, or
    is dropped if nothing is stored. List form matches an entry by ``name``
    only when that name is unique in both the stored and the new list, and
    raises (``invalid_message``) for any other entry with a placeholder. A
    whole-field placeholder keeps the stored field."""
    if field not in container:
        return
    new = container[field]
    if new == REDACTED:
        if old is None:
            del container[field]
        else:
            container[field] = copy.deepcopy(old)
        return
    if isinstance(new, dict):
        stored = old if isinstance(old, dict) else {}
        container[field] = {
            k: (stored[k] if v == REDACTED else v)
            for k, v in new.items()
            if v != REDACTED or k in stored
        }
        return
    if not _is_pair_list(new):
        return
    old_by_name = _unique_by(old if _is_pair_list(old) else [], _entry_name)
    new_by_name = _unique_by(new, _entry_name)
    out: list[dict[str, Any]] = []
    for j, entry in enumerate(new):
        if not _entry_has_placeholder(entry):
            out.append(entry)
            continue
        name = _entry_name(entry)
        if name is None or name not in new_by_name or name not in old_by_name:
            raise _unresolvable(f"{path}.{field}[{j}]")
        stored_entry = old_by_name[name]
        out.append(
            {
                k: (stored_entry[k] if k != "name" and v == REDACTED else v)
                for k, v in entry.items()
                if k == "name" or v != REDACTED or k in stored_entry
            }
        )
    container[field] = out


# Secret-bearing fields of an MCP server entry (#43): redacted by profile.get.
_MCP_SECRET_FIELDS = ("env", "headers")


def _mcp_identity(server: dict[str, Any]) -> str:
    # A server's non-secret fields (name/command/args/url/type/…), compared exactly.
    rest = {k: v for k, v in server.items() if k not in _MCP_SECRET_FIELDS}
    return json.dumps(rest, sort_keys=True, default=repr)


def _restore_mcp_servers(agent_path: str, new_servers: list[Any], old_servers: list[Any]) -> None:
    """Resolve placeholders in an agent's ``mcp_servers`` (``update``, #43).

    Never guesses. A server with a placeholder is matched to a stored server
    only by (1) exact identity — the same non-secret fields — when that
    identity occurs exactly once in both lists, else (2) its ``name`` when
    that name occurs exactly once in both lists. Unnamed servers have no
    fallback (the same command with new args may be a different server), and
    occurrence order is never used. Anything else raises ``invalid_message``
    naming the server, so a secret can't land on the wrong server. Servers
    with no placeholder need no match."""
    old = [m for m in old_servers if isinstance(m, dict)]
    new = [(i, m) for i, m in enumerate(new_servers) if isinstance(m, dict)]
    new_dicts = [m for _, m in new]
    old_by_id = _unique_by(old, _mcp_identity)
    new_by_id = _unique_by(new_dicts, _mcp_identity)
    old_by_name = _unique_by(old, _entry_name)
    new_by_name = _unique_by(new_dicts, _entry_name)
    for i, server in new:
        path = f"{agent_path}.mcp_servers[{i}]"
        pending = [
            (field, suffix)
            for field in _MCP_SECRET_FIELDS
            if field in server and (suffix := _placeholder_path(server[field])) is not None
        ]
        if not pending:
            continue
        ident, name = _mcp_identity(server), _entry_name(server)
        if ident in new_by_id and ident in old_by_id:
            match = old_by_id[ident]
        elif name is not None and name in new_by_name and name in old_by_name:
            match = old_by_name[name]
        else:
            field, suffix = pending[0]
            raise _unresolvable(f"{path}.{field}{suffix}")
        for field in _MCP_SECRET_FIELDS:
            _restore_field(server, field, match.get(field), path)


def _restore_redacted(spec: dict[str, Any], old: Profile | None) -> dict[str, Any]:
    """Return a copy of a raw profile spec with ``REDACTED`` placeholders
    resolved against the stored profile ``old`` (#43).

    ``profile.get`` never returns these secrets, but ``profile.update``
    replaces the whole spec — so an editor that round-trips ``profile.info``
    sends the placeholders back. On update they resolve to the stored value
    (agent ``env`` and ``notify`` by key, agents by name, MCP servers per
    :func:`_restore_mcp_servers`); a placeholder that can't be resolved for
    certain raises ``ProtocolError`` (``invalid_message``) before anything is
    written. On ``create`` (``old is None``) nothing is stored, so every
    placeholder is dropped rather than stored literally.
    """
    spec = copy.deepcopy(spec)
    if old is None:
        _drop_placeholders(spec, "notify")
        for _aname, aspec in _agent_specs(spec):
            _drop_placeholders(aspec, "env")
            if isinstance(aspec.get("mcp_servers"), list):
                for server in aspec["mcp_servers"]:
                    if isinstance(server, dict):
                        for field in _MCP_SECRET_FIELDS:
                            _drop_placeholders(server, field)
        return spec
    if isinstance(spec.get("notify"), dict):
        _restore_field(spec, "notify", old.notify, "notify")
    for aname, aspec in _agent_specs(spec):
        old_agent = old.agents.get(aname)
        agent_path = f"agents.{aname}"
        if isinstance(aspec.get("env"), dict):
            _restore_field(aspec, "env", old_agent.env if old_agent else None, agent_path)
        if isinstance(aspec.get("mcp_servers"), list):
            _restore_mcp_servers(
                agent_path, aspec["mcp_servers"], old_agent.mcp_servers if old_agent else []
            )
    return spec


def _redacted_mcp_server(spec: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    for field in _MCP_SECRET_FIELDS:
        if field in out:
            out[field] = _redact_secret_field(out[field])
    return out


def _attention_triggers_from_spec(
    name: str, pspec: dict[str, Any], log: Any | None = None
) -> set[str]:
    """Resolve the profile's attention policy (#51). Absent → blocked set;
    a configured ``attention.triggers`` list replaces it, with unknown
    trigger names warn-skipped so a typo can't silently disarm the rest."""
    attention = pspec.get("attention")
    triggers = attention.get("triggers") if isinstance(attention, dict) else None
    if not isinstance(triggers, list):
        return set(BLOCKED_TRIGGERS)
    armed: set[str] = set()
    for t in triggers:
        if isinstance(t, str) and t in KNOWN_TRIGGERS:
            armed.add(t)
        elif log is not None:
            log.warning("profile.unknown_attention_trigger", profile=name, trigger=str(t))
    return armed


def _profiles_from_config(config: Config, log: Any | None = None) -> dict[str, Profile]:
    """Build the profile registry: a synthesised ``default`` + config-file ones."""
    profiles: dict[str, Profile] = {
        DEFAULT_PROFILE: Profile(
            name=DEFAULT_PROFILE,
            agents={
                DEFAULT_AGENT: Agent(
                    name=DEFAULT_AGENT,
                    command=config.agent_command,
                    args=list(config.agent_args),
                )
            },
        )
    }
    for pname, pspec in (config.profiles or {}).items():
        if not isinstance(pspec, dict):
            continue
        if not is_valid_name(str(pname)):
            # Warn-and-skip so one bad table name can't keep the daemon down (#54).
            if log is not None:
                log.warning("profile.invalid_name_skipped", profile=str(pname))
            continue
        profile = _profile_from_spec(pname, pspec, config.agent_command, log)
        if profile is not None:
            profiles[pname] = profile
        elif log is not None:
            # All of the table's agents were invalid/skipped — say so rather
            # than letting the profile vanish silently.
            log.warning("profile.no_valid_agents_skipped", profile=str(pname))
    return profiles


class Supervisor:
    """Owns the profile/agent registry and one ACP process per agent."""

    def __init__(self, config: Config, logger: Any, registry: Registry | None = None) -> None:
        self._config = config
        self._log = logger
        self._registry = registry
        self._profiles: dict[str, Profile] = _profiles_from_config(config, logger)
        # Names that come from config (synthesised default + [profiles.*]); these
        # are config-managed and can't be mutated/deleted over the wire (#25).
        self._static_names: set[str] = set(self._profiles)
        # (profile, agent) → process
        self._processes: dict[tuple[str, str], AcpAgentProcess] = {}
        # (profile, agent) → monotonic time its last session closed (idle clock)
        self._idle_since: dict[tuple[str, str], float] = {}

    def load_persisted(self) -> None:
        """Adopt over-wire profiles persisted in the registry (#25). Call after
        ``registry.load()``. A persisted name colliding with a config-managed
        one is skipped — config wins."""
        if self._registry is None:
            return
        for spec in self._registry.all_profiles():
            name = spec.get("name")
            if not isinstance(name, str) or name in self._static_names:
                if name in self._static_names:
                    self._log.warning("profile.persisted_shadowed_by_config", profile=name)
                continue
            profile = _profile_from_spec(name, spec, self._config.agent_command, self._log)
            if profile is not None:
                self._profiles[name] = profile

    # -- registry -------------------------------------------------------

    def get_profile(self, name: str | None) -> Profile:
        try:
            return self._profiles[name or DEFAULT_PROFILE]
        except KeyError:
            raise ProfileUnknownError(name or DEFAULT_PROFILE) from None

    def resolve(self, profile_name: str | None, agent_name: str | None) -> tuple[Profile, Agent]:
        profile = self.get_profile(profile_name)
        if agent_name is None:
            return profile, profile.default_agent
        agent = profile.agents.get(agent_name)
        if agent is None:
            raise ProfileUnknownError(f"{profile.name}/{agent_name}")
        return profile, agent

    def profile_names(self) -> list[str]:
        return list(self._profiles)

    def webhook_url_for(self, profile_name: str | None) -> str | None:
        """The notify webhook URL for a profile: its own, else the global
        fallback from config (#24, §6). Unknown profiles get the fallback."""
        profile = self._profiles.get(profile_name or DEFAULT_PROFILE)
        if profile is not None:
            url = profile.notify.get("webhook_url")
            if url:
                return url
        return self._config.notify_webhook_url

    def webhook_format_for(self, profile_name: str | None) -> str:
        """The webhook payload format for a profile (#52): its own
        ``notify.format``, else the global default. Unknown values fall back
        to "json" with a warning so a typo degrades gracefully."""
        fmt = None
        profile = self._profiles.get(profile_name or DEFAULT_PROFILE)
        if profile is not None:
            fmt = profile.notify.get("format")
        fmt = fmt or self._config.notify_webhook_format or "json"
        if fmt not in ("json", "ntfy"):
            self._log.warning("notify.unknown_format", profile=profile_name, format=str(fmt))
            return "json"
        return fmt

    # -- processes ------------------------------------------------------

    def _process_for(self, profile: Profile, agent: Agent) -> AcpAgentProcess:
        key = (profile.name, agent.name)
        proc = self._processes.get(key)
        if proc is None:
            env = {**os.environ, **agent.env}
            proc = AcpAgentProcess(agent, key=key, logger=self._log, env=env)
            self._processes[key] = proc
        return proc

    def make_handle(
        self,
        profile_name: str | None,
        agent_name: str | None,
        *,
        on_event: Any,
        cwd: str | None,
        permission_cb: Any = None,
        resume_native_id: str | None = None,
    ) -> AcpSessionHandle:
        """Return a per-session handle bound to the (profile, agent)'s lazy process."""
        profile, agent = self.resolve(profile_name, agent_name)
        proc = self._process_for(profile, agent)
        self._idle_since.pop((profile.name, agent.name), None)  # acquiring → not idle
        return AcpSessionHandle(
            process=proc,
            on_event=on_event,
            cwd=cwd,
            on_close=self._on_session_close,
            permission_cb=permission_cb,
            resume_native_id=resume_native_id,
        )

    async def _on_session_close(self, proc: AcpAgentProcess) -> None:
        if proc.session_count() == 0:
            self._idle_since[proc.key_tuple] = time.monotonic()

    # -- explicit lifecycle (profile-wide: all of a profile's agents) ---

    async def start(self, name: str | None) -> list[AcpAgentProcess]:
        profile = self.get_profile(name)
        # Surface a missing binary as agent_unavailable rather than a spawn
        # crash partway through starting the profile's agents (#25).
        self._validate_agents_available(profile)
        started: list[AcpAgentProcess] = []
        for agent in profile.agents.values():
            proc = self._process_for(profile, agent)
            await proc.ensure_started()
            started.append(proc)
        return started

    async def stop(self, name: str | None) -> int:
        profile = self.get_profile(name)
        stopped = 0
        for agent in profile.agents.values():
            key = (profile.name, agent.name)
            proc = self._processes.pop(key, None)
            self._idle_since.pop(key, None)
            if proc is not None:
                await proc.close()
                stopped += 1
        return stopped

    # -- over-wire CRUD (#25) -------------------------------------------

    def _validate_agents_available(self, profile: Profile) -> None:
        """Raise :class:`AgentUnavailableError` if any agent binary is missing.

        ``shutil.which`` resolves both bare names on ``$PATH`` and explicit
        paths, so a typo'd profile fails cleanly here rather than as a spawn
        crash on first ``session.open``."""
        for agent in profile.agents.values():
            if shutil.which(agent.command) is None:
                raise AgentUnavailableError(agent.command, profile=profile.name)

    def _ensure_mutable(self, name: str) -> None:
        if name in self._static_names:
            raise ProfileProtectedError(name)

    def create_profile(self, name: str, spec: dict[str, Any]) -> Profile:
        if name in self._profiles:
            raise ProfileExistsError(name)
        # Nothing stored to restore: any REDACTED placeholder is dropped (#43).
        spec = _restore_redacted(spec, None)
        profile = _profile_from_spec(name, spec, self._config.agent_command, self._log)
        if profile is None:
            raise ProfileUnknownError(name)  # malformed spec → no agents
        self._validate_agents_available(profile)
        self._profiles[name] = profile
        self._persist_profile(name, spec)
        return profile

    async def update_profile(self, name: str, spec: dict[str, Any]) -> Profile:
        if name not in self._profiles:
            raise ProfileUnknownError(name)
        self._ensure_mutable(name)
        # A round-tripped profile.info carries REDACTED placeholders for
        # secrets; resolve them to the stored values before replacing (#43).
        spec = _restore_redacted(spec, self._profiles[name])
        profile = _profile_from_spec(name, spec, self._config.agent_command, self._log)
        if profile is None:
            raise ProfileUnknownError(name)
        self._validate_agents_available(profile)
        # Drop running processes so the next open respawns with the new config.
        await self.stop(name)
        self._profiles[name] = profile
        self._persist_profile(name, spec)
        return profile

    async def delete_profile(self, name: str) -> None:
        if name not in self._profiles:
            raise ProfileUnknownError(name)
        self._ensure_mutable(name)
        await self.stop(name)
        self._profiles.pop(name, None)
        if self._registry is not None:
            self._registry.remove_profile(name)
            self._registry.save()

    def _persist_profile(self, name: str, spec: dict[str, Any]) -> None:
        if self._registry is not None:
            self._registry.upsert_profile(name, spec)
            self._registry.save()

    def profile_list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pname, profile in self._profiles.items():
            agents = []
            for aname, agent in profile.agents.items():
                proc = self._processes.get((pname, aname))
                agents.append(
                    {
                        "name": aname,
                        "agent": agent.command,
                        "model": agent.model,
                        "running": bool(proc and proc.running),
                        "sessions": proc.session_count() if proc else 0,
                    }
                )
            rows.append(
                {
                    "name": pname,
                    "agents": agents,
                    # config-managed profiles can't be edited over the wire (#25).
                    "source": "config" if pname in self._static_names else "dynamic",
                }
            )
        return rows

    def is_config_managed(self, name: str) -> bool:
        return name in self._static_names

    def profile_spec(self, name: str | None) -> dict[str, Any]:
        """The full editable spec of one profile, for ``profile.get`` (#43).

        ``profile_list`` only carries per-agent name/command/model, and
        ``profile.update`` replaces the whole spec, so an editor seeded from
        it drops every other field. This is the complete spec (``agents``
        table with command/args/model/mode/agent_home/mcp_servers/env, plus
        ``permission_policy``, ``notify``, ``attention``), shaped so it can be
        sent back unchanged as a ``profile.update`` body. Works for config and
        dynamic profiles alike. Raises ``ProfileUnknownError`` for a bad name.

        Redacted (spec §10) to :data:`REDACTED`, keys kept so the editor can
        show them: agent ``env`` values; MCP-server ``env`` and ``headers``
        values (in list form every entry key except ``name``); every
        ``notify`` value except ``format``. Everything else, including ``args`` and ``url``, is
        returned as-is. Values are the resolved ones (e.g. the daemon's
        ``agent_command`` when the profile set none, default policy/triggers).
        """
        profile = self.get_profile(name)
        agents: dict[str, Any] = {}
        for aname, agent in profile.agents.items():
            aspec: dict[str, Any] = {"command": agent.command, "args": list(agent.args)}
            # Optional scalars are omitted when unset (not null) so the spec
            # stays valid as a profile.update body (AgentSpec types them string).
            if agent.model is not None:
                aspec["model"] = agent.model
            if agent.mode is not None:
                aspec["mode"] = agent.mode
            if agent.agent_home is not None:
                aspec["agent_home"] = agent.agent_home
            aspec["mcp_servers"] = [
                _redacted_mcp_server(m) if isinstance(m, dict) else m for m in agent.mcp_servers
            ]
            aspec["env"] = {k: REDACTED for k in agent.env}
            agents[aname] = aspec
        return {
            "name": profile.name,
            "agents": agents,
            "permission_policy": copy.deepcopy(profile.permission_policy),
            "notify": {k: (v if k == "format" else REDACTED) for k, v in profile.notify.items()},
            "attention": {"triggers": sorted(profile.attention_triggers)},
        }

    # -- reaping / shutdown --------------------------------------------

    async def reap_idle(self, idle_timeout_s: float, *, now: float | None = None) -> list[str]:
        if now is None:
            now = time.monotonic()
        reaped: list[str] = []
        for key, since in list(self._idle_since.items()):
            proc = self._processes.get(key)
            if proc is None:
                self._idle_since.pop(key, None)
                continue
            if proc.session_count() == 0 and now - since >= idle_timeout_s:
                await proc.close()
                self._processes.pop(key, None)
                self._idle_since.pop(key, None)
                reaped.append(f"{key[0]}/{key[1]}")
        return reaped

    async def close_all(self) -> None:
        for proc in list(self._processes.values()):
            await proc.close()
        self._processes.clear()
        self._idle_since.clear()
