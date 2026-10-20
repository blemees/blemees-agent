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
)
from .notify import BLOCKED_TRIGGERS, KNOWN_TRIGGERS
from .protocol import is_valid_name

if TYPE_CHECKING:
    from .registry import Registry

DEFAULT_PROFILE = "default"
DEFAULT_AGENT = "default"

# Placeholder that ``profile.get`` puts in place of a secret value (#43): agent
# ``env`` values, MCP-server ``env`` values, and every ``notify`` value except
# ``format`` (the webhook URL is secret-adjacent, spec §10). Sent back in a
# ``profile.update`` (or ``create``), it means "keep the stored value" — see
# :func:`_restore_redacted`.
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


def _mcp_key(spec: dict[str, Any]) -> Any:
    # Same identity the ACP backend uses for an MCP server (name, else command).
    return spec.get("name", spec.get("command"))


def _restore_map(new: Any, old: dict[str, Any]) -> None:
    """In place: swap each ``REDACTED`` value in ``new`` for the stored one
    under the same key; drop the key when nothing is stored to restore."""
    if not isinstance(new, dict):
        return
    for key, value in list(new.items()):
        if value == REDACTED:
            if key in old:
                new[key] = old[key]
            else:
                del new[key]


def _restore_redacted(spec: dict[str, Any], old: Profile | None) -> dict[str, Any]:
    """Return a copy of a raw profile spec with ``REDACTED`` placeholders
    replaced by the stored secret values of ``old`` (#43).

    ``profile.get`` never returns secrets, but ``profile.update`` replaces the
    whole spec — so an editor that round-trips ``profile.info`` sends the
    placeholders back. They resolve to the existing value of the same key
    (agent ``env``, MCP-server ``env`` matched by server name, ``notify``); a
    placeholder with no stored value (new key, renamed agent, or ``create``)
    is dropped rather than stored literally.
    """
    spec = copy.deepcopy(spec)
    _restore_map(spec.get("notify"), old.notify if old is not None else {})
    for aname, aspec in _agent_specs(spec):
        old_agent = old.agents.get(aname) if old is not None else None
        _restore_map(aspec.get("env"), old_agent.env if old_agent is not None else {})
        mcp_servers = aspec.get("mcp_servers")
        if not isinstance(mcp_servers, list):
            continue
        old_mcp = {
            _mcp_key(m): m
            for m in (old_agent.mcp_servers if old_agent is not None else [])
            if isinstance(m, dict)
        }
        for server in mcp_servers:
            if isinstance(server, dict):
                old_env = old_mcp.get(_mcp_key(server), {}).get("env")
                _restore_map(server.get("env"), old_env if isinstance(old_env, dict) else {})
    return spec


def _redacted_mcp_server(spec: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    if isinstance(out.get("env"), dict):
        out["env"] = {k: REDACTED for k in out["env"]}
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

        Secrets are never returned (spec §10): agent ``env`` and MCP-server
        ``env`` values and all ``notify`` values except ``format`` are replaced
        by :data:`REDACTED`; keys are kept so the editor can show them.
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
