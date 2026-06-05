"""Profiles, agents, and the per-agent process supervisor (blemees/3, #17).

The model is **Profile → Agent → Session**:

* A **Profile** is a named container of one or more **Agents** (plus, later,
  profile-level permission policy / notify config — #20 / #24).
* An **Agent** is an independently-configured ACP agent (its own binary, CLI
  args, model/mode, cwd, MCP servers, env). Two agents in a profile may even
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

import dataclasses
import os
import time
from typing import Any

from .backends.acp import AcpAgentProcess, AcpSessionHandle
from .config import Config
from .errors import ProfileUnknownError

DEFAULT_PROFILE = "default"
DEFAULT_AGENT = "default"


@dataclasses.dataclass(slots=True)
class Agent:
    """One independently-configured ACP agent (the process-supervision unit)."""

    name: str
    command: str
    args: list[str] = dataclasses.field(default_factory=list)
    model: str | None = None
    mode: str | None = None
    cwd: str | None = None
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
        cwd=spec.get("cwd"),
        mcp_servers=list(spec.get("mcp_servers") or []),
        env=dict(spec.get("env") or {}),
    )


def _profiles_from_config(config: Config) -> dict[str, Profile]:
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
        agents_spec = pspec.get("agents")
        if isinstance(agents_spec, dict) and agents_spec:
            agents = {
                aname: _agent_from_spec(aname, aspec, config.agent_command)
                for aname, aspec in agents_spec.items()
                if isinstance(aspec, dict)
            }
        else:
            # Flat sugar: the profile body defines a single "default" agent.
            agents = {DEFAULT_AGENT: _agent_from_spec(DEFAULT_AGENT, pspec, config.agent_command)}
        if agents:
            policy = pspec.get("permission_policy")
            notify = pspec.get("notify")
            profiles[pname] = Profile(
                name=pname,
                agents=agents,
                permission_policy=(
                    dict(policy)
                    if isinstance(policy, dict)
                    else {"mode": "relay", "detached": "stall"}
                ),
                notify=dict(notify) if isinstance(notify, dict) else {},
            )
    return profiles


class Supervisor:
    """Owns the profile/agent registry and one ACP process per agent."""

    def __init__(self, config: Config, logger: Any) -> None:
        self._config = config
        self._log = logger
        self._profiles: dict[str, Profile] = _profiles_from_config(config)
        # (profile, agent) → process
        self._processes: dict[tuple[str, str], AcpAgentProcess] = {}
        # (profile, agent) → monotonic time its last session closed (idle clock)
        self._idle_since: dict[tuple[str, str], float] = {}

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
            cwd=cwd or agent.cwd,
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
            rows.append({"name": pname, "agents": agents})
        return rows

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
