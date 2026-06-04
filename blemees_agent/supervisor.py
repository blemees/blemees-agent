"""Profiles and the per-profile process supervisor (blemees/3, #17).

A **profile** is a named config bundle (agent binary + args, default model /
mode / cwd, MCP servers) and the unit of process supervision: the daemon runs
at most one :class:`~blemees_agent.backends.acp.AcpAgentProcess` per profile,
lazily started on first ``session.open`` and idle-reaped when its last session
closes. Sessions opened under a profile are ACP sessions multiplexed inside
that one process.

Profiles come from the config file's ``[profiles.<name>]`` tables; a built-in
``default`` profile is always synthesised from the daemon's
``agent_command`` / ``agent_args`` so ``session.open`` works with no profile
configured at all.
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


@dataclasses.dataclass(slots=True)
class Profile:
    name: str
    command: str
    args: list[str] = dataclasses.field(default_factory=list)
    model: str | None = None
    mode: str | None = None
    cwd: str | None = None
    mcp_servers: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    env: dict[str, str] = dataclasses.field(default_factory=dict)


def _profiles_from_config(config: Config) -> dict[str, Profile]:
    """Build the profile registry: a synthesised ``default`` + config-file ones."""
    profiles: dict[str, Profile] = {
        DEFAULT_PROFILE: Profile(
            name=DEFAULT_PROFILE,
            command=config.agent_command,
            args=list(config.agent_args),
        )
    }
    for name, spec in (config.profiles or {}).items():
        if not isinstance(spec, dict):
            continue
        profiles[name] = Profile(
            name=name,
            command=spec.get("agent_command") or spec.get("command") or config.agent_command,
            args=list(spec.get("agent_args") or spec.get("args") or []),
            model=spec.get("model"),
            mode=spec.get("mode"),
            cwd=spec.get("cwd"),
            mcp_servers=list(spec.get("mcp_servers") or []),
            env=dict(spec.get("env") or {}),
        )
    return profiles


class Supervisor:
    """Owns the profile registry and one ACP agent process per profile."""

    def __init__(self, config: Config, logger: Any) -> None:
        self._config = config
        self._log = logger
        self._profiles: dict[str, Profile] = _profiles_from_config(config)
        self._processes: dict[str, AcpAgentProcess] = {}
        # profile name → monotonic time its last session closed (idle-reap clock)
        self._idle_since: dict[str, float] = {}

    # -- profiles -------------------------------------------------------

    def get_profile(self, name: str | None) -> Profile:
        key = name or DEFAULT_PROFILE
        try:
            return self._profiles[key]
        except KeyError:
            raise ProfileUnknownError(key) from None

    def _process_for(self, profile: Profile) -> AcpAgentProcess:
        proc = self._processes.get(profile.name)
        if proc is None:
            env = {**os.environ, **profile.env}
            proc = AcpAgentProcess(profile, logger=self._log, env=env)
            self._processes[profile.name] = proc
        return proc

    def make_handle(
        self, profile_name: str | None, *, on_event: Any, cwd: str | None
    ) -> AcpSessionHandle:
        """Return a per-session handle bound to the profile's (lazy) process."""
        profile = self.get_profile(profile_name)
        proc = self._process_for(profile)
        self._idle_since.pop(profile.name, None)  # acquiring → not idle
        return AcpSessionHandle(
            process=proc,
            on_event=on_event,
            cwd=cwd or profile.cwd,
            on_close=self._on_session_close,
        )

    async def _on_session_close(self, proc: AcpAgentProcess) -> None:
        if proc.session_count() == 0:
            self._idle_since[proc.profile.name] = time.monotonic()

    # -- explicit lifecycle --------------------------------------------

    async def start(self, name: str | None) -> AcpAgentProcess:
        proc = self._process_for(self.get_profile(name))
        await proc.ensure_started()
        return proc

    async def stop(self, name: str | None) -> bool:
        key = name or DEFAULT_PROFILE
        proc = self._processes.pop(key, None)
        self._idle_since.pop(key, None)
        if proc is None:
            return False
        await proc.close()
        return True

    def profile_names(self) -> list[str]:
        return list(self._profiles)

    def profile_list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, profile in self._profiles.items():
            proc = self._processes.get(name)
            rows.append(
                {
                    "name": name,
                    "agent": profile.command,
                    "model": profile.model,
                    "running": bool(proc and proc.running),
                    "sessions": proc.session_count() if proc else 0,
                }
            )
        return rows

    # -- reaping / shutdown --------------------------------------------

    async def reap_idle(self, idle_timeout_s: float, *, now: float | None = None) -> list[str]:
        if now is None:
            now = time.monotonic()
        reaped: list[str] = []
        for name, since in list(self._idle_since.items()):
            proc = self._processes.get(name)
            if proc is None:
                self._idle_since.pop(name, None)
                continue
            if proc.session_count() == 0 and now - since >= idle_timeout_s:
                await proc.close()
                self._processes.pop(name, None)
                self._idle_since.pop(name, None)
                reaped.append(name)
        return reaped

    async def close_all(self) -> None:
        for proc in list(self._processes.values()):
            await proc.close()
        self._processes.clear()
        self._idle_since.clear()
