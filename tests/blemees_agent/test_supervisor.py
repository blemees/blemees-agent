"""Unit tests for the profile registry / supervisor (#17)."""

from __future__ import annotations

import sys

import pytest

from blemees_agent.config import Config
from blemees_agent.errors import ProfileUnknownError
from blemees_agent.logging import configure
from blemees_agent.supervisor import DEFAULT_PROFILE, Supervisor


def _config(**profiles) -> Config:
    return Config(
        socket_path="/tmp/x.sock",
        agent_command=sys.executable,
        agent_args=["-c", "pass"],
        profiles=profiles,
    )


def _sup(**profiles) -> Supervisor:
    return Supervisor(_config(**profiles), configure("error"))


def test_default_profile_synthesised_from_agent_command():
    sup = _sup()
    p = sup.get_profile(None)
    assert p.name == DEFAULT_PROFILE
    assert p.command == sys.executable
    assert p.args == ["-c", "pass"]


def test_named_profiles_loaded_from_config():
    sup = _sup(codex={"agent_command": "codex-acp", "model": "gpt-5.5", "args": ["acp"]})
    p = sup.get_profile("codex")
    assert p.command == "codex-acp"
    assert p.model == "gpt-5.5"
    assert p.args == ["acp"]
    # default still present alongside named profiles
    assert DEFAULT_PROFILE in sup.profile_names()


def test_unknown_profile_raises():
    with pytest.raises(ProfileUnknownError):
        _sup().get_profile("nope")


def test_profile_list_reports_not_running_initially():
    sup = _sup(codex={"agent_command": "codex-acp"})
    rows = {r["name"]: r for r in sup.profile_list()}
    assert rows[DEFAULT_PROFILE]["running"] is False
    assert rows[DEFAULT_PROFILE]["sessions"] == 0
    assert rows["codex"]["agent"] == "codex-acp"


def test_mcp_servers_and_env_carried():
    sup = _sup(
        p={
            "agent_command": "x",
            "mcp_servers": [{"name": "fs", "command": "mcp-fs"}],
            "env": {"A": "1"},
        }
    )
    prof = sup.get_profile("p")
    assert prof.mcp_servers == [{"name": "fs", "command": "mcp-fs"}]
    assert prof.env == {"A": "1"}
