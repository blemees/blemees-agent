"""Unit tests for the profile/agent registry / supervisor (#17).

Model: Profile -> Agent -> Session. A profile contains one or more agents;
each agent is an independently-configured ACP process.
"""

from __future__ import annotations

import sys

import pytest

from blemees_agent.config import Config
from blemees_agent.errors import ProfileUnknownError
from blemees_agent.logging import configure
from blemees_agent.supervisor import DEFAULT_AGENT, DEFAULT_PROFILE, Supervisor


def _config(**profiles) -> Config:
    return Config(
        socket_path="/tmp/x.sock",
        agent_command=sys.executable,
        agent_args=["-c", "pass"],
        profiles=profiles,
    )


def _sup(**profiles) -> Supervisor:
    return Supervisor(_config(**profiles), configure("error"))


def test_default_profile_and_agent_synthesised():
    sup = _sup()
    profile = sup.get_profile(None)
    assert profile.name == DEFAULT_PROFILE
    agent = profile.default_agent
    assert agent.name == DEFAULT_AGENT
    assert agent.command == sys.executable
    assert agent.args == ["-c", "pass"]


def test_flat_profile_is_single_default_agent():
    # A profile written flat (no `agents` table) is sugar for one default agent.
    sup = _sup(codex={"agent_command": "codex-acp", "model": "gpt-5.5", "args": ["acp"]})
    _, agent = sup.resolve("codex", None)
    assert agent.name == DEFAULT_AGENT
    assert agent.command == "codex-acp"
    assert agent.model == "gpt-5.5"
    assert agent.args == ["acp"]


def test_profile_with_multiple_agents():
    sup = _sup(
        work={
            "agents": {
                "claude": {"agent_command": "claude-agent-acp", "model": "sonnet"},
                "codex": {"agent_command": "codex-acp", "args": ["acp"]},
                # same vendor, different config
                "claude-opus": {"agent_command": "claude-agent-acp", "model": "opus"},
            }
        }
    )
    profile = sup.get_profile("work")
    assert set(profile.agents) == {"claude", "codex", "claude-opus"}
    assert sup.resolve("work", "codex")[1].command == "codex-acp"
    assert sup.resolve("work", "claude-opus")[1].model == "opus"


def test_unknown_profile_and_agent_raise():
    sup = _sup(work={"agents": {"a": {"agent_command": "x"}}})
    with pytest.raises(ProfileUnknownError):
        sup.get_profile("nope")
    with pytest.raises(ProfileUnknownError):
        sup.resolve("work", "ghost")


def test_default_agent_when_no_named_default():
    # A profile whose sole agent isn't named "default" still resolves agent=None.
    sup = _sup(work={"agents": {"only": {"agent_command": "x"}}})
    _, agent = sup.resolve("work", None)
    assert agent.name == "only"


def test_profile_list_nested_and_not_running():
    sup = _sup(work={"agents": {"a": {"agent_command": "x"}, "b": {"agent_command": "y"}}})
    rows = {r["name"]: r for r in sup.profile_list()}
    assert DEFAULT_PROFILE in rows
    work_agents = {a["name"]: a for a in rows["work"]["agents"]}
    assert set(work_agents) == {"a", "b"}
    assert work_agents["a"]["running"] is False
    assert work_agents["a"]["sessions"] == 0


def test_agent_mcp_servers_and_env_carried():
    sup = _sup(
        p={
            "agents": {
                "x": {
                    "agent_command": "x",
                    "mcp_servers": [{"name": "fs", "command": "mcp-fs"}],
                    "env": {"A": "1"},
                }
            }
        }
    )
    _, agent = sup.resolve("p", "x")
    assert agent.mcp_servers == [{"name": "fs", "command": "mcp-fs"}]
    assert agent.env == {"A": "1"}
