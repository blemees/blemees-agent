"""Config loading: old (pre-0.11) keys, env vars and flags must not break startup.

The blemees/2 knobs ``claude_bin`` / ``codex_bin`` / ``--claude`` / ``--codex`` /
``BLEMEES_AGENTD_CLAUDE`` / ``BLEMEES_AGENTD_CODEX``,
``max_sessions_per_connection`` and ``stderr_rate_*`` are gone from ``Config``.
Old config files, env vars, launchd plists and systemd units can still carry
them, so ``load()`` must accept them and drop the values.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from blemees_agent.config import Config, load

REMOVED_FIELDS = (
    "claude_bin",
    "codex_bin",
    "max_sessions_per_connection",
    "stderr_rate_lines",
    "stderr_rate_window_s",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("BLEMEES_AGENTD_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def test_removed_fields_are_gone():
    names = {f.name for f in dataclasses.fields(Config)}
    assert names.isdisjoint(REMOVED_FIELDS)


def test_old_config_file_keys_load_and_are_ignored(tmp_path):
    cfg_file = tmp_path / "agentd.toml"
    cfg_file.write_text(
        'claude_bin = "/opt/claude"\n'
        'codex_bin = "/opt/codex"\n'
        "max_sessions_per_connection = 8\n"
        "stderr_rate_lines = 5\n"
        "stderr_rate_window_s = 1\n"
        'agent_command = "my-acp-agent"\n'
        "idle_timeout_s = 42\n"
    )
    cfg, want_version = load(["--config", str(cfg_file)])
    assert want_version is False
    assert cfg.agent_command == "my-acp-agent"
    assert cfg.idle_timeout_s == 42
    for name in REMOVED_FIELDS:
        assert not hasattr(cfg, name)


def test_old_env_vars_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("BLEMEES_AGENTD_CLAUDE", "/opt/claude")
    monkeypatch.setenv("BLEMEES_AGENTD_CODEX", "/opt/codex")
    cfg, _ = load(["--config", str(tmp_path / "missing.toml")])
    assert not hasattr(cfg, "claude_bin")
    assert not hasattr(cfg, "codex_bin")


def test_old_cli_flags_are_accepted_and_ignored(tmp_path):
    # The pre-0.11 README told users to bake these into ExecStart /
    # ProgramArguments; an old unit must still start.
    cfg, _ = load(
        [
            "--config",
            str(tmp_path / "missing.toml"),
            "--claude",
            "/opt/claude",
            "--codex",
            "/opt/codex",
            "--agent",
            "my-acp-agent",
        ]
    )
    assert cfg.agent_command == "my-acp-agent"
    assert not hasattr(cfg, "claude_bin")
    assert not hasattr(cfg, "codex_bin")


def test_old_cli_flags_are_hidden_from_help(capsys):
    with pytest.raises(SystemExit):
        load(["--help"])
    out = capsys.readouterr().out
    assert "--claude" not in out
    assert "--codex" not in out
    assert "--agent" in out
