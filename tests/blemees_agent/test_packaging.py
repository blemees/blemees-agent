"""Shipped service files under ``packaging/blemees-agentd/``."""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[2] / "packaging" / "blemees-agentd"
PLIST = PACKAGING / "com.blemees.blemees-agentd.plist"
LOG_REL = "Library/Logs/blemees-agentd.log"


def _plist() -> dict:
    with PLIST.open("rb") as fh:
        return plistlib.load(fh)


def test_plist_parses_and_keeps_service_shape():
    data = _plist()
    assert data["Label"] == "com.blemees.blemees-agentd"
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ThrottleInterval"] == 5


def test_plist_logs_to_per_user_library_logs_not_tmp():
    data = _plist()
    # launchd does not expand ~ / $HOME in these keys, so they must not be
    # used; the shell in ProgramArguments does the redirect instead.
    assert "StandardOutPath" not in data
    assert "StandardErrorPath" not in data
    args = data["ProgramArguments"]
    assert args[:2] == ["/bin/sh", "-c"]
    assert f'"$HOME/{LOG_REL}"' in args[2]
    assert args[2].startswith('mkdir -p "$HOME/Library/Logs" && exec blemees-agentd ')
    assert not any("/tmp" in a for a in args)


@pytest.mark.parametrize("logs_dir_exists", [True, False], ids=["logs-dir", "no-logs-dir"])
def test_plist_command_writes_stdout_and_stderr_to_home_log(tmp_path, logs_dir_exists):
    home = tmp_path / "home"
    if logs_dir_exists:
        (home / "Library" / "Logs").mkdir(parents=True)
    else:
        # A HOME without Library/Logs: the command must create it.
        home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "blemees-agentd"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "print('to-stdout', flush=True)\n"
        "print('to-stderr', file=sys.stderr, flush=True)\n"
        "sys.exit(3)\n"
    )
    fake.chmod(0o755)

    args = _plist()["ProgramArguments"]
    env = {"HOME": str(home), "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)

    # exec keeps the daemon's exit status (KeepAlive relies on it) and
    # nothing leaks to launchd's own stdout/stderr.
    assert proc.returncode == 3
    assert proc.stdout == ""
    assert proc.stderr == ""
    log = (home / LOG_REL).read_text()
    assert "to-stdout" in log
    assert "to-stderr" in log


def test_service_files_do_not_reference_removed_env_vars():
    for path in PACKAGING.iterdir():
        text = path.read_text()
        assert "BLEMEES_AGENTD_CLAUDE" not in text, path
        assert "BLEMEES_AGENTD_CODEX" not in text, path
