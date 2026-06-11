"""agentctl one-shot mode (#53): open/list subcommands, exit codes, REPL intact."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

from blemees_agent.cli import EXIT_OK, EXIT_OPEN_FAILED, EXIT_UNREACHABLE, main
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")

pytestmark = pytest.mark.asyncio


@contextlib.asynccontextmanager
async def _daemon(socket_path):
    cfg = Config(
        socket_path=str(socket_path),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        idle_timeout_s=60,
        max_concurrent_sessions=8,
    )
    daemon = Daemon(cfg, configure("error"))
    await daemon.start()
    serve = asyncio.create_task(daemon.serve_forever())
    try:
        yield
    finally:
        daemon.request_shutdown()
        try:
            await asyncio.wait_for(serve, timeout=5.0)
        except TimeoutError:
            serve.cancel()


def _main_in_thread(argv):
    """main() calls asyncio.run, so hop to a thread from async tests."""
    return asyncio.to_thread(main, argv)


async def test_open_prints_sid_and_session_runs_detached(capsys):
    sock = short_socket_path("ctl-os1")
    with socket_cleanup(sock):
        async with _daemon(sock):
            rc = await _main_in_thread(
                ["--socket", str(sock), "open", "--prompt", "finish", "--title", "tracer"]
            )
            out = capsys.readouterr().out.strip()
            assert rc == EXIT_OK
            sid = out.splitlines()[-1]
            assert len(sid) == 36  # uuid printed on stdout

            # The session lives on, detached, and shows up in list output.
            rc = await _main_in_thread(["--socket", str(sock), "list", "--json"])
            assert rc == EXIT_OK
            rows = json.loads(capsys.readouterr().out)
            row = next(r for r in rows if r["session_id"] == sid)
            assert "needs_attention" in row


async def test_open_under_unknown_profile_exits_3(capsys):
    sock = short_socket_path("ctl-os2")
    with socket_cleanup(sock):
        async with _daemon(sock):
            rc = await _main_in_thread(
                ["--socket", str(sock), "open", "--profile", "nope", "--prompt", "hi"]
            )
            assert rc == EXIT_OPEN_FAILED
            err = capsys.readouterr().err
            assert "profile" in err.lower()


async def test_daemon_unreachable_exits_2(capsys):
    rc = await _main_in_thread(
        ["--socket", "/tmp/definitely-not-a-daemon.sock", "open", "--prompt", "hi"]
    )
    assert rc == EXIT_UNREACHABLE
    assert "unreachable" in capsys.readouterr().err


async def test_open_with_no_prompt_on_tty_errors(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    rc = await _main_in_thread(["--socket", "/tmp/x.sock", "open"])
    assert rc == EXIT_OPEN_FAILED
    assert "no prompt" in capsys.readouterr().err


async def test_prompt_from_stdin(monkeypatch, capsys):
    sock = short_socket_path("ctl-os3")
    with socket_cleanup(sock):
        async with _daemon(sock):
            monkeypatch.setattr(sys, "stdin", io.StringIO("finish\n"))
            rc = await _main_in_thread(["--socket", str(sock), "open"])
            assert rc == EXIT_OK
