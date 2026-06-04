"""Daemon-level integration for the ACP backend over the blemees/3 wire (#16).

Drives a live ``Daemon`` configured to spawn ``fake_acp.py`` as its ACP agent,
over the real Unix socket: ``hello`` → ``session.open`` → ``session.prompt`` →
streamed ``session.update`` → ``session.result`` (monotonic ``seq``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import _StreamClient, short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")
SID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
async def acp_daemon():
    socket_path = short_socket_path("blemeesd-acp")
    with socket_cleanup(socket_path):
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
            yield str(socket_path)
        finally:
            daemon.request_shutdown()
            try:
                await asyncio.wait_for(serve, timeout=5.0)
            except TimeoutError:
                serve.cancel()


async def _connect(socket_path: str) -> _StreamClient:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    client = _StreamClient(reader, writer)
    await client.send({"type": "hello", "client": "test/1", "protocol": PROTOCOL_VERSION})
    ack = await client.recv()
    assert ack["type"] == "hello_ack"
    assert ack["protocol"] == PROTOCOL_VERSION
    return client


async def test_open_and_drive_acp_turn(acp_daemon):
    client = await _connect(acp_daemon)
    try:
        await client.send({"type": "session.open", "id": "o1", "session_id": SID, "options": {}})
        opened = await client.wait_for(lambda e: e.get("type") == "session.opened")
        assert opened["subprocess_pid"]
        assert opened["view_only"] is False

        await client.send({"type": "session.prompt", "session_id": SID, "prompt": "say pong"})
        frames = await client.wait_for(
            lambda e: e.get("type") == "session.result", collect=True, timeout=30.0
        )
    finally:
        await client.close()

    updates = [f for f in frames if f.get("type") == "session.update"]
    result = next(f for f in frames if f.get("type") == "session.result")
    assert result["stop_reason"] == "end_turn"
    assert updates, "expected at least one streamed session.update"
    # ACP update payload carried verbatim (camelCase wire shape).
    assert updates[0]["update"]["sessionUpdate"] == "agent_message_chunk"
    # Frames carry a session-assigned monotonic seq.
    assert result["seq"] > updates[0]["seq"]


async def test_prompt_without_open_is_unknown_session(acp_daemon):
    client = await _connect(acp_daemon)
    try:
        await client.send({"type": "session.prompt", "session_id": "no-such", "prompt": "hi"})
        err = await client.wait_for(lambda e: e.get("type") == "error")
    finally:
        await client.close()
    assert err["code"] == "session_unknown"
