"""Daemon-level over-wire profile CRUD (#25): create/update/delete, persistence,
agent_unavailable, and delete-safety with live sessions."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure

from .conftest import _StreamClient, short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")
GOOD = sys.executable
MISSING = "definitely-not-a-real-binary-xyz123"

pytestmark = pytest.mark.asyncio


@contextlib.asynccontextmanager
async def _daemon(state_dir, socket_path, *, profiles=None):
    cfg = Config(
        socket_path=str(socket_path),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        state_dir=str(state_dir),
        profiles=profiles or {},
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


async def _connect(socket_path) -> _StreamClient:
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    c = _StreamClient(reader, writer)
    await c.send({"type": "hello", "client": "t", "protocol": PROTOCOL_VERSION})
    assert (await c.recv())["type"] == "hello_ack"
    return c


async def _profile_names(c) -> dict:
    await c.send({"type": "profile.list", "id": "l"})
    reply = await c.wait_for(lambda e: e.get("type") == "profiles")
    return {p["name"]: p for p in reply["profiles"]}


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "state"


# ---- create + use + persist ----------------------------------------


async def test_create_then_open_under_new_profile(state_dir):
    sock = short_socket_path("blemeesd-pc1")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {
                            "name": "mine",
                            "agent": {"agent_command": GOOD, "args": [FAKE_ACP]},
                        },
                    }
                )
                created = await c.wait_for(lambda e: e.get("type") == "profile.created")
                assert created["name"] == "mine"

                rows = await _profile_names(c)
                assert rows["mine"]["source"] == "dynamic"
                assert rows["default"]["source"] == "config"

                # The new profile is usable for a session.
                await c.send(
                    {
                        "type": "session.open",
                        "id": "o",
                        "session_id": "s1",
                        "profile": "mine",
                        "options": {},
                    }
                )
                opened = await c.wait_for(lambda e: e.get("type") == "session.opened")
                assert opened["profile"] == "mine"
            finally:
                await c.close()


async def test_created_profile_survives_restart(state_dir):
    sock1 = short_socket_path("blemeesd-pc2a")
    sock2 = short_socket_path("blemeesd-pc2b")
    with socket_cleanup(sock1), socket_cleanup(sock2):
        async with _daemon(state_dir, sock1):
            c = await _connect(sock1)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {
                            "name": "persisted",
                            "agent": {"agent_command": GOOD},
                            "model": "sonnet",
                        },
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.created")
            finally:
                await c.close()

        # Fresh daemon over the same state dir: the profile is back.
        async with _daemon(state_dir, sock2):
            c = await _connect(sock2)
            try:
                rows = await _profile_names(c)
                assert "persisted" in rows
                assert rows["persisted"]["source"] == "dynamic"
            finally:
                await c.close()


# ---- update + delete ------------------------------------------------


async def test_update_and_delete_dynamic_profile(state_dir):
    sock = short_socket_path("blemeesd-pc3")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {"name": "mine", "agent": {"agent_command": GOOD}},
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.created")

                await c.send(
                    {
                        "type": "profile.update",
                        "id": "u",
                        "name": "mine",
                        "profile": {"agent": {"agent_command": GOOD}, "model": "opus"},
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.updated")

                await c.send({"type": "profile.delete", "id": "d", "name": "mine"})
                await c.wait_for(lambda e: e.get("type") == "profile.deleted")
                assert "mine" not in await _profile_names(c)
            finally:
                await c.close()


async def test_delete_config_profile_rejected(state_dir):
    sock = short_socket_path("blemeesd-pc4")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock, profiles={"cfg": {"agent_command": GOOD}}):
            c = await _connect(sock)
            try:
                await c.send({"type": "profile.delete", "id": "d", "name": "cfg"})
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "profile_protected"
            finally:
                await c.close()


async def test_delete_profile_with_live_session_rejected(state_dir):
    sock = short_socket_path("blemeesd-pc5")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {
                            "name": "mine",
                            "agent": {"agent_command": GOOD, "args": [FAKE_ACP]},
                        },
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.created")
                await c.send(
                    {
                        "type": "session.open",
                        "id": "o",
                        "session_id": "s1",
                        "profile": "mine",
                        "options": {},
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "session.opened")

                await c.send({"type": "profile.delete", "id": "d", "name": "mine"})
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "profile_in_use"

                # After closing the session, delete succeeds.
                await c.send(
                    {"type": "session.close", "id": "x", "session_id": "s1", "delete": True}
                )
                await c.wait_for(lambda e: e.get("type") == "session.closed")
                await c.send({"type": "profile.delete", "id": "d2", "name": "mine"})
                await c.wait_for(lambda e: e.get("type") == "profile.deleted")
            finally:
                await c.close()


# ---- agent_unavailable ----------------------------------------------


async def test_create_with_missing_binary_yields_agent_unavailable(state_dir):
    sock = short_socket_path("blemeesd-pc6")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {"name": "bad", "agent": {"agent_command": MISSING}},
                    }
                )
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "agent_unavailable"
                # Not registered.
                assert "bad" not in await _profile_names(c)
            finally:
                await c.close()


async def test_create_duplicate_yields_profile_exists(state_dir):
    sock = short_socket_path("blemeesd-pc7")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send({"type": "profile.delete", "id": "z", "name": "default"})
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "profile_protected"  # default is config-managed
            finally:
                await c.close()
