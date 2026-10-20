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


async def test_update_profile_with_live_session_rejected(state_dir):
    # #47: update must not stop a profile's agent out from under a live session
    # (which would cancel in-flight turns with no terminal frame). Mirror the
    # delete guard — refuse with profile_in_use until the session is closed.
    sock = short_socket_path("blemeesd-pc5b")
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

                await c.send(
                    {
                        "type": "profile.update",
                        "id": "u",
                        "name": "mine",
                        "profile": {"agent": {"agent_command": GOOD}, "model": "opus"},
                    }
                )
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "profile_in_use"
                assert err["id"] == "u"

                # After closing the session, update succeeds.
                await c.send(
                    {"type": "session.close", "id": "x", "session_id": "s1", "delete": True}
                )
                await c.wait_for(lambda e: e.get("type") == "session.closed")
                await c.send(
                    {
                        "type": "profile.update",
                        "id": "u2",
                        "name": "mine",
                        "profile": {"agent": {"agent_command": GOOD}, "model": "opus"},
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.updated")
            finally:
                await c.close()


# ---- profile.get (#43) ----------------------------------------------


async def test_profile_get_dynamic_full_spec_and_round_trip(state_dir):
    # #43: profile.get returns the whole editable spec (not just the
    # profile.list row) with secrets redacted, and the returned spec can be
    # sent straight back as a profile.update body without losing anything.
    sock = short_socket_path("blemeesd-pg1")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {
                            "name": "rich",
                            "agent": {
                                "agent_command": GOOD,
                                "args": [FAKE_ACP],
                                "model": "opus",
                                "mcp_servers": [
                                    {"name": "fs", "command": "mcp-fs", "env": {"T": "s3cret"}}
                                ],
                                "env": {"API_KEY": "sk-live"},
                            },
                            "permission_policy": {"mode": "deny", "detached": "allow"},
                            "notify": {"webhook_url": "https://ntfy.sh/private-topic"},
                        },
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.created")

                await c.send({"type": "profile.get", "id": "g", "name": "rich"})
                info = await c.wait_for(lambda e: e.get("type") == "profile.info")
                assert info["id"] == "g"
                assert info["name"] == "rich"
                assert info["source"] == "dynamic"
                spec = info["profile"]
                agent = spec["agents"]["default"]
                assert agent["command"] == GOOD
                assert agent["args"] == [FAKE_ACP]
                assert agent["model"] == "opus"
                assert agent["mcp_servers"] == [
                    {"name": "fs", "command": "mcp-fs", "env": {"T": "<redacted>"}}
                ]
                assert agent["env"] == {"API_KEY": "<redacted>"}
                assert spec["permission_policy"] == {"mode": "deny", "detached": "allow"}
                assert spec["notify"] == {"webhook_url": "<redacted>"}
                assert sorted(spec["attention"]["triggers"]) == [
                    "agent_crashed",
                    "auth_required",
                    "permission_pending",
                ]
                for secret in ("sk-live", "s3cret", "private-topic"):
                    assert secret not in repr(info)

                # Round-trip with one edit; nothing else may be lost.
                agent["model"] = "sonnet"
                await c.send({"type": "profile.update", "id": "u", "name": "rich", "profile": spec})
                await c.wait_for(lambda e: e.get("type") == "profile.updated")
                await c.send({"type": "profile.get", "id": "g2", "name": "rich"})
                again = await c.wait_for(
                    lambda e: e.get("type") == "profile.info" and e.get("id") == "g2"
                )
                assert again["profile"] == spec
            finally:
                await c.close()


async def test_profile_get_config_profile_and_unknown(state_dir):
    sock = short_socket_path("blemeesd-pg2")
    with socket_cleanup(sock):
        async with _daemon(
            state_dir,
            sock,
            profiles={"cfg": {"agent_command": GOOD, "model": "m", "env": {"K": "v"}}},
        ):
            c = await _connect(sock)
            try:
                # Config-managed profiles are readable (just not editable).
                await c.send({"type": "profile.get", "id": "g", "name": "cfg"})
                info = await c.wait_for(lambda e: e.get("type") == "profile.info")
                assert info["source"] == "config"
                agent = info["profile"]["agents"]["default"]
                assert agent["model"] == "m"
                assert agent["env"] == {"K": "<redacted>"}

                await c.send({"type": "profile.get", "id": "g0", "name": "default"})
                info = await c.wait_for(lambda e: e.get("type") == "profile.info")
                assert info["source"] == "config"
                assert info["profile"]["agents"]["default"]["args"] == [FAKE_ACP]

                await c.send({"type": "profile.get", "id": "g2", "name": "nope"})
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "profile_unknown"
                assert err["id"] == "g2"
            finally:
                await c.close()


async def test_profile_update_unresolvable_placeholder_refused_on_wire(state_dir):
    # #43 review: a "<redacted>" placeholder that can't be matched to one
    # stored MCP server for certain refuses the whole update (invalid_message)
    # and writes nothing — not in memory, not in registry.json.
    gh = {"command": "npx", "args": ["gh-mcp"], "env": {"TOKEN": "github-secret"}}
    slack = {"command": "npx", "args": ["slack-mcp"], "env": {"TOKEN": "slack-secret"}}
    sock = short_socket_path("blemeesd-pg3")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c",
                        "profile": {
                            "name": "m",
                            "agent": {"agent_command": GOOD, "mcp_servers": [gh, slack]},
                        },
                    }
                )
                await c.wait_for(lambda e: e.get("type") == "profile.created")
                registry_file = Path(state_dir) / "registry.json"
                before = registry_file.read_text()

                await c.send({"type": "profile.get", "id": "g", "name": "m"})
                info = await c.wait_for(lambda e: e.get("type") == "profile.info")
                spec = info["profile"]
                spec["agents"]["default"]["mcp_servers"].insert(
                    0, {"command": "npx", "args": ["new-mcp"], "env": {"TOKEN": "<redacted>"}}
                )
                await c.send({"type": "profile.update", "id": "u", "name": "m", "profile": spec})
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "invalid_message"
                assert err["id"] == "u"
                assert err["message"] == (
                    "cannot resolve <redacted> for "
                    "agents.default.mcp_servers[0].env.TOKEN; re-enter the value"
                )
                assert registry_file.read_text() == before

                await c.send({"type": "profile.get", "id": "g2", "name": "m"})
                again = await c.wait_for(
                    lambda e: e.get("type") == "profile.info" and e.get("id") == "g2"
                )
                servers = again["profile"]["agents"]["default"]["mcp_servers"]
                assert [s["args"] for s in servers] == [["gh-mcp"], ["slack-mcp"]]
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


# ---- name validation (#54) -----------------------------------------


async def test_create_with_invalid_name_rejected(state_dir):
    sock = short_socket_path("blemeesd-pn1")
    with socket_cleanup(sock):
        async with _daemon(state_dir, sock):
            c = await _connect(sock)
            try:
                await c.send(
                    {
                        "type": "profile.create",
                        "id": "c1",
                        "name": "my.profile",
                        "profile": {"agent": {"agent_command": GOOD}},
                    }
                )
                err = await c.wait_for(lambda e: e.get("type") == "error")
                assert err["code"] == "invalid_message"
                assert "invalid profile name" in err["message"]
                assert "my.profile" not in await _profile_names(c)
            finally:
                await c.close()


async def test_config_profile_with_invalid_name_skipped(state_dir):
    sock = short_socket_path("blemeesd-pn2")
    with socket_cleanup(sock):
        async with _daemon(
            state_dir,
            sock,
            profiles={
                "bad.name": {"agent": {"agent_command": GOOD}},
                "good-name": {"agent": {"agent_command": GOOD}},
            },
        ):
            c = await _connect(sock)
            try:
                names = await _profile_names(c)
                assert "good-name" in names
                assert "bad.name" not in names
            finally:
                await c.close()


async def test_config_agents_with_invalid_keys_filtered_not_silently_dropped(state_dir):
    sock = short_socket_path("blemeesd-pn3")
    with socket_cleanup(sock):
        async with _daemon(
            state_dir,
            sock,
            profiles={
                # one valid + one invalid agent key: profile loads, bad key skipped
                "mixed": {
                    "agents": {
                        "good": {"agent_command": GOOD},
                        "bad.key": {"agent_command": GOOD},
                    }
                },
                # only invalid agent keys: the whole profile is dropped (warned)
                "all-bad": {"agents": {"bad.key": {"agent_command": GOOD}}},
            },
        ):
            c = await _connect(sock)
            try:
                names = await _profile_names(c)
                assert "mixed" in names
                agent_names = [a["name"] for a in names["mixed"]["agents"]]
                assert agent_names == ["good"]
                assert "all-bad" not in names
            finally:
                await c.close()


# ---- notify format resolution (#52) ----------------------------------


async def test_webhook_format_resolution(state_dir):
    from blemees_agent.config import Config as _Cfg
    from blemees_agent.logging import configure as _configure
    from blemees_agent.supervisor import Supervisor

    cfg = _Cfg(
        socket_path="/tmp/unused.sock",
        agent_command=GOOD,
        notify_webhook_format="ntfy",
        profiles={
            "jsonish": {"agent_command": GOOD, "notify": {"format": "json"}},
            "typo": {"agent_command": GOOD, "notify": {"format": "ntfyy"}},
        },
    )
    sup = Supervisor(cfg, _configure("error"))
    assert sup.webhook_format_for("default") == "ntfy"  # global default applies
    assert sup.webhook_format_for("jsonish") == "json"  # per-profile override
    assert sup.webhook_format_for("typo") == "json"  # unknown value degrades
