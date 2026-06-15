"""Daemon-level notify service over the wire (#24): triggers, webhook, status."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from pathlib import Path

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.config import Config
from blemees_agent.daemon import Daemon
from blemees_agent.logging import configure
from blemees_agent.notify import WebhookSink
from blemees_agent.supervisor import Supervisor

from .conftest import _StreamClient, short_socket_path, socket_cleanup

FAKE_ACP = str(Path(__file__).parent / "fake_acp.py")

pytestmark = pytest.mark.asyncio


@pytest.fixture
def webhook_posts(monkeypatch):
    """Capture webhook POSTs instead of making real HTTP calls. Patched on the
    class before any daemon builds its WebhookSink (which binds ``_http_post``
    at construction)."""
    posts: list[tuple[str, dict]] = []

    def fake_post(self, url, body, headers):
        posts.append((url, json.loads(body)))

    monkeypatch.setattr(WebhookSink, "_http_post", fake_post)
    return posts


@contextlib.asynccontextmanager
async def _daemon(socket_path, *, webhook_url="https://hook.test/global", profiles=None):
    cfg = Config(
        socket_path=str(socket_path),
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        notify_webhook_url=webhook_url,
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


async def _open(c, sid):
    await c.send({"type": "session.open", "id": "o", "session_id": sid, "options": {}})
    await c.wait_for(lambda e: e.get("type") == "session.opened")


async def _poll_status(c, predicate, *, timeout=10.0):
    """Send status until ``predicate(reply)`` holds; returns that reply."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await c.send({"type": "status", "id": "st"})
        reply = await c.wait_for(lambda e: e.get("type") == "status_reply", timeout=5.0)
        if predicate(reply):
            return reply
        await asyncio.sleep(0.1)
    raise TimeoutError("status predicate never held")


# ---- detached permission stall --------------------------------------


async def test_detached_permission_stall_fires_webhook(webhook_posts):
    sock = short_socket_path("blemeesd-notify1")
    with socket_cleanup(sock):
        async with _daemon(sock):
            c1 = await _connect(sock)
            await _open(c1, "s-stall")
            # Start a turn that streams a chunk, then (after a beat) requests a
            # permission decision.
            await c1.send({"type": "session.prompt", "session_id": "s-stall", "prompt": "stall"})
            await c1.wait_for(lambda e: e.get("type") == "session.update", timeout=10.0)
            # Drop the owner mid-turn → soft-detach. The permission request now
            # arrives with no owner → stall → needs_attention + webhook.
            await c1.close()

            c2 = await _connect(sock)
            try:
                reply = await _poll_status(c2, lambda r: r.get("attention"))
                attention = reply["attention"]
                assert attention[0]["reason"] == "permission_pending"
                assert attention[0]["session_id"] == "s-stall"

                # session.list reflects the per-session flag + reason.
                await c2.send({"type": "session.list", "id": "l"})
                listing = await c2.wait_for(lambda e: e.get("type") == "sessions")
                row = next(r for r in listing["sessions"] if r["session_id"] == "s-stall")
                assert row["needs_attention"] is True
                assert row["attention_reason"] == "permission_pending"
            finally:
                await c2.close()

    # The webhook fired with the documented payload to the global URL.
    assert webhook_posts, "expected a webhook POST"
    url, payload = webhook_posts[0]
    assert url == "https://hook.test/global"
    assert payload["type"] == "blemees.notify"
    assert payload["reason"] == "permission_pending"
    assert payload["session_id"] == "s-stall"


# ---- turn-complete is NOT a trigger ---------------------------------


async def test_turn_complete_does_not_fire(webhook_posts):
    sock = short_socket_path("blemeesd-notify2")
    with socket_cleanup(sock):
        async with _daemon(sock):
            c = await _connect(sock)
            try:
                await _open(c, "s-quiet")
                await c.send({"type": "session.prompt", "session_id": "s-quiet", "prompt": "pong"})
                await c.wait_for(lambda e: e.get("type") == "session.result", timeout=30.0)

                await c.send({"type": "status", "id": "st"})
                reply = await c.wait_for(lambda e: e.get("type") == "status_reply")
                assert reply["attention"] == []

                await c.send({"type": "session.list", "id": "l"})
                listing = await c.wait_for(lambda e: e.get("type") == "sessions")
                row = next(r for r in listing["sessions"] if r["session_id"] == "s-quiet")
                assert row["needs_attention"] is False
            finally:
                await c.close()
    assert webhook_posts == []  # a completed turn must not notify


# ---- notify.test ----------------------------------------------------


async def test_notify_test_fires_event(webhook_posts):
    sock = short_socket_path("blemeesd-notify3")
    with socket_cleanup(sock):
        async with _daemon(sock):
            c = await _connect(sock)
            try:
                await c.send({"type": "notify.test", "id": "nt"})
                reply = await c.wait_for(lambda e: e.get("type") == "notify.test_result")
                assert reply["webhook_configured"] is True
                assert reply["notification"]["reason"] == "test"
            finally:
                await c.close()
    assert len(webhook_posts) == 1
    _url, payload = webhook_posts[0]
    assert payload["reason"] == "test"


async def test_notify_test_reports_no_webhook_when_unconfigured(webhook_posts):
    sock = short_socket_path("blemeesd-notify4")
    with socket_cleanup(sock):
        async with _daemon(sock, webhook_url=None):
            c = await _connect(sock)
            try:
                await c.send({"type": "notify.test", "id": "nt"})
                reply = await c.wait_for(lambda e: e.get("type") == "notify.test_result")
                assert reply["webhook_configured"] is False
            finally:
                await c.close()
    assert webhook_posts == []  # no URL → no POST


# ---- per-profile URL resolution (unit) ------------------------------


async def test_webhook_url_resolution_prefers_profile_then_global():
    agent = {"agent_command": sys.executable, "agent_args": [FAKE_ACP]}
    cfg = Config(
        socket_path="/tmp/x.sock",
        agent_command=sys.executable,
        agent_args=[FAKE_ACP],
        notify_webhook_url="https://global",
        profiles={
            "withhook": {**agent, "notify": {"webhook_url": "https://profile"}},
            "nohook": {**agent},
        },
    )
    sup = Supervisor(cfg, configure("error"))
    assert sup.webhook_url_for("withhook") == "https://profile"  # profile wins
    assert sup.webhook_url_for("nohook") == "https://global"  # falls back to global
    assert sup.webhook_url_for("unknown") == "https://global"  # unknown → global


# ---- per-profile attention policy (#51) ------------------------------


async def test_turn_complete_fires_for_opted_in_profile(webhook_posts):
    sock = short_socket_path("blemeesd-notify5")
    profiles = {
        "watch": {
            "agent_command": sys.executable,
            "agent_args": [FAKE_ACP],
            "attention": {"triggers": ["turn_complete"]},
        }
    }
    with socket_cleanup(sock):
        async with _daemon(sock, profiles=profiles):
            c1 = await _connect(sock)
            await c1.send(
                {"type": "session.open", "id": "o", "session_id": "s-watch", "profile": "watch"}
            )
            await c1.wait_for(lambda e: e.get("type") == "session.opened")
            # "finish" streams a chunk then completes ~0.5s later — drop the
            # owner mid-turn so the result lands while detached.
            await c1.send({"type": "session.prompt", "session_id": "s-watch", "prompt": "finish"})
            await c1.wait_for(lambda e: e.get("type") == "session.update", timeout=10.0)
            await c1.close()

            c2 = await _connect(sock)
            try:
                reply = await _poll_status(c2, lambda r: r.get("attention"))
                assert reply["attention"][0]["reason"] == "turn_complete"
                assert reply["attention"][0]["session_id"] == "s-watch"
            finally:
                await c2.close()
    assert webhook_posts, "expected a webhook POST for the opted-in profile"
    assert webhook_posts[0][1]["reason"] == "turn_complete"


async def test_turn_complete_override_via_open_options(webhook_posts):
    # The default profile is blocked-only, but session.open can arm
    # turn_complete for just this session (#51).
    sock = short_socket_path("blemeesd-notify6")
    with socket_cleanup(sock):
        async with _daemon(sock):
            c1 = await _connect(sock)
            await c1.send(
                {
                    "type": "session.open",
                    "id": "o",
                    "session_id": "s-once",
                    "options": {"attention_triggers": ["turn_complete", "permission_pending"]},
                }
            )
            await c1.wait_for(lambda e: e.get("type") == "session.opened")
            await c1.send({"type": "session.prompt", "session_id": "s-once", "prompt": "finish"})
            await c1.wait_for(lambda e: e.get("type") == "session.update", timeout=10.0)
            await c1.close()

            c2 = await _connect(sock)
            try:
                reply = await _poll_status(c2, lambda r: r.get("attention"))
                assert reply["attention"][0]["reason"] == "turn_complete"
            finally:
                await c2.close()
    assert webhook_posts and webhook_posts[0][1]["reason"] == "turn_complete"
