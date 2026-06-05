"""Permission policy + relay tests (#20).

Unit-level tests of Session.decide_permission cover the policy matrix
(allow/deny/relay, detached behavior, remembered *_always) deterministically;
the daemon-level relay round-trip lives in test_daemon_permissions.py.
"""

from __future__ import annotations

import asyncio

from blemees_agent.protocol import OpenMessage
from blemees_agent.session import Session

OPTIONS = [
    {"option_id": "a", "name": "Allow", "kind": "allow_once"},
    {"option_id": "aa", "name": "Always", "kind": "allow_always"},
    {"option_id": "d", "name": "Deny", "kind": "reject_once"},
    {"option_id": "da", "name": "Deny always", "kind": "reject_always"},
]


def _session(policy: dict) -> Session:
    msg = OpenMessage(id=None, session_id="s1", options={}, resume=False)
    sess = Session(session_id="s1", open_msg=msg, cwd=None, permission_policy=policy)
    return sess


def _capture(emitted: list[dict]):
    """An async writer (on_event awaits it) that records frames."""

    async def _writer(frame: dict) -> None:
        emitted.append(frame)

    return _writer


async def test_allow_policy_auto_selects_allow_option():
    sess = _session({"mode": "allow", "detached": "stall"})
    sess.connection_id = 1  # attached
    decision = await sess.decide_permission(OPTIONS, {})
    assert decision == {"outcome": "selected", "option_id": "a"}


async def test_deny_policy_auto_selects_reject_option():
    sess = _session({"mode": "deny", "detached": "stall"})
    sess.connection_id = 1
    decision = await sess.decide_permission(OPTIONS, {})
    assert decision == {"outcome": "selected", "option_id": "d"}


async def test_deny_with_no_reject_option_cancels():
    sess = _session({"mode": "deny"})
    sess.connection_id = 1
    decision = await sess.decide_permission(
        [{"option_id": "a", "name": "Allow", "kind": "allow_once"}], {}
    )
    assert decision == {"outcome": "cancelled", "option_id": None}


async def test_detached_allow_resolves_immediately():
    sess = _session({"mode": "relay", "detached": "allow"})
    sess.connection_id = None  # detached
    decision = await sess.decide_permission(OPTIONS, {})
    assert decision["outcome"] == "selected" and decision["option_id"] == "a"


async def test_detached_deny_resolves_immediately():
    sess = _session({"mode": "relay", "detached": "deny"})
    sess.connection_id = None
    decision = await sess.decide_permission(OPTIONS, {})
    assert decision["option_id"] == "d"


async def test_relay_emits_request_and_awaits_owner_response():
    sess = _session({"mode": "relay", "detached": "stall"})
    sess.connection_id = 1  # owner attached
    emitted: list[dict] = []
    sess._writer = _capture(emitted)

    task = asyncio.create_task(sess.decide_permission(OPTIONS, {"title": "x"}))
    await asyncio.sleep(0.05)
    req = next(f for f in emitted if f["type"] == "session.request_permission")
    assert req["options"] == OPTIONS
    # Owner selects "allow".
    assert sess.resolve_permission(req["request_id"], "selected", "a") is True
    decision = await asyncio.wait_for(task, timeout=2.0)
    assert decision == {"outcome": "selected", "option_id": "a"}


async def test_detached_stall_marks_attention_then_resolves_on_response():
    sess = _session({"mode": "relay", "detached": "stall"})
    sess.connection_id = None  # detached → stall
    emitted: list[dict] = []
    sess._writer = _capture(emitted)

    task = asyncio.create_task(sess.decide_permission(OPTIONS, {}))
    await asyncio.sleep(0.05)
    assert any(f["type"] == "session.needs_attention" for f in emitted)
    req = next(f for f in emitted if f["type"] == "session.request_permission")
    sess.resolve_permission(req["request_id"], "selected", "a")
    decision = await asyncio.wait_for(task, timeout=2.0)
    assert decision["option_id"] == "a"
    assert any(f["type"] == "session.attention_cleared" for f in emitted)


async def test_allow_always_is_remembered_for_session():
    sess = _session({"mode": "relay", "detached": "stall"})
    sess.connection_id = 1
    emitted: list[dict] = []
    sess._writer = _capture(emitted)

    task = asyncio.create_task(sess.decide_permission(OPTIONS, {}))
    await asyncio.sleep(0.05)
    req = next(f for f in emitted if f["type"] == "session.request_permission")
    sess.resolve_permission(req["request_id"], "selected", "aa")  # allow_always
    await asyncio.wait_for(task, timeout=2.0)
    assert sess.remembered_permission == "allow"

    # Next request auto-allows without relaying.
    emitted.clear()
    decision = await sess.decide_permission(OPTIONS, {})
    assert decision["option_id"] == "a"
    assert not any(f["type"] == "session.request_permission" for f in emitted)


def test_resolve_unknown_request_is_false():
    sess = _session({"mode": "relay"})
    assert sess.resolve_permission("nope", "selected", "a") is False
