"""Validate the shipped blemees/3 JSON Schemas (#30).

Parses + meta-validates every schema, checks that every inbound dispatch
verb and outbound frame the daemon emits has a schema, and validates a
canonical example frame against each (plus negative cases that exercise the
strict inbound `additionalProperties:false` contract). Loads via
`importlib.resources`, the same path clients use after `pip install`.
"""

from __future__ import annotations

import pytest

from blemees_agent import schemas

jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402

# Frame types the daemon dispatches (inbound) / emits (outbound). Each maps 1:1
# to a `<direction>/<type>.json` schema; this list is the coverage contract.
INBOUND_TYPES = [
    "hello",
    "ping",
    "status",
    "notify.test",
    "session.open",
    "session.prompt",
    "session.cancel",
    "session.close",
    "session.list",
    "session.info",
    "session.attach",
    "session.detach",
    "session.permission_response",
    "profile.list",
    "profile.start",
    "profile.stop",
    "profile.create",
    "profile.update",
    "profile.delete",
]
OUTBOUND_TYPES = [
    "hello_ack",
    "error",
    "pong",
    "sessions",
    "status_reply",
    "replay_gap",
    "profiles",
    "profile.created",
    "profile.updated",
    "profile.deleted",
    "profile.started",
    "profile.stopped",
    "notify.test_result",
    "session.opened",
    "session.update",
    "session.result",
    "session.error",
    "session.stderr",
    "session.cancelled",
    "session.closed",
    "session.closed_notice",
    "session.attached",
    "session.detached",
    "session.taken",
    "session.info_reply",
    "session.needs_attention",
    "session.attention_cleared",
    "session.request_permission",
]


def _sid(direction: str, type_: str) -> str:
    return f"https://blemees/schemas/{direction}/{type_}.json"


# One canonical, valid example frame per schema, keyed by $id.
EXAMPLES: dict[str, dict] = {
    _sid("inbound", "hello"): {"type": "hello", "protocol": "blemees/3", "client": "tui/1"},
    _sid("inbound", "ping"): {"type": "ping", "id": "p1", "data": {"x": 1}},
    _sid("inbound", "status"): {"type": "status", "id": "s1"},
    _sid("inbound", "notify.test"): {"type": "notify.test", "id": "n1", "profile": "default"},
    _sid("inbound", "session.open"): {
        "type": "session.open",
        "id": "o1",
        "session_id": "s1",
        "resume": False,
        "profile": "default",
        "options": {"cwd": "/tmp"},
    },
    _sid("inbound", "session.prompt"): {
        "type": "session.prompt",
        "session_id": "s1",
        "prompt": "hello",
    },
    _sid("inbound", "session.cancel"): {"type": "session.cancel", "session_id": "s1"},
    _sid("inbound", "session.close"): {
        "type": "session.close",
        "id": "c1",
        "session_id": "s1",
        "delete": True,
    },
    _sid("inbound", "session.list"): {"type": "session.list", "id": "l1", "cwd": "/tmp"},
    _sid("inbound", "session.info"): {"type": "session.info", "id": "i1", "session_id": "s1"},
    _sid("inbound", "session.attach"): {
        "type": "session.attach",
        "id": "a1",
        "session_id": "s1",
        "as": "owner",
        "last_seen_seq": 4,
    },
    _sid("inbound", "session.detach"): {"type": "session.detach", "id": "d1", "session_id": "s1"},
    _sid("inbound", "session.permission_response"): {
        "type": "session.permission_response",
        "session_id": "s1",
        "request_id": "perm_abc",
        "outcome": "selected",
        "option_id": "allow",
    },
    _sid("inbound", "profile.list"): {"type": "profile.list", "id": "pl"},
    _sid("inbound", "profile.start"): {"type": "profile.start", "id": "ps", "name": "claude"},
    _sid("inbound", "profile.stop"): {"type": "profile.stop", "id": "px", "name": "claude"},
    _sid("inbound", "profile.create"): {
        "type": "profile.create",
        "id": "pc",
        "profile": {"name": "mine", "agent": {"agent_command": "claude-agent-acp"}},
    },
    _sid("inbound", "profile.update"): {
        "type": "profile.update",
        "id": "pu",
        "name": "mine",
        "profile": {"agent": {"agent_command": "claude-agent-acp"}, "model": "opus"},
    },
    _sid("inbound", "profile.delete"): {"type": "profile.delete", "id": "pd", "name": "mine"},
    # -- outbound --
    _sid("outbound", "hello_ack"): {
        "type": "hello_ack",
        "daemon": "blemees-agentd/0",
        "protocol": "blemees/3",
        "pid": 10,
        "agents": {"claude-agent-acp": "1.0"},
        "profiles": ["default"],
    },
    _sid("outbound", "error"): {
        "type": "error",
        "code": "session_unknown",
        "message": "no such session",
        "id": "x1",
        "session_id": "s1",
    },
    _sid("outbound", "pong"): {"type": "pong", "id": "p1", "data": {"x": 1}},
    _sid("outbound", "sessions"): {
        "type": "sessions",
        "id": "l1",
        "sessions": [{"session_id": "s1", "profile": "default", "attached": True, "running": True}],
    },
    _sid("outbound", "status_reply"): {
        "type": "status_reply",
        "id": "s1",
        "protocol": "blemees/3",
        "pid": 10,
        "sessions": {"total": 1, "attached": 1, "detached": 0},
        "attention": [],
    },
    _sid("outbound", "replay_gap"): {
        "type": "replay_gap",
        "session_id": "s1",
        "since_seq": 1,
        "first_available_seq": 5,
        "seq": 6,
    },
    _sid("outbound", "profiles"): {
        "type": "profiles",
        "id": "pl",
        "profiles": [{"name": "default", "source": "config", "agents": []}],
    },
    _sid("outbound", "profile.created"): {"type": "profile.created", "id": "pc", "name": "mine"},
    _sid("outbound", "profile.updated"): {"type": "profile.updated", "id": "pu", "name": "mine"},
    _sid("outbound", "profile.deleted"): {"type": "profile.deleted", "id": "pd", "name": "mine"},
    _sid("outbound", "profile.started"): {
        "type": "profile.started",
        "id": "ps",
        "name": "mine",
        "agents_started": 1,
    },
    _sid("outbound", "profile.stopped"): {
        "type": "profile.stopped",
        "id": "px",
        "name": "mine",
        "agents_stopped": 1,
    },
    _sid("outbound", "notify.test_result"): {
        "type": "notify.test_result",
        "id": "n1",
        "profile": "default",
        "webhook_configured": True,
        "notification": {
            "type": "blemees.notify",
            "reason": "test",
            "profile": "default",
            "session_id": "notify-test",
            "title": "blemees: test notification",
            "detail": "test event",
            "ts_ms": 1769000000000,
        },
    },
    _sid("outbound", "session.opened"): {
        "type": "session.opened",
        "id": "o1",
        "session_id": "s1",
        "profile": "default",
        "agent": "default",
        "subprocess_pid": 99,
        "last_seq": 0,
        "view_only": False,
        "native_session_id": "fake-session-1",
    },
    _sid("outbound", "session.update"): {
        "type": "session.update",
        "session_id": "s1",
        "seq": 1,
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "hi"},
        },
    },
    _sid("outbound", "session.result"): {
        "type": "session.result",
        "session_id": "s1",
        "seq": 3,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10},
    },
    _sid("outbound", "session.error"): {
        "type": "session.error",
        "session_id": "s1",
        "seq": 4,
        "code": "agent_crashed",
        "message": "exited",
    },
    _sid("outbound", "session.stderr"): {
        "type": "session.stderr",
        "session_id": "s1",
        "seq": 2,
        "line": "warning: ...",
    },
    _sid("outbound", "session.cancelled"): {
        "type": "session.cancelled",
        "session_id": "s1",
        "was_idle": False,
    },
    _sid("outbound", "session.closed"): {"type": "session.closed", "id": "c1", "session_id": "s1"},
    _sid("outbound", "session.closed_notice"): {
        "type": "session.closed_notice",
        "session_id": "s1",
        "reason": "owner_closed",
    },
    _sid("outbound", "session.attached"): {
        "type": "session.attached",
        "id": "a1",
        "session_id": "s1",
        "role": "owner",
        "last_seq": 7,
    },
    _sid("outbound", "session.detached"): {
        "type": "session.detached",
        "id": "d1",
        "session_id": "s1",
        "was_attached": True,
    },
    _sid("outbound", "session.taken"): {
        "type": "session.taken",
        "session_id": "s1",
        "by_peer_pid": 1234,
    },
    _sid("outbound", "session.info_reply"): {
        "type": "session.info_reply",
        "id": "i1",
        "session_id": "s1",
        "profile": "default",
        "agent": "default",
        "attached": True,
        "subprocess_running": True,
        "needs_attention": False,
        "attention_reason": None,
        "view_only": False,
    },
    _sid("outbound", "session.needs_attention"): {
        "type": "session.needs_attention",
        "session_id": "s1",
        "seq": 5,
        "reason": "permission_pending",
    },
    _sid("outbound", "session.attention_cleared"): {
        "type": "session.attention_cleared",
        "session_id": "s1",
        "seq": 6,
    },
    _sid("outbound", "session.request_permission"): {
        "type": "session.request_permission",
        "session_id": "s1",
        "seq": 7,
        "request_id": "perm_abc",
        "options": [{"option_id": "allow", "name": "Allow", "kind": "allow_once"}],
        "tool_call": {"tool_call_id": "tc1", "title": "Run a command", "status": "pending"},
    },
}

# Frames that must be REJECTED — exercises the strict inbound contract.
NEGATIVE: list[tuple[str, dict]] = [
    # wrong protocol version
    (_sid("inbound", "hello"), {"type": "hello", "protocol": "blemees/2"}),
    # unknown top-level key (additionalProperties:false)
    (_sid("inbound", "session.open"), {"type": "session.open", "session_id": "s1", "bogus": 1}),
    # missing required session_id
    (_sid("inbound", "session.prompt"), {"type": "session.prompt", "prompt": "hi"}),
    # bad enum value
    (
        _sid("inbound", "session.permission_response"),
        {
            "type": "session.permission_response",
            "session_id": "s1",
            "request_id": "r",
            "outcome": "maybe",
        },
    ),
    # bad attach role
    (
        _sid("inbound", "session.attach"),
        {"type": "session.attach", "session_id": "s1", "as": "admin"},
    ),
]


@pytest.fixture(scope="module")
def store_and_registry():
    store = {s["$id"]: s for s in schemas.iter_schemas()}
    registry = Registry().with_resources(
        [(uri, Resource.from_contents(schema)) for uri, schema in store.items()]
    )
    return store, registry


# ---- loader API + meta-validation -----------------------------------


def test_loader_api():
    assert schemas.load("inbound/hello.json")["$id"] == _sid("inbound", "hello")
    names = {e.name for e in schemas.files().iterdir()}
    assert {"inbound", "outbound", "_common.json"}.issubset(names)


def test_every_schema_meta_validates():
    count = 0
    for schema in schemas.iter_schemas():
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"].startswith("https://blemees/schemas/")
        Draft202012Validator.check_schema(schema)
        count += 1
    assert count == len(INBOUND_TYPES) + len(OUTBOUND_TYPES) + 1  # +1 = _common


# ---- coverage: every dispatched/emitted frame has a schema ----------


@pytest.mark.parametrize("type_", INBOUND_TYPES)
def test_inbound_type_has_schema(type_):
    assert schemas.load(f"inbound/{type_}.json")["$id"] == _sid("inbound", type_)


@pytest.mark.parametrize("type_", OUTBOUND_TYPES)
def test_outbound_type_has_schema(type_):
    assert schemas.load(f"outbound/{type_}.json")["$id"] == _sid("outbound", type_)


def test_every_frame_has_an_example():
    expected = {_sid("inbound", t) for t in INBOUND_TYPES} | {
        _sid("outbound", t) for t in OUTBOUND_TYPES
    }
    assert set(EXAMPLES) == expected


# ---- example validation (positive + negative) -----------------------


@pytest.mark.parametrize("schema_id", sorted(EXAMPLES))
def test_canonical_example_validates(store_and_registry, schema_id):
    store, registry = store_and_registry
    Draft202012Validator(store[schema_id], registry=registry).validate(EXAMPLES[schema_id])


@pytest.mark.parametrize("schema_id,frame", NEGATIVE)
def test_invalid_frame_rejected(store_and_registry, schema_id, frame):
    store, registry = store_and_registry
    with pytest.raises(jsonschema.ValidationError):
        Draft202012Validator(store[schema_id], registry=registry).validate(frame)
