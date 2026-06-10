"""Unit tests for blemees_agent.protocol (blemees/3)."""

from __future__ import annotations

import json

import pytest

from blemees_agent import PROTOCOL_VERSION
from blemees_agent.errors import OversizeMessageError, ProtocolError
from blemees_agent.protocol import (
    encode,
    error_frame,
    hello_ack,
    parse_attach,
    parse_cancel,
    parse_close,
    parse_detach,
    parse_hello,
    parse_line,
    parse_list,
    parse_open,
    parse_ping,
    parse_profile_action,
    parse_profile_mutate,
    parse_prompt,
    parse_session_info,
    parse_status,
)

# ---------------------------------------------------------------------------
# Framing / encode / decode
# ---------------------------------------------------------------------------


def test_encode_is_newline_terminated_utf8():
    data = encode({"type": "hello", "emoji": "🌟"})
    assert data.endswith(b"\n")
    assert b"\n" not in data[:-1]
    assert json.loads(data)["emoji"] == "🌟"


def test_parse_line_accepts_valid_object():
    obj = parse_line(b'{"type":"hello","protocol":"blemees/3"}\n')
    assert obj["type"] == "hello"


@pytest.mark.parametrize("raw", [b"[]\n", b'{"foo":"bar"}\n', b"\n", b"not-json\n"])
def test_parse_line_rejects_bad_input(raw):
    with pytest.raises(ProtocolError):
        parse_line(raw)


def test_parse_line_rejects_oversize():
    with pytest.raises(OversizeMessageError):
        parse_line(b"x" * 100, max_bytes=50)


def test_parse_line_handles_surrogate_pairs_and_nul():
    raw = json.dumps({"type": "x", "text": "\U0001f600\x00"}).encode("utf-8") + b"\n"
    assert parse_line(raw)["text"] == "\U0001f600\x00"


# ---------------------------------------------------------------------------
# hello / hello_ack
# ---------------------------------------------------------------------------


def test_parse_hello_requires_protocol():
    with pytest.raises(ProtocolError):
        parse_hello({"type": "hello"})


def test_parse_hello_ok():
    h = parse_hello({"type": "hello", "protocol": "blemees/3", "client": "t/0.1"})
    assert h.protocol == "blemees/3"
    assert h.client == "t/0.1"


def test_parse_hello_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_hello({"type": "hello", "protocol": "blemees/3", "unknown": True})


def test_hello_ack_shape():
    ack = hello_ack("0.2.0", 1234, {"claude-agent-acp": "available"}, ["default", "codex"])
    assert ack["type"] == "hello_ack"
    assert ack["daemon"] == "blemees-agentd/0.2.0"
    assert ack["protocol"] == PROTOCOL_VERSION
    assert ack["pid"] == 1234
    assert ack["agents"] == {"claude-agent-acp": "available"}
    assert ack["profiles"] == ["default", "codex"]


# ---------------------------------------------------------------------------
# session.open
# ---------------------------------------------------------------------------


def _open(**overrides):
    base = {"type": "session.open", "session_id": "s1"}
    base.update(overrides)
    return base


def test_parse_open_minimal():
    msg = parse_open(_open())
    assert msg.session_id == "s1"
    assert msg.options == {}
    assert msg.resume is False
    assert msg.last_seen_seq is None


def test_parse_open_requires_session():
    with pytest.raises(ProtocolError):
        parse_open({"type": "session.open", "options": {}})


def test_parse_open_options_and_resume():
    msg = parse_open(_open(options={"cwd": "/p", "model": "sonnet"}, resume=True, last_seen_seq=42))
    assert msg.options == {"cwd": "/p", "model": "sonnet"}
    assert msg.resume is True
    assert msg.last_seen_seq == 42


def test_parse_open_rejects_negative_last_seen_seq():
    with pytest.raises(ProtocolError):
        parse_open(_open(last_seen_seq=-1))


def test_parse_open_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_open(_open(backend="claude"))


# ---------------------------------------------------------------------------
# session.prompt
# ---------------------------------------------------------------------------


def test_parse_prompt_string():
    p = parse_prompt({"type": "session.prompt", "session_id": "s1", "prompt": "hello"})
    assert p.message == {"role": "user", "content": "hello"}


def test_parse_prompt_content_blocks():
    blocks = [{"type": "text", "text": "hi"}]
    p = parse_prompt({"type": "session.prompt", "session_id": "s1", "prompt": blocks})
    assert p.message["content"] == blocks


def test_parse_prompt_requires_prompt():
    with pytest.raises(ProtocolError):
        parse_prompt({"type": "session.prompt", "session_id": "s1"})


def test_parse_prompt_rejects_non_str_non_list():
    with pytest.raises(ProtocolError):
        parse_prompt({"type": "session.prompt", "session_id": "s1", "prompt": 42})


def test_parse_prompt_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_prompt({"type": "session.prompt", "session_id": "s1", "prompt": "hi", "extra": 1})


# ---------------------------------------------------------------------------
# cancel / close
# ---------------------------------------------------------------------------


def test_parse_cancel_requires_session():
    with pytest.raises(ProtocolError):
        parse_cancel({"type": "session.cancel"})


def test_parse_cancel_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_cancel({"type": "session.cancel", "session_id": "s1", "extra": 1})


def test_parse_close_delete_flag():
    assert parse_close({"type": "session.close", "session_id": "s1"}).delete is False
    assert parse_close({"type": "session.close", "session_id": "s1", "delete": True}).delete is True


def test_parse_close_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_close({"type": "session.close", "session_id": "s1", "extra": 1})


# ---------------------------------------------------------------------------
# session.list
# ---------------------------------------------------------------------------


def test_parse_list_empty():
    msg = parse_list({"type": "session.list"})
    assert msg.cwd is None


def test_parse_list_with_cwd():
    msg = parse_list({"type": "session.list", "id": "r1", "cwd": "/proj"})
    assert msg.cwd == "/proj" and msg.id == "r1"


def test_parse_list_rejects_bad_cwd():
    with pytest.raises(ProtocolError):
        parse_list({"type": "session.list", "cwd": 42})


def test_parse_list_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_list({"type": "session.list", "cwd": "/tmp", "extra": 1})


# ---------------------------------------------------------------------------
# error_frame
# ---------------------------------------------------------------------------


def test_error_frame_includes_optional_fields():
    frame = error_frame("invalid_message", "oops", id="req_1", session_id="s1")
    assert frame["type"] == "error"
    assert frame["code"] == "invalid_message"
    assert frame["id"] == "req_1"
    assert frame["session_id"] == "s1"


def test_error_frame_omits_unset_ids():
    frame = error_frame("internal", "bad")
    assert "id" not in frame and "session_id" not in frame


# ---------------------------------------------------------------------------
# ping / status / watch / unwatch / session.info
# ---------------------------------------------------------------------------


def test_parse_ping_no_data():
    assert parse_ping({"type": "ping"}).id is None


def test_parse_ping_with_id_and_data():
    msg = parse_ping({"type": "ping", "id": "p1", "data": {"x": 1}})
    assert msg.id == "p1" and msg.data == {"x": 1}


def test_parse_ping_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_ping({"type": "ping", "bogus": True})


def test_parse_status_ok():
    assert parse_status({"type": "status", "id": "s1"}).id == "s1"


def test_parse_status_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_status({"type": "status", "extra": 1})


def test_parse_attach_role_and_default():
    assert parse_attach({"type": "session.attach", "session_id": "s1"}).role == "viewer"
    m = parse_attach({"type": "session.attach", "session_id": "s1", "as": "owner"})
    assert m.role == "owner"


def test_parse_attach_rejects_bad_role():
    with pytest.raises(ProtocolError):
        parse_attach({"type": "session.attach", "session_id": "s1", "as": "boss"})


def test_parse_attach_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_attach({"type": "session.attach", "session_id": "s1", "extra": 1})


def test_parse_detach_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_detach({"type": "session.detach", "session_id": "s1", "extra": 1})


def test_parse_session_info_rejects_extra_keys():
    with pytest.raises(ProtocolError, match="unexpected field"):
        parse_session_info({"type": "session.info", "session_id": "s1", "extra": 1})


# ---------------------------------------------------------------------------
# Profile / agent name validation (#54)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["ok", "claude-sonnet", "A_1", "a" * 64])
def test_parse_profile_mutate_accepts_valid_names(name):
    msg = parse_profile_mutate(
        {"type": "profile.create", "name": name, "profile": {"agent": {"agent_command": "x"}}}
    )
    assert msg.name == name


@pytest.mark.parametrize("name", ["my.profile", "a b", "naïve", "a/b", "a" * 65])
def test_parse_profile_mutate_rejects_invalid_names(name):
    with pytest.raises(ProtocolError, match="invalid profile name"):
        parse_profile_mutate(
            {"type": "profile.create", "name": name, "profile": {"agent": {"agent_command": "x"}}}
        )


def test_parse_profile_mutate_rejects_invalid_agent_names():
    with pytest.raises(ProtocolError, match="invalid agent name"):
        parse_profile_mutate(
            {
                "type": "profile.create",
                "name": "ok",
                "profile": {"agents": {"bad.agent": {"agent_command": "x"}}},
            }
        )


@pytest.mark.parametrize("verb", ["profile.start", "profile.stop", "profile.delete"])
def test_parse_profile_action_allows_legacy_names(verb):
    # All action verbs target *existing* registry entries, which may predate
    # name validation (#54) — a legacy-named profile must stay stoppable and
    # deletable. Unknown names fail with profile_unknown at the registry
    # lookup, so permissive parsing is safe.
    msg = parse_profile_action({"type": verb, "name": "my.profile"})
    assert msg.name == "my.profile"


def test_parse_profile_mutate_rejects_explicit_empty_top_level_name():
    # An explicit empty "name" is a client bug — no silent fallback to
    # profile.name (#55 review).
    with pytest.raises(ProtocolError, match="non-empty 'name'"):
        parse_profile_mutate(
            {
                "type": "profile.create",
                "name": "",
                "profile": {"name": "ok", "agent": {"agent_command": "x"}},
            }
        )


def test_parse_profile_mutate_absent_name_falls_back_to_profile_name():
    msg = parse_profile_mutate(
        {"type": "profile.create", "profile": {"name": "ok", "agent": {"agent_command": "x"}}}
    )
    assert msg.name == "ok"
