"""Unit tests for the SessionTable. No claude subprocess involved."""

from __future__ import annotations

import json

import pytest

from blemees_agent.errors import SessionExistsError, SessionUnknownError
from blemees_agent.protocol import OpenMessage
from blemees_agent.session import Session, SessionTable


def _open_msg(session: str = "s1") -> OpenMessage:
    return OpenMessage(id=None, session_id=session, options={}, resume=False)


async def test_register_and_get():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    await table.register(sess)
    assert table.get("s1") is sess


async def test_register_duplicate_raises():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    await table.register(Session(session_id="s1", open_msg=_open_msg(), cwd=None))
    with pytest.raises(SessionExistsError):
        await table.register(Session(session_id="s1", open_msg=_open_msg(), cwd=None))


async def test_get_unknown_raises():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    with pytest.raises(SessionUnknownError):
        table.get("missing")


async def test_detach_marks_idle_time():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None, connection_id=42)
    await table.register(sess)
    await table.detach_soft("s1")
    again = table.get("s1")
    assert again.connection_id is None
    assert again.detached_at is not None


async def test_reap_idle_removes_expired_only():
    table = SessionTable(idle_timeout_s=1.0, max_concurrent=8)
    a = Session(session_id="a", open_msg=_open_msg("a"), cwd=None, connection_id=1)
    b = Session(session_id="b", open_msg=_open_msg("b"), cwd=None, connection_id=2)
    await table.register(a)
    await table.register(b)
    await table.detach_soft("a")
    # Backdate detachment so the reaper considers it expired.
    import time

    table.get("a").detached_at = time.monotonic() - 2.0
    expired = await table.reap_idle()
    assert expired == ["a"]
    assert table.try_get("a") is None
    assert table.try_get("b") is not None  # still attached


async def test_detach_all_for_connection():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    await table.register(
        Session(session_id="a", open_msg=_open_msg("a"), cwd=None, connection_id=1)
    )
    await table.register(
        Session(session_id="b", open_msg=_open_msg("b"), cwd=None, connection_id=1)
    )
    await table.register(
        Session(session_id="c", open_msg=_open_msg("c"), cwd=None, connection_id=2)
    )
    detached = await table.detach_all_for_connection(1)
    assert set(detached) == {"a", "b"}
    assert table.get("a").connection_id is None
    assert table.get("b").connection_id is None
    assert table.get("c").connection_id == 2


async def test_reattach_clears_idle_time():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    await table.register(Session(session_id="s1", open_msg=_open_msg(), cwd=None, connection_id=1))
    await table.detach_soft("s1")

    async def writer(_frame):
        pass

    sess = table.get("s1")
    await sess.attach(connection_id=7, writer=writer)
    assert sess.connection_id == 7
    assert sess.detached_at is None


async def test_max_concurrent_enforced():
    table = SessionTable(idle_timeout_s=60, max_concurrent=1)
    await table.register(Session(session_id="a", open_msg=_open_msg("a"), cwd=None))
    with pytest.raises(SessionExistsError):
        await table.register(Session(session_id="b", open_msg=_open_msg("b"), cwd=None))


async def test_remove_is_idempotent():
    table = SessionTable(idle_timeout_s=60, max_concurrent=8)
    await table.register(Session(session_id="s1", open_msg=_open_msg(), cwd=None))
    await table.remove("s1")
    await table.remove("s1")  # should not raise
    assert table.try_get("s1") is None


def test_context_tokens_from_acp_camelcase_usage():
    # Under blemees/3, session.result.usage is ACP usage dumped by_alias
    # (camelCase). context_tokens must sum the last turn's input side (#66).
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    usage = {
        "inputTokens": 3824,
        "outputTokens": 5,
        "cachedReadTokens": 14000,
        "cachedWriteTokens": 20022,
        "totalTokens": 37851,
    }
    sess._update_usage_from_frame({"type": "session.result", "usage": usage})
    snap = sess.usage_snapshot(attached=True, subprocess_running=True)
    assert snap["context_tokens"] == 3824 + 14000 + 20022
    # last_turn_usage stays verbatim ACP camelCase.
    assert snap["last_turn_usage"] == usage


def test_context_tokens_snake_case_sidecar_fallback(tmp_path):
    # Usage sidecars written before #66 may hold snake_case keys.
    (tmp_path / "s1.usage.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "turns": 2,
                "last_turn_usage": {
                    "input_tokens": 500,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 14000,
                    "cache_creation_input_tokens": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    sess = Session(session_id="s1", open_msg=_open_msg(), cwd=None)
    sess.enable_durable_log(tmp_path)
    try:
        snap = sess.usage_snapshot(attached=False, subprocess_running=False)
    finally:
        sess.log.close()
    assert snap["turns"] == 2
    assert snap["context_tokens"] == 500 + 14000 + 7
