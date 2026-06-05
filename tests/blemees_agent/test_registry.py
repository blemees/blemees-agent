"""Unit tests for the persistent session registry (#21)."""

from __future__ import annotations

from blemees_agent.registry import Registry


def test_upsert_sets_timestamps_and_fields():
    reg = Registry(None)
    rec = reg.upsert("s1", profile="work", agent="claude", cwd="/p", model="sonnet")
    assert rec["session_id"] == "s1"
    assert rec["profile"] == "work" and rec["agent"] == "claude"
    assert rec["created_at_ms"] > 0
    assert rec["last_active_at_ms"] >= rec["created_at_ms"]


def test_upsert_merges_and_keeps_created_at():
    reg = Registry(None)
    first = reg.upsert("s1", profile="work")
    created = first["created_at_ms"]
    reg.upsert("s1", model="opus")
    rec = reg.get("s1")
    assert rec["profile"] == "work" and rec["model"] == "opus"
    assert rec["created_at_ms"] == created


def test_touch_updates_turns_and_activity():
    reg = Registry(None)
    reg.upsert("s1", profile="work")
    reg.touch("s1", turns=3, model="haiku")
    rec = reg.get("s1")
    assert rec["turns"] == 3 and rec["model"] == "haiku"


def test_remove():
    reg = Registry(None)
    reg.upsert("s1")
    removed = reg.remove("s1")
    assert removed is True
    assert reg.get("s1") is None
    removed_again = reg.remove("s1")
    assert removed_again is False


def test_in_memory_when_no_path():
    reg = Registry(None)
    assert reg.persistent is False
    reg.upsert("s1", profile="p")
    reg.save()  # no-op, must not raise
    reg.load()  # no-op


def test_persist_round_trip(tmp_path):
    path = tmp_path / "registry.json"
    reg = Registry(path)
    reg.upsert("s1", profile="work", agent="claude", cwd="/p", model="sonnet", view_only=False)
    reg.touch("s1", turns=2)
    reg.save()
    assert path.is_file()

    # A fresh registry over the same file recovers the record.
    reg2 = Registry(path)
    reg2.load()
    rec = reg2.get("s1")
    assert rec is not None
    assert rec["profile"] == "work"
    assert rec["agent"] == "claude"
    assert rec["turns"] == 2


def test_save_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "registry.json"
    reg = Registry(path)
    reg.upsert("s1", profile="p")
    reg.save()
    assert not (tmp_path / "registry.json.tmp").exists()
    assert list(tmp_path.glob("*.json")) == [path]


def test_load_tolerates_missing_and_corrupt(tmp_path):
    path = tmp_path / "registry.json"
    Registry(path).load()  # missing file → no error
    path.write_text("{ not json", encoding="utf-8")
    reg = Registry(path)
    reg.load()  # corrupt → ignored
    assert reg.all() == []
