"""Unit tests for over-wire profile CRUD on the Supervisor (#25)."""

from __future__ import annotations

import sys

import pytest

from blemees_agent.config import Config
from blemees_agent.errors import (
    AgentUnavailableError,
    ProfileExistsError,
    ProfileProtectedError,
    ProfileUnknownError,
)
from blemees_agent.logging import configure
from blemees_agent.registry import Registry
from blemees_agent.supervisor import Supervisor

LOG = configure("error")
GOOD = sys.executable  # always resolvable on PATH
MISSING = "definitely-not-a-real-binary-xyz123"


def _sup(tmp_path=None, **profiles) -> tuple[Supervisor, Registry]:
    reg = Registry(tmp_path / "registry.json" if tmp_path else None)
    cfg = Config(socket_path="/tmp/x.sock", agent_command=GOOD, profiles=profiles)
    return Supervisor(cfg, LOG, registry=reg), reg


pytestmark = pytest.mark.asyncio


async def test_create_profile_adds_dynamic_and_persists(tmp_path):
    sup, reg = _sup(tmp_path)
    sup.create_profile("mine", {"agent_command": GOOD, "model": "sonnet"})
    # Resolvable + listed as dynamic.
    profile = sup.get_profile("mine")
    assert profile.default_agent.command == GOOD
    row = next(r for r in sup.profile_list() if r["name"] == "mine")
    assert row["source"] == "dynamic"
    # Persisted to the registry file.
    assert reg.get_profile_spec("mine")["model"] == "sonnet"


async def test_create_profile_collision_raises(tmp_path):
    sup, _ = _sup(tmp_path, existing={"agent_command": GOOD})
    with pytest.raises(ProfileExistsError):
        sup.create_profile("existing", {"agent_command": GOOD})
    with pytest.raises(ProfileExistsError):
        sup.create_profile("default", {"agent_command": GOOD})  # synthesised name


async def test_create_profile_missing_binary_raises(tmp_path):
    sup, reg = _sup(tmp_path)
    with pytest.raises(AgentUnavailableError):
        sup.create_profile("bad", {"agent_command": MISSING})
    assert reg.get_profile_spec("bad") is None  # not persisted on failure


async def test_create_multi_agent_profile(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.create_profile(
        "multi",
        {"agents": {"a": {"agent_command": GOOD}, "b": {"agent_command": GOOD}}},
    )
    assert set(sup.get_profile("multi").agents) == {"a", "b"}


async def test_create_with_single_agent_object(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.create_profile("solo", {"agent": {"agent_command": GOOD, "model": "opus"}})
    profile = sup.get_profile("solo")
    assert profile.default_agent.model == "opus"


async def test_update_dynamic_profile(tmp_path):
    sup, reg = _sup(tmp_path)
    sup.create_profile("mine", {"agent_command": GOOD, "model": "sonnet"})
    await sup.update_profile("mine", {"agent_command": GOOD, "model": "opus"})
    assert sup.get_profile("mine").default_agent.model == "opus"
    assert reg.get_profile_spec("mine")["model"] == "opus"


async def test_update_unknown_raises(tmp_path):
    sup, _ = _sup(tmp_path)
    with pytest.raises(ProfileUnknownError):
        await sup.update_profile("nope", {"agent_command": GOOD})


async def test_update_config_profile_is_protected(tmp_path):
    sup, _ = _sup(tmp_path, fromcfg={"agent_command": GOOD})
    with pytest.raises(ProfileProtectedError):
        await sup.update_profile("fromcfg", {"agent_command": GOOD})
    with pytest.raises(ProfileProtectedError):
        await sup.update_profile("default", {"agent_command": GOOD})


async def test_delete_dynamic_profile(tmp_path):
    sup, reg = _sup(tmp_path)
    sup.create_profile("mine", {"agent_command": GOOD})
    await sup.delete_profile("mine")
    with pytest.raises(ProfileUnknownError):
        sup.get_profile("mine")
    assert reg.get_profile_spec("mine") is None


async def test_delete_config_profile_is_protected(tmp_path):
    sup, _ = _sup(tmp_path, fromcfg={"agent_command": GOOD})
    with pytest.raises(ProfileProtectedError):
        await sup.delete_profile("fromcfg")


async def test_load_persisted_rehydrates_dynamic_profiles(tmp_path):
    # First supervisor creates a profile (persisted to the registry file).
    sup1, reg1 = _sup(tmp_path)
    sup1.create_profile("mine", {"agent_command": GOOD, "model": "sonnet"})

    # A fresh supervisor over the same registry adopts it after load().
    reg2 = Registry(tmp_path / "registry.json")
    reg2.load()
    sup2 = Supervisor(Config(socket_path="/tmp/x.sock", agent_command=GOOD), LOG, registry=reg2)
    sup2.load_persisted()
    assert sup2.get_profile("mine").default_agent.model == "sonnet"
    assert next(r for r in sup2.profile_list() if r["name"] == "mine")["source"] == "dynamic"


async def test_load_persisted_skips_config_name_collision(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.upsert_profile("clash", {"agent_command": GOOD, "model": "persisted"})
    reg.save()
    # Config also defines "clash" → config wins, persisted is shadowed.
    cfg = Config(
        socket_path="/tmp/x.sock",
        agent_command=GOOD,
        profiles={"clash": {"agent_command": GOOD, "model": "fromconfig"}},
    )
    sup = Supervisor(cfg, LOG, registry=reg)
    sup.load_persisted()
    assert sup.get_profile("clash").default_agent.model == "fromconfig"


async def test_start_missing_binary_raises_agent_unavailable():
    # A profile whose binary isn't on PATH fails start() with agent_unavailable
    # rather than a spawn crash partway through.
    cfg = Config(socket_path="/tmp/x.sock", agent_command=MISSING)
    sup = Supervisor(cfg, LOG)
    with pytest.raises(AgentUnavailableError):
        await sup.start("default")
