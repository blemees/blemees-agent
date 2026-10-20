"""Unit tests for over-wire profile CRUD on the Supervisor (#25)."""

from __future__ import annotations

import copy
import sys

import pytest

from blemees_agent.config import Config
from blemees_agent.errors import (
    AgentUnavailableError,
    ProfileExistsError,
    ProfileProtectedError,
    ProfileUnknownError,
    ProtocolError,
)
from blemees_agent.logging import configure
from blemees_agent.registry import Registry
from blemees_agent.supervisor import REDACTED, Supervisor

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


# ---- profile_spec / profile.get (#43) --------------------------------

_RICH = {
    "agent": {
        "agent_command": GOOD,
        "args": ["--flag"],
        "model": "opus",
        "mode": "plan",
        "agent_home": "/tmp/home",
        "mcp_servers": [{"name": "fs", "command": "mcp-fs", "env": {"FS_TOKEN": "s3cret"}}],
        "env": {"API_KEY": "sk-live", "REGION": "us"},
    },
    "permission_policy": {"mode": "deny", "detached": "allow"},
    "notify": {"webhook_url": "https://ntfy.sh/private-topic", "format": "ntfy"},
    "attention": {"triggers": ["turn_complete", "permission_pending"]},
}


async def test_profile_spec_returns_full_spec_with_secrets_redacted(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.create_profile("rich", _RICH)
    spec = sup.profile_spec("rich")
    assert spec == {
        "name": "rich",
        "agents": {
            "default": {
                "command": GOOD,
                "args": ["--flag"],
                "model": "opus",
                "mode": "plan",
                "agent_home": "/tmp/home",
                "mcp_servers": [{"name": "fs", "command": "mcp-fs", "env": {"FS_TOKEN": REDACTED}}],
                "env": {"API_KEY": REDACTED, "REGION": REDACTED},
            }
        },
        "permission_policy": {"mode": "deny", "detached": "allow"},
        "notify": {"webhook_url": REDACTED, "format": "ntfy"},
        "attention": {"triggers": ["permission_pending", "turn_complete"]},
    }
    # Redaction is a view: the live profile still holds the real secrets.
    agent = sup.get_profile("rich").default_agent
    assert agent.env["API_KEY"] == "sk-live"
    assert agent.mcp_servers[0]["env"]["FS_TOKEN"] == "s3cret"
    assert "s3cret" not in repr(spec) and "sk-live" not in repr(spec)
    assert "private-topic" not in repr(spec)


async def test_profile_spec_config_managed_multi_agent(tmp_path):
    sup, _ = _sup(
        tmp_path,
        work={"agents": {"a": {"agent_command": GOOD}, "b": {"agent_command": GOOD, "model": "x"}}},
    )
    spec = sup.profile_spec("work")
    assert set(spec["agents"]) == {"a", "b"}
    assert spec["agents"]["a"] == {"command": GOOD, "args": [], "mcp_servers": [], "env": {}}
    assert spec["agents"]["b"]["model"] == "x"
    assert spec["permission_policy"] == {"mode": "relay", "detached": "stall"}
    assert spec["notify"] == {}
    assert sup.is_config_managed("work") and sup.is_config_managed("default")
    # The synthesised default profile is gettable too.
    assert sup.profile_spec("default")["agents"]["default"]["command"] == GOOD


async def test_profile_spec_unknown_raises(tmp_path):
    sup, _ = _sup(tmp_path)
    with pytest.raises(ProfileUnknownError):
        sup.profile_spec("nope")


async def test_update_with_round_tripped_spec_keeps_secrets(tmp_path):
    # An editor sends profile.info's spec straight back (placeholders and
    # all), changing only the model. Stored secrets must survive, both in
    # memory and in the persisted registry spec.
    sup, reg = _sup(tmp_path)
    sup.create_profile("rich", _RICH)
    spec = sup.profile_spec("rich")
    spec["agents"]["default"]["model"] = "sonnet"
    await sup.update_profile("rich", spec)

    profile = sup.get_profile("rich")
    agent = profile.default_agent
    assert agent.model == "sonnet"
    assert agent.args == ["--flag"] and agent.mode == "plan"
    assert agent.env == {"API_KEY": "sk-live", "REGION": "us"}
    assert agent.mcp_servers[0]["env"] == {"FS_TOKEN": "s3cret"}
    assert profile.notify == {"webhook_url": "https://ntfy.sh/private-topic", "format": "ntfy"}
    assert profile.permission_policy == {"mode": "deny", "detached": "allow"}
    assert profile.attention_triggers == {"permission_pending", "turn_complete"}
    persisted = reg.get_profile_spec("rich")
    assert REDACTED not in repr(persisted)
    assert persisted["agents"]["default"]["env"]["API_KEY"] == "sk-live"
    # Round-tripping again yields the same redacted view (idempotent).
    assert sup.profile_spec("rich")["agents"]["default"]["model"] == "sonnet"


async def test_update_replaces_and_drops_secrets_explicitly(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.create_profile("rich", _RICH)
    spec = sup.profile_spec("rich")
    env = spec["agents"]["default"]["env"]
    env["API_KEY"] = "sk-new"  # explicit new value wins
    del env["REGION"]  # removed key is gone
    env["ADDED"] = REDACTED  # placeholder with nothing stored → dropped
    spec["notify"].pop("webhook_url")
    await sup.update_profile("rich", spec)
    profile = sup.get_profile("rich")
    assert profile.default_agent.env == {"API_KEY": "sk-new"}
    assert profile.notify == {"format": "ntfy"}


async def test_create_drops_redacted_placeholders(tmp_path):
    # Duplicating a profile from profile.info: placeholders have no stored
    # value to resolve to, so they're dropped instead of stored literally.
    sup, reg = _sup(tmp_path)
    sup.create_profile("rich", _RICH)
    sup.create_profile("copy", sup.profile_spec("rich"))
    copied = sup.get_profile("copy")
    assert copied.default_agent.env == {}
    assert copied.default_agent.mcp_servers[0]["env"] == {}
    assert copied.notify == {"format": "ntfy"}
    assert REDACTED not in repr(reg.get_profile_spec("copy"))


# ---- MCP restore matching: exact identity, else unique name, else refuse ---
#
# A "<redacted>" placeholder in mcp_servers[i] resolves only against a stored
# server we can identify for certain (#43 review rounds 1-2): the unique
# server with the same non-secret fields, else the unique server with the
# same `name`. Anything else refuses the whole update with invalid_message.

GH = {"command": "npx", "args": ["gh-mcp"], "env": {"TOKEN": "github-secret"}}
SLACK = {"command": "npx", "args": ["slack-mcp"], "env": {"TOKEN": "slack-secret"}}


def _servers(sup, name="m"):
    return sup.get_profile(name).default_agent.mcp_servers


def _stored_servers(reg, name="m"):
    return reg.get_profile_spec(name)["agents"]["default"]["mcp_servers"]


def _with_servers(sup, servers, name="m"):
    sup.create_profile(name, {"agent": {"agent_command": GOOD, "mcp_servers": servers}})
    spec = sup.profile_spec(name)
    return spec, spec["agents"]["default"]["mcp_servers"]


async def _assert_refused(sup, reg, spec, match, name="m"):
    before_mem = copy.deepcopy(_servers(sup, name))
    before_reg = copy.deepcopy(reg.get_profile_spec(name))
    with pytest.raises(ProtocolError, match=match) as exc:
        await sup.update_profile(name, spec)
    assert exc.value.code == "invalid_message"
    assert "re-enter the value" in exc.value.message
    # No partial write, in memory or in the registry.
    assert _servers(sup, name) == before_mem
    assert reg.get_profile_spec(name) == before_reg


async def test_round_trip_unnamed_mcp_servers_same_command_keep_own_secrets(tmp_path):
    # Two unnamed servers share the command "npx"; unchanged, each must get
    # its own secret back (matched by exact identity: command + args).
    sup, reg = _sup(tmp_path)
    spec, _ = _with_servers(sup, [GH, SLACK])
    await sup.update_profile("m", spec)
    assert _servers(sup) == [GH, SLACK]
    assert _stored_servers(reg) == [GH, SLACK]


async def test_mcp_delete_first_server_keeps_remaining_secret(tmp_path):
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [GH, SLACK])
    del servers[0]
    await sup.update_profile("m", spec)
    assert _servers(sup) == [SLACK]
    assert _stored_servers(reg) == [SLACK]


async def test_mcp_swap_servers_keeps_own_secrets(tmp_path):
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [GH, SLACK])
    servers.reverse()
    await sup.update_profile("m", spec)
    assert _servers(sup) == [SLACK, GH]
    assert _stored_servers(reg) == [SLACK, GH]


async def test_mcp_insert_server_with_unresolvable_placeholder_refused(tmp_path):
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [GH, SLACK])
    servers.insert(0, {"command": "npx", "args": ["new-mcp"], "env": {"TOKEN": REDACTED}})
    await _assert_refused(sup, reg, spec, r"agents\.default\.mcp_servers\[0\]\.env\.TOKEN")


async def test_mcp_edit_args_of_unnamed_server_with_placeholder_refused(tmp_path):
    # Command alone is not an identity: new args may mean a different server.
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [GH, SLACK])
    servers[1]["args"] = ["slack-mcp", "--v2"]
    await _assert_refused(sup, reg, spec, r"mcp_servers\[1\]\.env\.TOKEN")


async def test_mcp_edit_args_of_unnamed_server_with_real_value_accepted(tmp_path):
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [GH, SLACK])
    servers[1]["args"] = ["slack-mcp", "--v2"]
    servers[1]["env"]["TOKEN"] = "new-slack-secret"
    await sup.update_profile("m", spec)
    expected = [
        GH,
        {"command": "npx", "args": ["slack-mcp", "--v2"], "env": {"TOKEN": "new-slack-secret"}},
    ]
    assert _servers(sup) == expected
    assert _stored_servers(reg) == expected


async def test_named_mcp_server_args_edit_keeps_secret(tmp_path):
    # A unique name with the same command is an explicit identity: it
    # survives an args edit.
    sup, reg = _sup(tmp_path)
    fs = {"name": "fs", "command": "mcp-fs", "args": ["/a"], "env": {"K": "fs-secret"}}
    spec, servers = _with_servers(sup, [fs, GH])
    servers[0]["args"] = ["/b"]
    await sup.update_profile("m", spec)
    expected = [{**fs, "args": ["/b"]}, GH]
    assert _servers(sup) == expected
    assert _stored_servers(reg) == expected


async def test_named_mcp_server_command_or_url_change_with_placeholder_refused(tmp_path):
    # The name fallback also needs the same endpoint (command, and url when
    # either side has one); a new endpoint must not inherit the secret.
    sup, reg = _sup(tmp_path)
    fs = {"name": "fs", "command": "mcp-fs", "args": ["/a"], "env": {"K": "fs-secret"}}
    spec, servers = _with_servers(sup, [fs])
    servers[0]["command"] = "mcp-fs2"
    await _assert_refused(sup, reg, spec, r"mcp_servers\[0\]\.env\.K")

    remote = {"name": "r", "url": "https://a.example", "headers": {"Authorization": "B s"}}
    spec, servers = _with_servers(sup, [remote], name="u")
    servers[0]["url"] = "https://evil.example"
    await _assert_refused(sup, reg, spec, r"mcp_servers\[0\]\.headers\.Authorization", name="u")


async def test_delete_and_rename_onto_deleted_name_refused(tmp_path):
    # Review round 3: delete server y, rename server x to "y" keeping its
    # placeholder. The name now matches y, but the command is x's, so y's
    # secret must not move to x's process.
    sup, reg = _sup(tmp_path)
    x = {"name": "x", "command": "a", "env": {"T": "x-secret"}}
    y = {"name": "y", "command": "b", "env": {"T": "y-secret"}}
    spec, servers = _with_servers(sup, [x, y])
    del servers[1]
    servers[0]["name"] = "y"
    await _assert_refused(sup, reg, spec, r"agents\.default\.mcp_servers\[0\]\.env\.T")


async def test_same_named_servers_different_command_delete_and_reorder(tmp_path):
    a = {"name": "fs", "command": "mcp-a", "env": {"K": "secret-a"}}
    b = {"name": "fs", "command": "mcp-b", "env": {"K": "secret-b"}}
    # Reorder: exact identity (name + command) still tells them apart.
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [a, b])
    servers.reverse()
    await sup.update_profile("m", spec)
    assert _servers(sup) == [b, a] and _stored_servers(reg) == [b, a]
    # Delete one: the other keeps its own secret.
    spec = sup.profile_spec("m")
    del spec["agents"]["default"]["mcp_servers"][0]
    await sup.update_profile("m", spec)
    assert _servers(sup) == [a] and _stored_servers(reg) == [a]


async def test_same_named_servers_ambiguous_edit_refused(tmp_path):
    # The edited server matches no identity and its name is not unique.
    a = {"name": "fs", "command": "mcp-a", "env": {"K": "secret-a"}}
    b = {"name": "fs", "command": "mcp-b", "env": {"K": "secret-b"}}
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [a, b])
    servers[1]["command"] = "mcp-c"
    await _assert_refused(sup, reg, spec, r"mcp_servers\[1\]\.env\.K")


async def test_identical_servers_differing_only_in_env_refused(tmp_path):
    # Known limit: nothing but the (redacted) secret tells these apart, so
    # even an unchanged round trip is refused rather than guessed.
    a = {"name": "fs", "command": "mcp-fs", "env": {"A": "secret-a"}}
    b = {"name": "fs", "command": "mcp-fs", "env": {"B": "secret-b"}}
    sup, reg = _sup(tmp_path)
    spec, _ = _with_servers(sup, [a, b])
    await _assert_refused(sup, reg, spec, r"mcp_servers\[0\]\.env\.A")
    # Re-entering the values resolves it.
    spec["agents"]["default"]["mcp_servers"][0]["env"]["A"] = "secret-a"
    spec["agents"]["default"]["mcp_servers"][1]["env"]["B"] = "secret-b"
    await sup.update_profile("m", spec)
    assert _servers(sup) == [a, b] and _stored_servers(reg) == [a, b]


async def test_servers_without_placeholders_never_refused(tmp_path):
    # Duplicates are fine when nothing needs resolving.
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [GH, SLACK])
    servers[:] = [
        {"command": "npx", "args": ["x"], "env": {"T": "v1"}},
        {"command": "npx", "args": ["x"], "env": {"T": "v2"}},
        {"command": "npx", "args": ["y"]},
    ]
    await sup.update_profile("m", spec)
    assert _servers(sup) == servers
    assert _stored_servers(reg) == servers


async def test_placeholder_under_new_agent_refused_on_update(tmp_path):
    # No stored server can match under an agent name that didn't exist.
    sup, reg = _sup(tmp_path)
    spec, _ = _with_servers(sup, [GH])
    spec["agents"]["other"] = copy.deepcopy(spec["agents"]["default"])
    await _assert_refused(sup, reg, spec, r"agents\.other\.mcp_servers\[0\]\.env\.TOKEN")


# ---- MCP headers + list-form env redaction (#43 review) ----------------


_HTTP_AND_LIST = [
    {
        "name": "remote",
        "type": "http",
        "url": "https://mcp.example/sse?team=1",
        "headers": {"Authorization": "Bearer hdr-secret"},
    },
    {
        "name": "remote2",
        "url": "https://mcp2.example",
        "headers": [{"name": "X-Key", "value": "list-hdr-secret"}],
    },
    {
        "name": "stdio",
        "command": "mcp-x",
        "args": ["--verbose"],
        "env": [
            {"name": "TOK", "value": "list-env-1"},
            {"name": "OTHER", "value": "list-env-3", "extra": "extra-secret"},
        ],
    },
]


async def test_profile_spec_redacts_mcp_headers_and_list_env(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.create_profile("h", {"agent": {"agent_command": GOOD, "mcp_servers": _HTTP_AND_LIST}})
    servers = sup.profile_spec("h")["agents"]["default"]["mcp_servers"]
    assert servers == [
        {
            "name": "remote",
            "type": "http",
            "url": "https://mcp.example/sse?team=1",  # url is returned as-is
            "headers": {"Authorization": REDACTED},
        },
        {
            "name": "remote2",
            "url": "https://mcp2.example",
            "headers": [{"name": "X-Key", "value": REDACTED}],
        },
        {
            "name": "stdio",
            "command": "mcp-x",
            "args": ["--verbose"],  # args are returned as-is
            "env": [
                {"name": "TOK", "value": REDACTED},
                # every key but `name` is redacted, not only `value`
                {"name": "OTHER", "value": REDACTED, "extra": REDACTED},
            ],
        },
    ]
    assert "secret" not in repr(servers) and "list-env" not in repr(servers)


async def test_round_trip_mcp_headers_and_list_env_keep_secrets(tmp_path):
    sup, reg = _sup(tmp_path)
    sup.create_profile("h", {"agent": {"agent_command": GOOD, "mcp_servers": _HTTP_AND_LIST}})
    await sup.update_profile("h", sup.profile_spec("h"))
    assert _servers(sup, "h") == _HTTP_AND_LIST
    assert _stored_servers(reg, "h") == _HTTP_AND_LIST


async def test_list_env_and_headers_replace_remove_and_drop(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.create_profile("h", {"agent": {"agent_command": GOOD, "mcp_servers": _HTTP_AND_LIST}})
    spec = sup.profile_spec("h")
    remote, remote2, stdio = spec["agents"]["default"]["mcp_servers"]
    remote["headers"]["Authorization"] = "Bearer new"  # explicit value replaces
    remote["headers"]["X-New"] = REDACTED  # dict form, nothing stored → dropped
    remote2["headers"] = []  # omitted entries are removed
    stdio["env"][0]["value"] = "replaced-1"  # explicit value replaces
    del stdio["env"][1]["extra"]  # omitted key is removed, value kept
    stdio["env"].reverse()  # entries match by unique name, not position
    await sup.update_profile("h", spec)
    got = _servers(sup, "h")
    assert got[0]["headers"] == {"Authorization": "Bearer new"}
    assert got[1]["headers"] == []
    assert got[2]["env"] == [
        {"name": "OTHER", "value": "list-env-3"},
        {"name": "TOK", "value": "replaced-1"},
    ]


async def test_list_env_duplicate_name_with_placeholder_refused(tmp_path):
    sup, reg = _sup(tmp_path)
    dup = {
        "name": "s",
        "command": "c",
        "env": [{"name": "TOK", "value": "one"}, {"name": "TOK", "value": "two"}],
    }
    spec, _ = _with_servers(sup, [dup])
    await _assert_refused(sup, reg, spec, r"mcp_servers\[0\]\.env\[0\]")


async def test_list_env_new_name_with_placeholder_refused(tmp_path):
    sup, reg = _sup(tmp_path)
    spec, servers = _with_servers(sup, [_HTTP_AND_LIST[2]])
    servers[0]["env"].append({"name": "NEW", "value": REDACTED})
    await _assert_refused(sup, reg, spec, r"mcp_servers\[0\]\.env\[2\]")


async def test_unrecognised_secret_field_shape_redacted_whole(tmp_path):
    # An env/headers shape the daemon doesn't know is redacted whole (fails
    # safe) and a whole-field placeholder restores the stored value.
    sup, _ = _sup(tmp_path)
    odd = [{"name": "odd", "command": "x", "headers": "Authorization: Bearer odd-secret"}]
    sup.create_profile("o", {"agent": {"agent_command": GOOD, "mcp_servers": odd}})
    spec = sup.profile_spec("o")
    assert spec["agents"]["default"]["mcp_servers"][0]["headers"] == REDACTED
    await sup.update_profile("o", spec)
    assert sup.get_profile("o").default_agent.mcp_servers == odd
    # On create there is nothing to restore → the field is dropped.
    sup.create_profile("o2", spec)
    assert "headers" not in sup.get_profile("o2").default_agent.mcp_servers[0]
