"""Wire protocol codec for blemees-agentd (blemees/3).

Responsibilities:
    * Encode/decode newline-delimited JSON frames.
    * Validate ``blemees/3`` control messages into typed dataclasses.

blemees/3 is a clean break from the ``agent.*`` vocabulary: the daemon is an
ACP supervisor/proxy, so the conversation flows in ``session.*`` frames that
carry ACP payloads (see ``backends/acp.py`` and ``docs/acp-migration-spec.md``).
Inbound control frames reject unknown keys.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from . import PROTOCOL_VERSION
from .errors import OversizeMessageError, ProtocolError

DEFAULT_MAX_LINE_BYTES = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Outbound helpers (daemon → client).
# ---------------------------------------------------------------------------


def encode(obj: dict[str, Any]) -> bytes:
    """Encode a message as a single UTF-8 JSON line (with trailing ``\\n``)."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def hello_ack(
    daemon_version: str, pid: int, agents: dict[str, str], profiles: list[str]
) -> dict[str, Any]:
    return {
        "type": "hello_ack",
        "daemon": f"blemees-agentd/{daemon_version}",
        "protocol": PROTOCOL_VERSION,
        "pid": pid,
        # Detected ACP agent binaries (name → version-ish), best-effort.
        "agents": dict(agents),
        # Configured profile names (#17).
        "profiles": list(profiles),
    }


def error_frame(
    code: str,
    message: str,
    *,
    id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if id is not None:
        frame["id"] = id
    if session_id is not None:
        frame["session_id"] = session_id
    return frame


# ---------------------------------------------------------------------------
# Inbound parsing.
# ---------------------------------------------------------------------------


def parse_line(line: bytes, *, max_bytes: int = DEFAULT_MAX_LINE_BYTES) -> dict[str, Any]:
    """Parse a single wire line; raises :class:`ProtocolError` on bad input."""
    if len(line) > max_bytes:
        raise OversizeMessageError(max_bytes)
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - trivial
        raise ProtocolError(f"invalid utf-8: {exc}") from exc
    text = text.rstrip("\r\n")
    if not text:
        raise ProtocolError("empty frame")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid json: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("frame must be a JSON object")
    if "type" not in obj or not isinstance(obj["type"], str):
        raise ProtocolError("missing string 'type'")
    return obj


def _reject_extra_keys(obj: dict[str, Any], allowed: frozenset[str]) -> None:
    """Raise :class:`ProtocolError` when *obj* contains keys not in *allowed*."""
    extra = obj.keys() - allowed
    if extra:
        field = next(iter(sorted(extra)))
        raise ProtocolError(f"unexpected field: {field!r}")


def _opt_str_id(obj: dict[str, Any]) -> str | None:
    req_id = obj.get("id")
    if req_id is not None and not isinstance(req_id, str):
        raise ProtocolError("'id' must be a string")
    return req_id


def _require_session_id(obj: dict[str, Any], verb: str) -> str:
    session_id = obj.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError(f"{verb} requires non-empty 'session_id'")
    return session_id


# ---------------------------------------------------------------------------
# Typed control-message dataclasses.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class HelloMessage:
    client: str | None
    protocol: str


@dataclasses.dataclass(slots=True)
class OpenMessage:
    id: str | None
    session_id: str
    options: dict[str, Any]
    resume: bool
    profile: str | None = None
    agent: str | None = None
    last_seen_seq: int | None = None
    alias: str | None = None


@dataclasses.dataclass(slots=True)
class ProfileListMessage:
    id: str | None


@dataclasses.dataclass(slots=True)
class ProfileActionMessage:
    id: str | None
    name: str


@dataclasses.dataclass(slots=True)
class PromptMessage:
    session_id: str
    # The user turn: a blemees message envelope ``{"role":"user","content":...}``
    # where content is a string or an array of ACP content blocks.
    message: dict[str, Any]


@dataclasses.dataclass(slots=True)
class CancelMessage:
    session_id: str


@dataclasses.dataclass(slots=True)
class CloseMessage:
    id: str | None
    session_id: str
    delete: bool


@dataclasses.dataclass(slots=True)
class ListMessage:
    id: str | None
    cwd: str | None


@dataclasses.dataclass(slots=True)
class AttachMessage:
    id: str | None
    session_id: str
    role: str  # "owner" | "viewer"
    last_seen_seq: int | None


@dataclasses.dataclass(slots=True)
class DetachMessage:
    id: str | None
    session_id: str


@dataclasses.dataclass(slots=True)
class PermissionResponseMessage:
    session_id: str
    request_id: str
    outcome: str  # "selected" | "cancelled"
    option_id: str | None


@dataclasses.dataclass(slots=True)
class SessionInfoMessage:
    id: str | None
    session_id: str


_MISSING: Any = object()  # sentinel for optional fields not present in the wire frame


@dataclasses.dataclass(slots=True)
class PingMessage:
    id: str | None
    data: Any  # opaque; echoed back on pong; _MISSING means key was absent


@dataclasses.dataclass(slots=True)
class StatusMessage:
    id: str | None


def parse_hello(obj: dict[str, Any]) -> HelloMessage:
    _reject_extra_keys(obj, frozenset({"type", "protocol", "client"}))
    protocol = obj.get("protocol")
    if not isinstance(protocol, str):
        raise ProtocolError("hello missing 'protocol'")
    client = obj.get("client")
    if client is not None and not isinstance(client, str):
        raise ProtocolError("'client' must be a string")
    return HelloMessage(client=client, protocol=protocol)


_OPEN_TOP_LEVEL = frozenset(
    {"type", "id", "session_id", "resume", "last_seen_seq", "options", "alias", "profile", "agent"}
)


def parse_open(obj: dict[str, Any]) -> OpenMessage:
    _reject_extra_keys(obj, _OPEN_TOP_LEVEL)
    session_id = _require_session_id(obj, "session.open")

    options_field = obj.get("options")
    if options_field is None:
        options_field = {}
    if not isinstance(options_field, dict):
        raise ProtocolError("'options' must be an object")

    resume = bool(obj.get("resume", False))

    last_seen_seq = obj.get("last_seen_seq")
    if last_seen_seq is not None and (not isinstance(last_seen_seq, int) or last_seen_seq < 0):
        raise ProtocolError("'last_seen_seq' must be a non-negative integer")

    alias = obj.get("alias")
    if alias is not None and not isinstance(alias, str):
        raise ProtocolError("'alias' must be a string")

    profile = obj.get("profile")
    if profile is not None and (not isinstance(profile, str) or not profile):
        raise ProtocolError("'profile' must be a non-empty string when set")

    agent = obj.get("agent")
    if agent is not None and (not isinstance(agent, str) or not agent):
        raise ProtocolError("'agent' must be a non-empty string when set")

    return OpenMessage(
        id=_opt_str_id(obj),
        session_id=session_id,
        options=options_field,
        resume=resume,
        profile=profile,
        agent=agent,
        last_seen_seq=last_seen_seq,
        alias=alias or None,
    )


def parse_profile_list(obj: dict[str, Any]) -> ProfileListMessage:
    _reject_extra_keys(obj, frozenset({"type", "id"}))
    return ProfileListMessage(id=_opt_str_id(obj))


def parse_profile_action(obj: dict[str, Any]) -> ProfileActionMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "name"}))
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError("profile action requires non-empty 'name'")
    return ProfileActionMessage(id=_opt_str_id(obj), name=name)


def parse_prompt(obj: dict[str, Any]) -> PromptMessage:
    _reject_extra_keys(obj, frozenset({"type", "session_id", "prompt"}))
    session_id = _require_session_id(obj, "session.prompt")
    prompt = obj.get("prompt")
    if not isinstance(prompt, (str, list)):
        raise ProtocolError("'prompt' must be a string or array of content blocks")
    # Normalise to the backend's message envelope.
    return PromptMessage(session_id=session_id, message={"role": "user", "content": prompt})


def parse_cancel(obj: dict[str, Any]) -> CancelMessage:
    _reject_extra_keys(obj, frozenset({"type", "session_id"}))
    return CancelMessage(session_id=_require_session_id(obj, "session.cancel"))


def parse_close(obj: dict[str, Any]) -> CloseMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "session_id", "delete"}))
    session_id = _require_session_id(obj, "session.close")
    return CloseMessage(
        id=_opt_str_id(obj), session_id=session_id, delete=bool(obj.get("delete", False))
    )


def parse_list(obj: dict[str, Any]) -> ListMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "cwd"}))
    cwd_field = obj.get("cwd")
    if cwd_field is not None and (not isinstance(cwd_field, str) or not cwd_field):
        raise ProtocolError("'cwd' must be a non-empty string when set")
    return ListMessage(id=_opt_str_id(obj), cwd=cwd_field)


def parse_ping(obj: dict[str, Any]) -> PingMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "data"}))
    return PingMessage(id=_opt_str_id(obj), data=obj.get("data", _MISSING))


def parse_status(obj: dict[str, Any]) -> StatusMessage:
    _reject_extra_keys(obj, frozenset({"type", "id"}))
    return StatusMessage(id=_opt_str_id(obj))


def parse_attach(obj: dict[str, Any]) -> AttachMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "session_id", "as", "last_seen_seq"}))
    session_id = _require_session_id(obj, "session.attach")
    role = obj.get("as", "viewer")
    if role not in ("owner", "viewer"):
        raise ProtocolError("'as' must be 'owner' or 'viewer'")
    last_seen_seq = obj.get("last_seen_seq")
    if last_seen_seq is not None and (not isinstance(last_seen_seq, int) or last_seen_seq < 0):
        raise ProtocolError("'last_seen_seq' must be a non-negative integer")
    return AttachMessage(
        id=_opt_str_id(obj), session_id=session_id, role=role, last_seen_seq=last_seen_seq
    )


def parse_detach(obj: dict[str, Any]) -> DetachMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "session_id"}))
    return DetachMessage(id=_opt_str_id(obj), session_id=_require_session_id(obj, "session.detach"))


def parse_permission_response(obj: dict[str, Any]) -> PermissionResponseMessage:
    _reject_extra_keys(obj, frozenset({"type", "session_id", "request_id", "outcome", "option_id"}))
    session_id = _require_session_id(obj, "session.permission_response")
    request_id = obj.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("session.permission_response requires 'request_id'")
    outcome = obj.get("outcome")
    if outcome not in ("selected", "cancelled"):
        raise ProtocolError("'outcome' must be 'selected' or 'cancelled'")
    option_id = obj.get("option_id")
    if option_id is not None and not isinstance(option_id, str):
        raise ProtocolError("'option_id' must be a string")
    if outcome == "selected" and not option_id:
        raise ProtocolError("'option_id' is required when outcome is 'selected'")
    return PermissionResponseMessage(
        session_id=session_id, request_id=request_id, outcome=outcome, option_id=option_id
    )


def parse_session_info(obj: dict[str, Any]) -> SessionInfoMessage:
    _reject_extra_keys(obj, frozenset({"type", "id", "session_id"}))
    return SessionInfoMessage(
        id=_opt_str_id(obj), session_id=_require_session_id(obj, "session.info")
    )
