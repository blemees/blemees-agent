# blemees/3 — wire-frame JSON Schemas

Machine-readable contract for every frame on the `blemees/3` wire protocol
(#30). The prose spec is [`docs/acp-migration-spec.md`](../../docs/acp-migration-spec.md)
§9; `blemees_agent/protocol.py` is the Python source of truth. These schemas
formalize the same frame shapes so clients can validate without reading Python.

They ship inside the wheel as the `blemees_agent.schemas` subpackage, so
installed clients validate frames without copying JSON anywhere:

```python
from blemees_agent.schemas import load, iter_schemas, files

opened = load("outbound/session.opened.json")   # parsed dict
all_frames = list(iter_schemas())                # every shipped schema
root = files()                                   # importlib.resources Traversable
```

## Layout

```
blemees_agent/schemas/
  _common.json     # shared $defs (SessionId, RequestId, Seq, ErrorCode, ProfileSpec, Notification, …)
  inbound/         # client → daemon frames (one <type>.json per dispatched verb)
  outbound/        # daemon → client frames (one <type>.json per emitted frame)
```

The file name equals the frame's `type` plus `.json` (e.g. the `session.open`
frame is `inbound/session.open.json`, `hello_ack` is `outbound/hello_ack.json`).
The full set mirrors the daemon's dispatch table and emit sites; the coverage
is pinned by `tests/blemees_agent/test_schemas.py`.

## Draft / compatibility rules

* **Draft**: JSON Schema `2020-12` (via `$schema`).
* **Inbound frames** are strict (`additionalProperties: false`) — the daemon
  rejects unknown keys with `invalid_message`, so the schemas mirror that.
* **Outbound frames** allow `additionalProperties: true` so the daemon can grow
  an envelope (new advisory fields) without breaking conforming clients.
* **`session.update`** wraps a *verbatim* ACP `session/update` payload; its
  inner `update` shape is owned by the `agent-client-protocol` schema and is
  intentionally left open here.

## Use

```python
from blemees_agent.schemas import iter_schemas
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

# Build a registry once so cross-schema $refs (into _common.json) resolve.
store = {s["$id"]: s for s in iter_schemas()}
registry = Registry().with_resources(
    [(uri, Resource.from_contents(schema)) for uri, schema in store.items()]
)

def validate(frame_type: str, frame: dict, direction: str = "inbound") -> None:
    url = f"https://blemees/schemas/{direction}/{frame_type}.json"
    Draft202012Validator(store[url], registry=registry).validate(frame)
```

For tooling that needs on-disk paths, use `importlib.resources.as_file` on
`files() / "inbound" / "session.open.json"`.

## Versioning

Breaking frame-shape changes bump the protocol version (`blemees/3` →
`blemees/4`); the daemon rejects a mismatch on `hello` with
`code: protocol_mismatch`. Additive, backward-compatible changes stay on the
same version. The daemon speaks a single version at a time.
