"""JSON Schemas for the blemees/3 wire protocol.

> **blemees/3 migration (#16):** the `blemees-agent/1` (`agent.*`) schemas
> were removed in the clean break. The `blemees/3` `session.*` schemas are
> not yet authored — `iter_schemas()` currently yields nothing. Re-authoring
> them is a tracked follow-up; `protocol.py` is the contract in the interim.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib import resources
from typing import Any

__all__ = ["files", "load", "iter_schemas"]


def files() -> resources.abc.Traversable:
    """Return a `Traversable` rooted at this subpackage.

    Use `/` or `joinpath()` to drill into `inbound/` or `outbound/`,
    and `as_file()` if a downstream tool needs an on-disk path.
    """
    return resources.files(__name__)


def load(name: str) -> dict[str, Any]:
    """Load one schema by its package-relative path.

    Example: ``load("inbound/agent.hello.json")``.
    """
    return json.loads((files() / name).read_text(encoding="utf-8"))


def iter_schemas() -> Iterator[dict[str, Any]]:
    """Yield every shipped schema as a parsed dict.

    Order is unspecified. Useful for building a `referencing.Registry`
    that resolves cross-schema `$ref`s without per-file path knowledge.
    """
    root = files()
    for direction in ("inbound", "outbound"):
        sub = root / direction
        if not sub.is_dir():
            continue
        for entry in sub.iterdir():
            if entry.name.endswith(".json"):
                yield json.loads(entry.read_text(encoding="utf-8"))
    common = root / "_common.json"
    if common.is_file():
        yield json.loads(common.read_text(encoding="utf-8"))
