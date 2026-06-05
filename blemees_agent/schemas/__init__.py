"""JSON Schemas for the blemees/3 wire protocol.

One Draft 2020-12 schema per frame under ``inbound/`` and ``outbound/`` (file
name = frame ``type`` + ``.json``), plus shared ``$defs`` in ``_common.json``.
These are the machine-readable contract (#30); ``protocol.py`` is the matching
Python source of truth. Inbound schemas are strict
(``additionalProperties: false``); outbound schemas are permissive so the
daemon can grow an envelope without breaking clients.
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
