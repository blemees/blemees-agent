"""Smoke tests for the schemas subpackage loader.

The blemees-agent/1 (`agent.*`) JSON schemas were removed in the blemees/3
clean break (#16); the blemees/3 `session.*` schemas are a tracked follow-up.
Until they land, `iter_schemas()` yields nothing — this test guards that the
loader API stays importable and robust to an empty schema set, and
meta-validates any schemas that do exist.
"""

from __future__ import annotations

import pytest

from blemees_agent import schemas


def test_iter_schemas_is_robust_to_empty_set():
    # Must not raise even though inbound/outbound currently ship no schemas.
    assert isinstance(list(schemas.iter_schemas()), list)


def test_files_traversable_available():
    assert schemas.files() is not None


def test_shipped_schemas_meta_validate_when_present():
    jsonschema = pytest.importorskip("jsonschema")
    for schema in schemas.iter_schemas():
        jsonschema.Draft202012Validator.check_schema(schema)
