from __future__ import annotations

from enum import Enum

import pytest

from lmctx.adapters._util import (
    _as_object_dict,
    _deep_merge,
    _json_object,
    _plan_extra_hints,
    _to_dict,
    _to_json_compatible,
    _validate_adapter_spec,
)
from lmctx.plan import AdapterId
from lmctx.spec import RunSpec


class _Color(Enum):
    RED = "red"


class _ModelDumpNoKwargs:
    def model_dump(self) -> dict[str, object]:
        return {"source": "model_dump_no_kwargs"}


class _ModelDumpWithKwargs:
    def model_dump(self, *, mode: str = "python", warnings: bool = True) -> dict[str, object]:
        if mode != "python" or warnings is not False:
            msg = "unexpected model_dump kwargs"
            raise AssertionError(msg)
        return {"source": "model_dump_with_kwargs"}


class _ToDictObject:
    def to_dict(self) -> dict[str, object]:
        return {"source": "to_dict"}


class _BadObject:
    pass


def test_as_object_dict_normalizes_mapping_and_non_mapping() -> None:
    assert _as_object_dict({"a": 1}) == {"a": 1}
    assert _as_object_dict(123) == {}


def test_to_dict_supports_mapping() -> None:
    assert _to_dict({"k": 1}) == {"k": 1}


def test_to_dict_supports_model_dump_with_kwargs() -> None:
    assert _to_dict(_ModelDumpWithKwargs()) == {"source": "model_dump_with_kwargs"}


def test_to_dict_falls_back_when_model_dump_kwargs_unsupported() -> None:
    assert _to_dict(_ModelDumpNoKwargs()) == {"source": "model_dump_no_kwargs"}


def test_to_dict_supports_to_dict_method() -> None:
    assert _to_dict(_ToDictObject()) == {"source": "to_dict"}


def test_to_dict_raises_for_unsupported_object() -> None:
    with pytest.raises(TypeError, match="Cannot convert _BadObject to dict"):
        _to_dict(_BadObject())


def test_deep_merge_recursively_merges_mappings() -> None:
    base = {"config": {"a": 1, "b": 2}, "keep": True}
    override = {"config": {"b": 20, "c": 30}, "replace": "x"}
    merged = _deep_merge(base, override)

    assert merged == {"config": {"a": 1, "b": 20, "c": 30}, "keep": True, "replace": "x"}
    assert base == {"config": {"a": 1, "b": 2}, "keep": True}


def test_to_json_compatible_handles_enum_and_sequences() -> None:
    payload = {
        "enum": _Color.RED,
        "tuple": (1, 2),
        "list": [3, _Color.RED],
    }
    assert _to_json_compatible(payload) == {
        "enum": "red",
        "tuple": [1, 2],
        "list": [3, "red"],
    }


def test_json_object_normalizes_keys_and_values() -> None:
    assert _json_object({"enum": _Color.RED, "tuple": (1, 2)}) == {"enum": "red", "tuple": [1, 2]}


def test_validate_adapter_spec_accepts_matching_spec() -> None:
    adapter_id = AdapterId(provider="openai", endpoint="chat.completions", api_version="2024-07-01")
    spec = RunSpec(
        provider="openai",
        endpoint="chat.completions",
        api_version="2024-07-01",
        model="gpt-4o",
    )
    _validate_adapter_spec(adapter_id, spec)


def test_validate_adapter_spec_rejects_mismatches() -> None:
    adapter_id = AdapterId(provider="openai", endpoint="chat.completions", api_version="2024-07-01")

    with pytest.raises(ValueError, match="provider mismatch"):
        _validate_adapter_spec(
            adapter_id,
            RunSpec(provider="google", endpoint="chat.completions", api_version="2024-07-01", model="gpt-4o"),
        )

    with pytest.raises(ValueError, match="endpoint mismatch"):
        _validate_adapter_spec(
            adapter_id,
            RunSpec(provider="openai", endpoint="responses.create", api_version="2024-07-01", model="gpt-4o"),
        )

    with pytest.raises(ValueError, match="api_version mismatch"):
        _validate_adapter_spec(
            adapter_id,
            RunSpec(provider="openai", endpoint="chat.completions", api_version="2025-01-01", model="gpt-4o"),
        )


def test_plan_extra_hints_returns_only_non_empty_hints() -> None:
    assert _plan_extra_hints(RunSpec(provider="openai", endpoint="chat.completions", model="gpt-4o")) == {}

    hints = _plan_extra_hints(
        RunSpec(
            provider="openai",
            endpoint="chat.completions",
            model="gpt-4o",
            base_url="https://example.test/v1",
            api_version="2024-07-01",
        )
    )
    assert hints == {"base_url": "https://example.test/v1", "api_version": "2024-07-01"}
