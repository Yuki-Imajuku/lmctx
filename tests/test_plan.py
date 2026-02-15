from types import MappingProxyType

import pytest

from lmctx.errors import PlanValidationError
from lmctx.plan import AdapterCapabilities, AdapterId, ExcludedItem, LmctxAdapter, RequestPlan


def test_adapter_id_defaults() -> None:
    aid = AdapterId(provider="openai", endpoint="chat.completions")
    assert aid.provider == "openai"
    assert aid.endpoint == "chat.completions"
    assert aid.api_version is None


def test_adapter_id_with_version() -> None:
    aid = AdapterId(provider="azure", endpoint="chat.completions", api_version="2024-02-01")
    assert aid.api_version == "2024-02-01"


def test_adapter_capabilities_valid_payload() -> None:
    capabilities = AdapterCapabilities(
        id=AdapterId(provider="openai", endpoint="chat.completions"),
        fields={"tools": "yes", "seed": "partial", "cursor_chaining": "no"},
        notes={"seed": "Only some models expose deterministic seed support."},
    )

    assert capabilities.level("tools") == "yes"
    assert capabilities.level("seed") == "partial"
    assert capabilities.level("unknown") is None
    assert capabilities.is_supported("tools")
    assert capabilities.is_supported("seed")
    assert not capabilities.is_supported("seed", allow_partial=False)
    assert not capabilities.is_supported("cursor_chaining")


def test_adapter_capabilities_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="Invalid capability level"):
        AdapterCapabilities(
            id=AdapterId(provider="openai", endpoint="chat.completions"),
            fields={"tools": "maybe"},  # type: ignore[arg-type]
        )


def test_adapter_capabilities_rejects_unknown_note_keys() -> None:
    with pytest.raises(ValueError, match="unknown field keys"):
        AdapterCapabilities(
            id=AdapterId(provider="openai", endpoint="chat.completions"),
            fields={"tools": "yes"},
            notes={"seed": "not part of fields map"},
        )


def test_excluded_item() -> None:
    item = ExcludedItem(description="message[3]", reason="exceeds token budget")
    assert item.description == "message[3]"
    assert item.reason == "exceeds token budget"


def test_request_plan_defaults() -> None:
    plan = RequestPlan(request={"model": "gpt-4o", "messages": []})
    assert plan.request == {"model": "gpt-4o", "messages": []}
    assert plan.included == ()
    assert plan.excluded == ()
    assert plan.must_roundtrip == ()
    assert plan.warnings == ()
    assert plan.errors == ()
    assert plan.token_estimate is None
    assert plan.extra == {}


def test_request_plan_with_diagnostics() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        included=("5 messages", "2 tools"),
        excluded=(ExcludedItem("msg[0]", "system instruction moved to RunSpec"),),
        warnings=("max_tokens not set",),
        errors=(),
        token_estimate=1500,
    )
    assert len(plan.included) == 2
    assert len(plan.excluded) == 1
    assert plan.excluded[0].reason == "system instruction moved to RunSpec"
    assert plan.token_estimate == 1500


def test_request_plan_normalizes_nested_mappings() -> None:
    plan = RequestPlan(
        request={"a": MappingProxyType({"b": MappingProxyType({"c": 1})})},
        extra={"x": MappingProxyType({"y": 2})},
    )
    assert plan.request == {"a": {"b": {"c": 1}}}
    assert plan.extra == {"x": {"y": 2}}


def test_lmctx_adapter_is_runtime_checkable() -> None:
    assert isinstance(LmctxAdapter, type)
    # Protocol itself should be checkable
    assert hasattr(LmctxAdapter, "capabilities")
    assert hasattr(LmctxAdapter, "plan")
    assert hasattr(LmctxAdapter, "ingest")


def test_request_plan_unused_parameter_warnings() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        excluded=(
            ExcludedItem(description="seed", reason="provider does not support deterministic seed"),
            ExcludedItem(description="extra_headers", reason="adapter does not map transport overrides"),
        ),
    )

    assert plan.unused_parameter_warnings() == (
        "unused parameter 'seed': provider does not support deterministic seed",
        "unused parameter 'extra_headers': adapter does not map transport overrides",
    )


def test_request_plan_warning_messages_include_unused_parameters_by_default() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        warnings=("max_output_tokens not set",),
        excluded=(ExcludedItem(description="seed", reason="provider does not support deterministic seed"),),
    )

    assert plan.warning_messages() == (
        "max_output_tokens not set",
        "unused parameter 'seed': provider does not support deterministic seed",
    )


def test_request_plan_warning_messages_can_exclude_unused_parameters() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        warnings=("max_output_tokens not set",),
        excluded=(ExcludedItem(description="seed", reason="provider does not support deterministic seed"),),
    )

    assert plan.warning_messages(include_unused_parameters=False) == ("max_output_tokens not set",)


def test_request_plan_assert_valid_raises_on_errors_by_default() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        errors=("request has no input items",),
    )

    with pytest.raises(PlanValidationError, match="request has no input items"):
        plan.assert_valid()


def test_request_plan_assert_valid_can_fail_on_warnings() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        warnings=("max_output_tokens not set",),
    )

    with pytest.raises(PlanValidationError, match="warning: max_output_tokens not set"):
        plan.assert_valid(fail_on_warnings=True)


def test_request_plan_assert_valid_can_fail_on_excluded_items() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        excluded=(ExcludedItem(description="seed", reason="provider does not support deterministic seed"),),
    )

    with pytest.raises(PlanValidationError, match="unused parameter 'seed'"):
        plan.assert_valid(fail_on_excluded=True)


def test_request_plan_assert_valid_passes_when_strict_modes_disabled() -> None:
    plan = RequestPlan(
        request={"model": "gpt-4o"},
        warnings=("max_output_tokens not set",),
        excluded=(ExcludedItem(description="seed", reason="provider does not support deterministic seed"),),
    )

    plan.assert_valid()
