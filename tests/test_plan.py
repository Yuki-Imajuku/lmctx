from types import MappingProxyType

from lmctx.plan import AdapterId, ExcludedItem, LmctxAdapter, RequestPlan


def test_adapter_id_defaults() -> None:
    aid = AdapterId(provider="openai", endpoint="chat.completions")
    assert aid.provider == "openai"
    assert aid.endpoint == "chat.completions"
    assert aid.api_version is None


def test_adapter_id_with_version() -> None:
    aid = AdapterId(provider="azure", endpoint="chat.completions", api_version="2024-02-01")
    assert aid.api_version == "2024-02-01"


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
    assert hasattr(LmctxAdapter, "plan")
    assert hasattr(LmctxAdapter, "ingest")
