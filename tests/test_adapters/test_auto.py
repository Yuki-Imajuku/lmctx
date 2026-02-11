import pytest

from lmctx import Context, RunSpec
from lmctx.adapters import AutoAdapter, OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter
from lmctx.plan import AdapterId, RequestPlan


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {
        "provider": "openai",
        "endpoint": "responses.create",
        "model": "gpt-4o",
    }
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


def test_resolve_built_in_adapter() -> None:
    adapter = AutoAdapter()
    resolved = adapter.resolve(_spec())
    assert isinstance(resolved, OpenAIResponsesAdapter)


def test_plan_delegates_to_resolved_adapter() -> None:
    auto = AutoAdapter()
    direct = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    spec = _spec(endpoint="chat.completions")

    auto_plan = auto.plan(ctx, spec)
    direct_plan = direct.plan(ctx, spec)

    assert auto_plan.request == direct_plan.request
    assert auto_plan.included == direct_plan.included
    assert auto_plan.excluded == direct_plan.excluded


def test_ingest_delegates_to_resolved_adapter() -> None:
    auto = AutoAdapter()
    ctx = Context().user("Hello")
    spec = _spec(endpoint="chat.completions")
    response = {
        "id": "chatcmpl_1",
        "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}],
    }

    updated = auto.ingest(ctx, response, spec=spec)
    last = updated.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 1
    assert last.parts[0].type == "text"
    assert last.parts[0].text == "Hi there!"


def test_register_duplicate_adapter_requires_replace() -> None:
    auto = AutoAdapter(adapters=[])
    first = OpenAIResponsesAdapter()
    second = OpenAIResponsesAdapter()
    auto.register(first)

    with pytest.raises(ValueError, match="already registered"):
        auto.register(second)

    auto.register(second, replace=True)
    resolved = auto.resolve(_spec())
    assert resolved is second


def test_register_custom_versioned_adapter() -> None:
    class _CustomAdapter:
        id = AdapterId(provider="acme", endpoint="chat.generate", api_version="2026-02-01")

        def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
            return RequestPlan(request={"model": spec.model, "messages": len(ctx)})

        def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
            return ctx.assistant("ok")

    auto = AutoAdapter(adapters=[])
    custom = _CustomAdapter()
    auto.register(custom)

    spec = RunSpec(provider="acme", endpoint="chat.generate", api_version="2026-02-01", model="acme-1")
    resolved = auto.resolve(spec)
    assert resolved is custom


def test_resolve_falls_back_to_unversioned_adapter() -> None:
    class _CustomAdapter:
        id = AdapterId(provider="acme", endpoint="chat.generate")

        def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
            return RequestPlan(request={"model": spec.model, "messages": len(ctx)})

        def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
            return ctx.assistant("ok")

    auto = AutoAdapter(adapters=[])
    custom = _CustomAdapter()
    auto.register(custom)

    spec = RunSpec(provider="acme", endpoint="chat.generate", api_version="2026-02-01", model="acme-1")
    resolved = auto.resolve(spec)
    assert resolved is custom


def test_available_ids_returns_registered_ids_sorted() -> None:
    auto = AutoAdapter(adapters=[])
    auto.register(OpenAIChatCompletionsAdapter())
    auto.register(OpenAIResponsesAdapter())

    available = auto.available_ids()
    assert available == (
        AdapterId(provider="openai", endpoint="chat.completions"),
        AdapterId(provider="openai", endpoint="responses.create"),
    )


def test_available_ids_supports_mixed_versioned_and_unversioned_ids() -> None:
    class _UnversionedAdapter:
        id = AdapterId(provider="acme", endpoint="chat.generate")

        def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
            return RequestPlan(request={"model": spec.model, "messages": len(ctx)})

        def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
            return ctx

    class _VersionedAdapter:
        id = AdapterId(provider="acme", endpoint="chat.generate", api_version="2026-02-01")

        def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
            return RequestPlan(request={"model": spec.model, "messages": len(ctx)})

        def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
            return ctx

    auto = AutoAdapter(adapters=[])
    auto.register(_VersionedAdapter())
    auto.register(_UnversionedAdapter())

    available = auto.available_ids()
    assert available == (
        AdapterId(provider="acme", endpoint="chat.generate"),
        AdapterId(provider="acme", endpoint="chat.generate", api_version="2026-02-01"),
    )


def test_resolve_raises_on_unknown_target() -> None:
    auto = AutoAdapter(adapters=[])
    spec = RunSpec(provider="unknown", endpoint="missing", model="x")
    with pytest.raises(ValueError, match="No adapter registered"):
        auto.resolve(spec)


def test_resolve_unknown_target_reports_available_ids_with_mixed_versions() -> None:
    class _UnversionedAdapter:
        id = AdapterId(provider="acme", endpoint="chat.generate")

        def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
            return RequestPlan(request={"model": spec.model, "messages": len(ctx)})

        def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
            return ctx

    class _VersionedAdapter:
        id = AdapterId(provider="acme", endpoint="chat.generate", api_version="2026-02-01")

        def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
            return RequestPlan(request={"model": spec.model, "messages": len(ctx)})

        def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
            return ctx

    auto = AutoAdapter(adapters=[])
    auto.register(_VersionedAdapter())
    auto.register(_UnversionedAdapter())

    spec = RunSpec(provider="unknown", endpoint="missing", model="x")
    with pytest.raises(ValueError, match="No adapter registered") as exc_info:
        auto.resolve(spec)

    message = str(exc_info.value)
    assert "acme/chat.generate" in message
    assert "acme/chat.generate@2026-02-01" in message
