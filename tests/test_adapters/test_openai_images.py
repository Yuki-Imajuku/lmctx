import pytest

from lmctx import Context, Instructions, Part, RunSpec, ToolSpecification
from lmctx.adapters import OpenAIImagesAdapter
from lmctx.adapters._openai_images import _modalities_to_response_format
from lmctx.plan import LmctxAdapter


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {
        "provider": "openai",
        "endpoint": "images.generate",
        "model": "gpt-image-1",
    }
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


def test_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = OpenAIImagesAdapter()
    assert isinstance(adapter, LmctxAdapter)


def test_plan_uses_latest_user_prompt() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Old prompt").user("Draw a red fox in watercolor style.")

    plan = adapter.plan(ctx, _spec())
    assert plan.request["model"] == "gpt-image-1"
    assert plan.request["prompt"] == "Draw a red fox in watercolor style."


def test_plan_combines_instructions_and_multiple_text_parts() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user((Part(type="text", text="Draw a fox."), Part(type="text", text="Use watercolor style.")))

    plan = adapter.plan(
        ctx,
        _spec(
            instructions=Instructions(
                system="You are an image generation assistant.",
                developer="Follow style requirements exactly.",
            )
        ),
    )
    assert plan.request["prompt"] == (
        "You are an image generation assistant.\n\n"
        "Follow style requirements exactly.\n\n"
        "Draw a fox.\nUse watercolor style."
    )
    assert "system instruction" in plan.included
    assert "developer instruction" in plan.included


def test_plan_with_response_modalities_sets_response_format() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a city skyline.")

    plan = adapter.plan(ctx, _spec(model="dall-e-3", response_modalities=("b64_json",)))
    assert plan.request["response_format"] == "b64_json"
    assert "response_modalities" in plan.included


def test_plan_with_url_response_modalities_sets_url_response_format() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a city skyline.")

    plan = adapter.plan(ctx, _spec(model="dall-e-3", response_modalities=(" URL ",)))
    assert plan.request["response_format"] == "url"
    assert "response_modalities" in plan.included


def test_plan_ignores_response_modalities_for_gpt_image_models() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a city skyline.")

    plan = adapter.plan(ctx, _spec(response_modalities=("b64_json",)))
    assert "response_format" not in plan.request
    assert "response_modalities" not in plan.included
    assert any("response_modalities is ignored" in warning for warning in plan.warnings)


def test_plan_warns_for_unknown_response_modalities_value() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a city skyline.")

    plan = adapter.plan(ctx, _spec(model="dall-e-3", response_modalities=("jpeg",)))
    assert "response_format" not in plan.request
    assert any("value is not recognized" in warning for warning in plan.warnings)


def test_plan_reports_seed_as_excluded() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a city skyline.")

    plan = adapter.plan(ctx, _spec(seed=42))
    assert any(item.description == "seed" for item in plan.excluded)


def test_plan_excludes_unsupported_runspec_fields() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a city skyline.")
    tool = ToolSpecification(
        name="lookup_style",
        description="Return style guide",
        input_schema={"type": "object", "properties": {"style": {"type": "string"}}},
    )

    plan = adapter.plan(
        ctx,
        _spec(
            max_output_tokens=512,
            temperature=0.3,
            top_p=0.9,
            seed=7,
            tools=(tool,),
            tool_choice="auto",
            response_schema={"type": "object", "properties": {"caption": {"type": "string"}}},
        ),
    )

    excluded = {item.description for item in plan.excluded}
    assert excluded == {
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "tools",
        "tool_choice",
        "response_schema",
    }


def test_plan_deep_merges_extra_body() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a mountain.")

    plan = adapter.plan(ctx, _spec(extra_body={"size": "1024x1024", "quality": "high"}))
    assert plan.request["size"] == "1024x1024"
    assert plan.request["quality"] == "high"


def test_plan_includes_extra_headers_and_extra_query() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a mountain.")

    plan = adapter.plan(
        ctx,
        _spec(
            extra_headers={"X-Test": "1"},
            extra_query={"api-version": "2024-07-01"},
        ),
    )
    assert plan.request["extra_headers"] == {"X-Test": "1"}
    assert plan.request["extra_query"] == {"api-version": "2024-07-01"}
    assert "extra_headers" in plan.included
    assert "extra_query" in plan.included


def test_plan_surfaces_transport_hints_in_extra() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a mountain.")

    plan = adapter.plan(ctx, _spec(base_url="https://example.test/v1", api_version="2024-07-01"))
    assert plan.extra == {
        "base_url": "https://example.test/v1",
        "api_version": "2024-07-01",
    }


def test_plan_requires_user_text() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context()
    with pytest.raises(ValueError, match="requires at least one user message"):
        adapter.plan(ctx, _spec())


def test_plan_requires_text_content_in_latest_user_message() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat.").user((Part(type="image"),))
    with pytest.raises(ValueError, match="requires text content in the latest user message"):
        adapter.plan(ctx, _spec())


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_plan_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_ingest_b64_image_response() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")

    response = {
        "created": 1735689600,
        "data": [
            {
                "b64_json": "aGVsbG8=",
            }
        ],
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 1
    assert last.parts[0].type == "image"
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get_blob(last.parts[0].blob) == b"hello"


def test_ingest_url_image_response() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")
    response = {
        "data": [
            {
                "url": "https://example.com/generated.png",
            }
        ]
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert last.parts[0] == Part(type="image", provider_raw={"url": "https://example.com/generated.png"})


def test_ingest_returns_same_context_when_data_is_not_list() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")

    ctx2 = adapter.ingest(ctx, {"data": {"url": "https://example.com/generated.png"}}, spec=_spec())
    assert ctx2 is ctx


def test_ingest_skips_invalid_and_non_mapping_items() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")
    response = {
        "data": [
            123,
            {"b64_json": "%%%%"},
            {"url": ""},
            {"note": "no image payload"},
        ]
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert ctx2.last(role="assistant") is None


def test_ingest_usage_if_present() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")
    response = {
        "data": [{"b64_json": "aGVsbG8="}],
        "usage": {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2.usage_log) == 1
    assert ctx2.usage_log[0].input_tokens == 5
    assert ctx2.usage_log[0].output_tokens == 10
    assert ctx2.usage_log[0].total_tokens == 15


def test_ingest_usage_keeps_only_integer_token_fields() -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")
    response = {
        "data": [],
        "usage": {
            "input_tokens": "5",
            "output_tokens": 10.5,
            "total_tokens": 15,
            "cache_hits": 2,
        },
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2.usage_log) == 1
    usage = ctx2.usage_log[0]
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens == 15
    assert usage.provider_usage["cache_hits"] == 2


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        ((), None),
        (("webp",), None),
    ],
)
def test_modalities_to_response_format_returns_none_for_empty_or_unknown(
    modalities: tuple[str, ...], expected: str | None
) -> None:
    assert _modalities_to_response_format(modalities) == expected


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_ingest_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = OpenAIImagesAdapter()
    ctx = Context().user("Draw a cat")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"data": []}, spec=bad_spec)
