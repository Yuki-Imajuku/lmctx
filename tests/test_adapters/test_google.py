import pytest

from lmctx import Context, Message, Part, ToolSpecification
from lmctx.adapters import GoogleGenAIAdapter
from lmctx.blobs import InMemoryBlobStore
from lmctx.plan import LmctxAdapter
from lmctx.spec import Instructions, RunSpec


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {
        "provider": "google",
        "endpoint": "models.generate_content",
        "model": "gemini-2.0-flash",
    }
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Protocol Conformance
# =============================================================================


def test_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = GoogleGenAIAdapter()
    assert isinstance(adapter, LmctxAdapter)


# =============================================================================
# plan()
# =============================================================================


def test_plan_simple_text() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello").assistant("Hi").user("How are you?")
    plan = adapter.plan(ctx, _spec())

    assert plan.request["model"] == "gemini-2.0-flash"
    contents = plan.request["contents"]
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "Hello"}]
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": "Hi"}]
    assert contents[2]["role"] == "user"


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_plan_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_plan_with_system_instructions() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(instructions=Instructions(system="You are helpful."))
    plan = adapter.plan(ctx, spec)

    assert plan.request["config"]["system_instruction"] == "You are helpful."
    assert "system instruction" in plan.included


def test_plan_system_message_in_context() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context()
    ctx = ctx.append(Message(role="system", parts=(Part(type="text", text="From context."),)))
    ctx = ctx.user("Hello")

    spec = _spec(instructions=Instructions(system="From RunSpec."))
    plan = adapter.plan(ctx, spec)

    system = plan.request["config"]["system_instruction"]
    assert "From RunSpec." in system
    assert "From context." in system


def test_plan_with_tools() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,))
    plan = adapter.plan(ctx, spec)

    tools = plan.request["config"]["tools"]
    assert len(tools) == 1
    decls = tools[0]["function_declarations"]
    assert decls[0]["name"] == "get_weather"


def test_plan_tool_call_and_result() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("What's the weather?")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_name="get_weather", tool_args={"city": "Tokyo"}),),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_name="get_weather", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    contents = plan.request["contents"]

    # user, model (function_call), user (function_response)
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["function_call"]["name"] == "get_weather"
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["function_response"]["name"] == "get_weather"
    assert contents[2]["parts"][0]["function_response"]["response"] == {"temp": 22}


def test_plan_multimodal_image() -> None:
    adapter = GoogleGenAIAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"fake-png-bytes", media_type="image/png", kind="image")
    ctx = ctx.user(
        [
            Part(type="text", text="Describe this."),
            Part(type="image", blob=ref),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    parts = plan.request["contents"][0]["parts"]
    assert parts[0] == {"text": "Describe this."}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"


def test_plan_with_file_part() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user(
        [
            Part(type="text", text="Summarize this file."),
            Part(type="file", provider_raw={"file_uri": "gs://bucket/notebook.pdf", "mime_type": "application/pdf"}),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    parts = plan.request["contents"][0]["parts"]
    assert parts[0] == {"text": "Summarize this file."}
    assert parts[1] == {"file_data": {"file_uri": "gs://bucket/notebook.pdf", "mime_type": "application/pdf"}}


def test_plan_with_file_blob_part() -> None:
    adapter = GoogleGenAIAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"%PDF-1.4 fake content", media_type="application/pdf", kind="document")
    ctx = ctx.user([Part(type="file", blob=ref)])

    plan = adapter.plan(ctx, _spec())
    parts = plan.request["contents"][0]["parts"]
    assert parts == [
        {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": "JVBERi0xLjQgZmFrZSBjb250ZW50",
            }
        }
    ]


def test_plan_reports_invalid_part_with_part_level_excluded() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user([Part(type="text", text="Keep"), Part(type="tool_result", tool_output={"ok": True})])

    plan = adapter.plan(ctx, _spec())
    assert plan.request["contents"] == [{"role": "user", "parts": [{"text": "Keep"}]}]
    assert any(item.description == "context.messages[0].parts[1]" for item in plan.excluded)


def test_plan_generation_params() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(temperature=0.5, max_output_tokens=100, top_p=0.9)
    plan = adapter.plan(ctx, spec)

    config = plan.request["config"]
    assert config["temperature"] == 0.5
    assert config["max_output_tokens"] == 100
    assert config["top_p"] == 0.9


def test_plan_with_response_modalities() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Generate an image")
    spec = _spec(response_modalities=("IMAGE",))
    plan = adapter.plan(ctx, spec)

    assert plan.request["config"]["response_modalities"] == ["IMAGE"]
    assert "response_modalities" in plan.included


def test_plan_extra_body_deep_merge() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        tool_choice={"function_calling_config": {"mode": "ANY", "allowed_function_names": ["weather"]}},
        extra_body={"tool_config": {"function_calling_config": {"allowed_function_names": ["search"]}}},
    )
    plan = adapter.plan(ctx, spec)

    tool_config = plan.request["config"]["tool_config"]["function_calling_config"]
    assert tool_config["mode"] == "ANY"
    assert tool_config["allowed_function_names"] == ["search"]


def test_plan_extra_headers_and_extra_body_http_options_deep_merge() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        extra_headers={"X-Trace-Id": "trace-1"},
        extra_body={"http_options": {"timeout": 30}},
    )
    plan = adapter.plan(ctx, spec)

    http_options = plan.request["config"]["http_options"]
    assert http_options["headers"] == {"X-Trace-Id": "trace-1"}
    assert http_options["timeout"] == 30


def test_plan_extra_headers_and_extra_body_http_options_headers_deep_merge() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        extra_headers={"X-A": "1"},
        extra_body={"http_options": {"headers": {"X-B": "2"}}},
    )
    plan = adapter.plan(ctx, spec)

    http_options = plan.request["config"]["http_options"]
    assert http_options["headers"] == {"X-A": "1", "X-B": "2"}


def test_plan_extra_body_headers_override_extra_headers_on_conflict() -> None:
    """When both sources set the same header key, extra_body wins (more explicit)."""
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        extra_headers={"X-A": "1"},
        extra_body={"http_options": {"headers": {"X-A": "2"}}},
    )
    plan = adapter.plan(ctx, spec)

    http_options = plan.request["config"]["http_options"]
    assert http_options["headers"] == {"X-A": "2"}


def test_plan_extra_headers_mapped_to_http_options_headers() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(extra_headers={"X-Test": "1"})
    plan = adapter.plan(ctx, spec)

    assert plan.request["config"]["http_options"]["headers"] == {"X-Test": "1"}
    assert "extra_headers" in plan.included


def test_plan_extra_query_reported_as_excluded() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    spec = _spec(extra_query={"api-version": "2025-01-01"})
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "extra_query" for item in plan.excluded)


def test_plan_consecutive_user_messages_merged() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("First").user("Second")
    plan = adapter.plan(ctx, _spec())

    contents = plan.request["contents"]
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert len(contents[0]["parts"]) == 2


def test_plan_omits_invalid_tool_result_without_name() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="fc_001", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    contents = plan.request["contents"]
    assert len(contents) == 1
    assert any(item.description == "context.messages[1].parts[0]" for item in plan.excluded)
    assert any("requires tool_name" in item.reason for item in plan.excluded)


def test_plan_tool_result_can_resolve_name_from_prior_tool_call_id() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Weather?")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="tool_call", tool_call_id="fc_001", tool_name="get_weather", tool_args={"city": "Tokyo"}),
            ),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="fc_001", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    function_response = plan.request["contents"][2]["parts"][0]["function_response"]
    assert function_response["name"] == "get_weather"
    assert function_response["id"] == "fc_001"


# =============================================================================
# ingest()
# =============================================================================


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_ingest_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"candidates": []}, spec=bad_spec)


def test_ingest_text_response() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hi there!"}],
                },
                "finish_reason": "STOP",
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
            "cached_content_token_count": 4,
            "thoughts_token_count": 2,
            "tool_use_prompt_token_count": 1,
            "prompt_tokens_details": [{"modality": "TEXT", "token_count": 8}],
            "candidates_tokens_details": [{"modality": "TEXT", "token_count": 5}],
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx) == 2
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].text == "Hi there!"
    assert last.provider == "google"

    assert len(ctx.usage_log) == 1
    assert ctx.usage_log[0].input_tokens == 10
    assert ctx.usage_log[0].output_tokens == 5
    assert ctx.usage_log[0].total_tokens == 15
    assert ctx.usage_log[0].provider_usage["prompt_token_count"] == 10
    assert ctx.usage_log[0].provider_usage["candidates_token_count"] == 5
    assert ctx.usage_log[0].provider_usage["total_token_count"] == 15
    assert ctx.usage_log[0].provider_usage["cached_content_token_count"] == 4
    assert ctx.usage_log[0].provider_usage["thoughts_token_count"] == 2
    assert ctx.usage_log[0].provider_usage["tool_use_prompt_token_count"] == 1


def test_ingest_function_call_response() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"function_call": {"name": "get_weather", "args": {"city": "Tokyo"}}},
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "tool_call"
    assert last.parts[0].tool_name == "get_weather"
    assert last.parts[0].tool_args == {"city": "Tokyo"}


def test_ingest_inline_image_response() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Generate an image")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"inline_data": {"mime_type": "image/png", "data": "aGVsbG8="}},
                    ],
                },
            }
        ]
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "image"
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get(last.parts[0].blob) == b"hello"


def test_ingest_inline_image_response_with_bytes_payload() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Generate an image")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": b"\x89PNG\x0d\x0a"}},
                    ],
                },
            }
        ]
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "image"
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get(last.parts[0].blob) == b"\x89PNG\x0d\x0a"


def test_ingest_inline_audio_response_maps_to_audio_part() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Generate audio")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"inline_data": {"mime_type": "audio/wav", "data": "aGVsbG8="}},
                    ],
                },
            }
        ]
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "audio"
    assert last.parts[0].blob is not None
    assert last.parts[0].blob.kind == "audio"
    assert last.parts[0].blob.media_type == "audio/wav"
    assert ctx2.blob_store.get(last.parts[0].blob) == b"hello"


def test_ingest_function_call_response_camel_case() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"id": "fc_001", "name": "get_weather", "args": {"city": "Tokyo"}}},
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "tool_call"
    assert last.parts[0].tool_call_id == "fc_001"
    assert last.parts[0].tool_name == "get_weather"
    assert last.parts[0].tool_args == {"city": "Tokyo"}


def test_ingest_empty_candidates() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")

    response = {"candidates": []}
    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2) == 1


def test_ingest_camel_case_usage() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Hello")

    response = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}}],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
            "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 3}],
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    assert ctx.usage_log[0].input_tokens == 10
    assert ctx.usage_log[0].output_tokens == 5
    assert ctx.usage_log[0].total_tokens == 15
    assert ctx.usage_log[0].provider_usage["promptTokenCount"] == 10
    assert ctx.usage_log[0].provider_usage["candidatesTokenCount"] == 5
    assert ctx.usage_log[0].provider_usage["totalTokenCount"] == 15
    assert ctx.usage_log[0].provider_usage["cacheTokensDetails"] == ({"modality": "TEXT", "tokenCount": 3},)


def test_ingest_usage_prefers_extended_buckets_when_matching_total() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Think and answer.")

    response = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Done."}]}}],
        "usage_metadata": {
            "prompt_token_count": 16,
            "candidates_token_count": 8,
            "thoughts_token_count": 16,
            "tool_use_prompt_token_count": 0,
            "total_token_count": 40,
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    usage = ctx.usage_log[0]
    assert usage.input_tokens == 16
    assert usage.output_tokens == 24
    assert usage.total_tokens == 40


def test_ingest_usage_includes_extended_buckets_when_total_missing() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Use tool and think.")

    response = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Done."}]}}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "toolUsePromptTokenCount": 20,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 50,
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    usage = ctx.usage_log[0]
    assert usage.input_tokens == 120
    assert usage.output_tokens == 80
    assert usage.total_tokens == 200


# =============================================================================
# Thinking/Thought
# =============================================================================


def test_ingest_thought_part() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Solve this.")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me think...", "thought": True},
                        {"text": "The answer is 42."},
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2

    thinking = last.parts[0]
    assert thinking.type == "thinking"
    assert thinking.text == "Let me think..."
    assert thinking.provider_raw is not None
    assert thinking.provider_raw["thought"] is True

    text = last.parts[1]
    assert text.type == "text"
    assert text.text == "The answer is 42."


def test_plan_thought_roundtrip() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Solve this.")

    # Simulate: ingest a thinking response.
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Thinking...", "thought": True},
                        {"text": "Answer."},
                    ],
                },
            }
        ],
    }
    ctx = adapter.ingest(ctx, response, spec=_spec())
    ctx = ctx.user("Follow up.")

    plan = adapter.plan(ctx, _spec())
    contents = plan.request["contents"]

    # contents[0] = user "Solve this."
    # contents[1] = model with thinking + text
    # contents[2] = user "Follow up."
    model_parts = contents[1]["parts"]
    assert model_parts[0]["thought"] is True
    assert model_parts[0]["text"] == "Thinking..."
    assert model_parts[1] == {"text": "Answer."}


def test_ingest_function_call_with_id() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": "fc_gemini_001",
                                "name": "get_weather",
                                "args": {"city": "Tokyo"},
                            }
                        },
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "tool_call"
    assert last.parts[0].tool_call_id == "fc_gemini_001"
    assert last.parts[0].tool_name == "get_weather"


def test_plan_function_call_with_id() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Weather?")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="tool_call", tool_call_id="fc_001", tool_name="get_weather", tool_args={"city": "Tokyo"}),
            ),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="fc_001", tool_name="get_weather", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    contents = plan.request["contents"]

    # Model's function_call should include the id.
    fc_part = contents[1]["parts"][0]["function_call"]
    assert fc_part["id"] == "fc_001"
    assert fc_part["name"] == "get_weather"

    # User's function_response should include the id.
    fr_part = contents[2]["parts"][0]["function_response"]
    assert fr_part["id"] == "fc_001"
    assert fr_part["name"] == "get_weather"


def test_plan_preserves_function_call_thought_signature() -> None:
    adapter = GoogleGenAIAdapter()
    ctx = Context().user("Weather?")

    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "fc_sig_001",
                                "name": "get_weather",
                                "args": {"city": "Tokyo"},
                            },
                            "thoughtSignature": "sig_abc123",
                        },
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    plan = adapter.plan(ctx, _spec())
    contents = plan.request["contents"]
    tool_part = contents[1]["parts"][0]
    fc_part = tool_part["function_call"]

    assert fc_part["id"] == "fc_sig_001"
    assert fc_part["name"] == "get_weather"
    assert tool_part["thought_signature"] == "sig_abc123"
