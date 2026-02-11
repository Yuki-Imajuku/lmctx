import json

import pytest

from lmctx import Context, Message, Part, ToolSpecification
from lmctx.adapters import OpenAIChatCompletionsAdapter
from lmctx.blobs import InMemoryBlobStore
from lmctx.plan import LmctxAdapter
from lmctx.spec import Instructions, RunSpec


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {"provider": "openai", "endpoint": "chat.completions", "model": "gpt-4o"}
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Protocol Conformance
# =============================================================================


def test_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    assert isinstance(adapter, LmctxAdapter)


# =============================================================================
# plan()
# =============================================================================


def test_plan_simple_text() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello").assistant("Hi there").user("How are you?")
    plan = adapter.plan(ctx, _spec())

    messages = plan.request["messages"]
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi there"}
    assert messages[2] == {"role": "user", "content": "How are you?"}
    assert plan.request["model"] == "gpt-4o"


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_plan_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_plan_with_system_instructions() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    spec = _spec(instructions=Instructions(system="You are helpful.", developer="Be concise."))
    plan = adapter.plan(ctx, spec)

    messages = plan.request["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "developer", "content": "Be concise."}
    assert messages[2] == {"role": "user", "content": "Hello"}
    assert "system instruction" in plan.included
    assert "developer instruction" in plan.included


def test_plan_with_tools() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,), tool_choice="auto")
    plan = adapter.plan(ctx, spec)

    assert len(plan.request["tools"]) == 1
    assert plan.request["tools"][0]["function"]["name"] == "get_weather"
    assert plan.request["tool_choice"] == "auto"
    assert "1 tools" in plan.included


def test_plan_with_generation_params() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    spec = _spec(temperature=0.5, max_output_tokens=100, top_p=0.9, seed=42)
    plan = adapter.plan(ctx, spec)

    assert plan.request["temperature"] == 0.5
    assert plan.request["max_tokens"] == 100
    assert plan.request["top_p"] == 0.9
    assert plan.request["seed"] == 42


def test_plan_with_response_modalities() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Generate speech output.")
    spec = _spec(response_modalities=("audio",))
    plan = adapter.plan(ctx, spec)

    assert plan.request["modalities"] == ["audio"]
    assert "response_modalities" in plan.included


def test_plan_with_tool_call_history() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("What's the weather?")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="tool_call", tool_call_id="call_1", tool_name="get_weather", tool_args={"city": "Tokyo"}),
            ),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="call_1", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]

    # Assistant message with tool_calls
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}

    # Tool result message
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_1"


def test_plan_multimodal_image() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"fake-png-bytes", media_type="image/png", kind="image")
    ctx = ctx.user(
        [
            Part(type="text", text="What's in this image?"),
            Part(type="image", blob=ref),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]

    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "What's in this image?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_plan_with_file_part() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user(
        [
            Part(type="text", text="Summarize this file."),
            Part(type="file", provider_raw={"file_id": "file_123", "filename": "notebook.pdf"}),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Summarize this file."}
    assert content[1] == {"type": "file", "file": {"file_id": "file_123", "filename": "notebook.pdf"}}


def test_plan_with_file_part_from_text_file_id() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user([Part(type="file", text="file_from_text")])

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content == [{"type": "file", "file": {"file_id": "file_from_text"}}]


def test_plan_with_file_blob_part_uses_default_filename() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"fake-document-bytes", media_type="application/pdf", kind="document")
    ctx = ctx.user([Part(type="file", blob=ref)])

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content == [
        {
            "type": "file",
            "file": {"file_data": "ZmFrZS1kb2N1bWVudC1ieXRlcw==", "filename": "upload.bin"},
        }
    ]


def test_plan_extra_body_deep_merged() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    spec = _spec(extra_body={"stream": True, "logprobs": True})
    plan = adapter.plan(ctx, spec)

    assert plan.request["stream"] is True
    assert plan.request["logprobs"] is True


def test_plan_transport_overrides() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        extra_headers={"X-Trace-Id": "trace-1"},
        extra_query={"api-version": "2025-01-01"},
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["extra_headers"] == {"X-Trace-Id": "trace-1"}
    assert plan.request["extra_query"] == {"api-version": "2025-01-01"}
    assert "extra_headers" in plan.included
    assert "extra_query" in plan.included


def test_plan_assistant_text_and_tool_call() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Tell me the weather")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="text", text="Let me check..."),
                Part(type="tool_call", tool_call_id="call_1", tool_name="weather", tool_args={"city": "NYC"}),
            ),
        )
    )

    plan = adapter.plan(ctx, _spec())
    msg = plan.request["messages"][1]
    assert msg["content"] == "Let me check..."
    assert len(msg["tool_calls"]) == 1


def test_plan_assistant_tool_call_with_reasoning_content_roundtrip() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Compare Tokyo and London")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(
                    type="thinking",
                    provider_raw={"reasoning_content": "Internal reasoning trace"},
                ),
                Part(
                    type="tool_call",
                    tool_call_id="call_1",
                    tool_name="get_weather",
                    tool_args={"city": "Tokyo"},
                ),
            ),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="call_1", tool_output={"city": "Tokyo"}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assistant_msg = plan.request["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_content"] == "Internal reasoning trace"
    assert assistant_msg["tool_calls"][0]["id"] == "call_1"


def test_plan_assistant_thinking_text_fallback_maps_to_reasoning_content() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Think first")
    ctx = ctx.append(Message(role="assistant", parts=(Part(type="thinking", text="Hidden thoughts"),)))

    plan = adapter.plan(ctx, _spec())
    assistant_msg = plan.request["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["reasoning_content"] == "Hidden thoughts"


def test_plan_assistant_thinking_provider_raw_reasoning_fields() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Think")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(
                    type="thinking",
                    provider_raw={"reasoning": "internal", "reasoning_details": [{"type": "summary"}]},
                ),
            ),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assistant_msg = plan.request["messages"][1]
    assert assistant_msg["reasoning"] == "internal"
    assert assistant_msg["reasoning_details"] == [{"type": "summary"}]


def test_plan_multiple_tool_results() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Compare weather")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="tool_call", tool_call_id="a", tool_name="weather", tool_args={"city": "NYC"}),
                Part(type="tool_call", tool_call_id="b", tool_name="weather", tool_args={"city": "LA"}),
            ),
        )
    )
    # Two tool results in one message -> should produce two OpenAI messages
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(
                Part(type="tool_result", tool_call_id="a", tool_output="sunny"),
                Part(type="tool_result", tool_call_id="b", tool_output="cloudy"),
            ),
        )
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "a"
    assert tool_msgs[1]["tool_call_id"] == "b"


def test_plan_omits_invalid_tool_call_without_id() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Call tool")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_name="weather", tool_args={"city": "Tokyo"}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assert plan.request["messages"] == [{"role": "user", "content": "Call tool"}]
    assert any(item.description == "context.messages[1]" for item in plan.excluded)
    assert any("context.messages[1]" in error for error in plan.errors)


def test_plan_reports_invalid_user_message_without_serializable_content() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().append(Message(role="user", parts=(Part(type="image"),)))

    plan = adapter.plan(ctx, _spec())

    assert plan.request["messages"] == []
    assert any(item.description == "context.messages[0]" for item in plan.excluded)
    assert any("no messages" in error for error in plan.errors)


def test_plan_tool_message_without_tool_result_reports_no_payload() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().append(Message(role="tool", parts=(Part(type="text", text="invalid tool role part"),)))

    plan = adapter.plan(ctx, _spec())
    assert plan.request["messages"] == []
    assert any(item.description == "context.messages[0].parts[0]" for item in plan.excluded)
    assert any(item.description == "context.messages[0]" for item in plan.excluded)
    assert any("produced no Chat Completions payload" in error for error in plan.errors)


def test_plan_reports_invalid_part_with_part_level_excluded() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user([Part(type="text", text="Keep"), Part(type="tool_call", tool_name="weather")])

    plan = adapter.plan(ctx, _spec())
    assert plan.request["messages"] == [{"role": "user", "content": "Keep"}]
    assert any(item.description == "context.messages[0].parts[1]" for item in plan.excluded)


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
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"choices": []}, spec=bad_spec)


def test_ingest_text_response() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 2, "audio_tokens": 1},
            "completion_tokens_details": {
                "reasoning_tokens": 3,
                "audio_tokens": 4,
                "accepted_prediction_tokens": 6,
                "rejected_prediction_tokens": 7,
            },
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx) == 2
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].text == "Hi there!"
    assert last.id == "chatcmpl-123"
    assert last.provider == "openai"

    assert len(ctx.usage_log) == 1
    assert ctx.usage_log[0].input_tokens == 10
    assert ctx.usage_log[0].output_tokens == 5
    assert ctx.usage_log[0].total_tokens == 15
    assert ctx.usage_log[0].provider_usage["prompt_tokens"] == 10
    assert ctx.usage_log[0].provider_usage["completion_tokens"] == 5
    assert ctx.usage_log[0].provider_usage["total_tokens"] == 15
    assert ctx.usage_log[0].provider_usage["prompt_tokens_details"] == {"cached_tokens": 2, "audio_tokens": 1}
    assert ctx.usage_log[0].provider_usage["completion_tokens_details"] == {
        "reasoning_tokens": 3,
        "audio_tokens": 4,
        "accepted_prediction_tokens": 6,
        "rejected_prediction_tokens": 7,
    }


def test_ingest_tool_call_response() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "id": "chatcmpl-456",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 1
    tc = last.parts[0]
    assert tc.type == "tool_call"
    assert tc.tool_call_id == "call_abc"
    assert tc.tool_name == "get_weather"
    assert tc.tool_args == {"city": "Tokyo"}


def test_ingest_tool_call_response_with_object_arguments() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "id": "chatcmpl-456b",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_obj",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Tokyo", "unit": "celsius"},
                            },
                        }
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "tool_call"
    assert last.parts[0].tool_args == {"city": "Tokyo", "unit": "celsius"}


def test_ingest_text_and_tool_calls() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "chatcmpl-789",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Sure, let me check.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "test"}'},
                        }
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "text"
    assert last.parts[0].text == "Sure, let me check."
    assert last.parts[1].type == "tool_call"


def test_ingest_content_blocks_text_and_image() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Describe this.")

    response = {
        "id": "chatcmpl-blocks",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here is the result."},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "text"
    assert last.parts[0].text == "Here is the result."
    assert last.parts[1].type == "image"
    assert last.parts[1].provider_raw == {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}


def test_ingest_reasoning_content_with_tool_calls() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Use tools")

    response = {
        "id": "chatcmpl-reasoning",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "reasoning trace",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "tokyo"}'},
                        }
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "thinking"
    assert last.parts[0].text == "reasoning trace"
    assert last.parts[0].provider_raw == {"reasoning_content": "reasoning trace"}
    assert last.parts[1].type == "tool_call"


def test_ingest_reasoning_fallback_and_malformed_tool_calls() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "chatcmpl-456",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning": "fallback reasoning",
                    "tool_calls": [
                        123,
                        {
                            "id": "call_1",
                            "function": {"name": "weather", "arguments": "[]"},
                        },
                    ],
                }
            }
        ],
    }

    updated = adapter.ingest(ctx, response, spec=_spec())
    assistant = updated.last(role="assistant")
    assert assistant is not None
    assert [part.type for part in assistant.parts] == ["text", "thinking", "tool_call"]
    assert assistant.parts[1].text == "fallback reasoning"
    assert assistant.parts[2].tool_args == {"_raw": "[]"}


def test_ingest_empty_choices() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")

    response = {"id": "chatcmpl-empty", "choices": []}
    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2) == 1  # No new message added


def test_ingest_malformed_tool_args() -> None:
    adapter = OpenAIChatCompletionsAdapter()
    ctx = Context().user("Hello")

    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {"name": "test", "arguments": "not-valid-json"},
                        }
                    ],
                },
            }
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].tool_args == {"_raw": "not-valid-json"}
