import pytest

from lmctx import Context, Message, Part, ToolSpecification
from lmctx.adapters import AnthropicMessagesAdapter
from lmctx.blobs import InMemoryBlobStore
from lmctx.plan import LmctxAdapter
from lmctx.spec import Instructions, RunSpec


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {
        "provider": "anthropic",
        "endpoint": "messages.create",
        "model": "claude-sonnet-4-5-20250929",
    }
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Protocol Conformance
# =============================================================================


def test_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = AnthropicMessagesAdapter()
    assert isinstance(adapter, LmctxAdapter)


# =============================================================================
# plan()
# =============================================================================


def test_plan_simple_text() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello").assistant("Hi").user("How are you?")
    plan = adapter.plan(ctx, _spec())

    assert plan.request["model"] == "claude-sonnet-4-5-20250929"
    assert plan.request["max_tokens"] == 4096  # default

    messages = plan.request["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello"}]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_plan_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_plan_with_system_instructions() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(instructions=Instructions(system="You are helpful."))
    plan = adapter.plan(ctx, spec)

    assert plan.request["system"] == "You are helpful."
    assert "system instruction" in plan.included


def test_plan_system_message_in_context() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context()
    ctx = ctx.append(Message(role="system", parts=(Part(type="text", text="From context."),)))
    ctx = ctx.user("Hello")

    spec = _spec(instructions=Instructions(system="From RunSpec."))
    plan = adapter.plan(ctx, spec)

    # Both should be merged into system
    assert "From RunSpec." in plan.request["system"]
    assert "From context." in plan.request["system"]

    # Messages should not contain system role
    for msg in plan.request["messages"]:
        assert msg["role"] in ("user", "assistant")


def test_plan_with_max_output_tokens() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(max_output_tokens=2048)
    plan = adapter.plan(ctx, spec)

    assert plan.request["max_tokens"] == 2048
    assert not plan.warnings  # No warning when explicitly set


def test_plan_default_max_tokens_warning() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    plan = adapter.plan(ctx, _spec())

    assert any("max_output_tokens" in w for w in plan.warnings)


def test_plan_warns_on_response_modalities() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(response_modalities=("image",))
    plan = adapter.plan(ctx, spec)

    assert any("response_modalities" in warning for warning in plan.warnings)


def test_plan_reports_seed_as_excluded() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(seed=42)
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "seed" for item in plan.excluded)


def test_plan_with_response_schema() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Return JSON.")
    schema = {
        "type": "object",
        "properties": {
            "country": {"type": "string"},
            "capital": {"type": "string"},
        },
        "required": ["country", "capital"],
    }
    spec = _spec(response_schema=schema)
    plan = adapter.plan(ctx, spec)

    assert plan.request["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "capital": {"type": "string"},
                },
                "required": ["country", "capital"],
                "additionalProperties": False,
            },
        }
    }
    assert "response_schema" in plan.included


def test_plan_with_response_schema_closes_nested_object_schemas() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Return JSON.")
    schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
        "required": ["city"],
    }
    spec = _spec(response_schema=schema)

    plan = adapter.plan(ctx, spec)
    json_schema = plan.request["output_config"]["format"]["schema"]

    assert json_schema["additionalProperties"] is False
    assert json_schema["properties"]["city"]["additionalProperties"] is False


def test_plan_with_tools() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,))
    plan = adapter.plan(ctx, spec)

    assert len(plan.request["tools"]) == 1
    assert plan.request["tools"][0]["name"] == "get_weather"
    assert plan.request["tools"][0]["input_schema"] == tool.input_schema


def test_plan_tool_call_and_result() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("What's the weather?")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="tool_call", tool_call_id="toolu_1", tool_name="get_weather", tool_args={"city": "Tokyo"}),
            ),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="toolu_1", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]

    # user, assistant (tool_use), user (tool_result)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["id"] == "toolu_1"

    # tool_result should be in a user message
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["type"] == "tool_result"
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_plan_omits_invalid_tool_call_without_name() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Call tool")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_call_id="toolu_1", tool_args={"city": "Tokyo"}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]
    assert len(messages) == 1


def test_plan_reports_invalid_message_without_supported_parts() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().append(Message(role="assistant", parts=(Part(type="image"),)))

    plan = adapter.plan(ctx, _spec())
    assert plan.request["messages"] == []
    assert any(item.description == "context.messages[0]" for item in plan.excluded)
    assert any("requires at least one valid user/assistant message" in error for error in plan.errors)


def test_plan_multimodal_image() -> None:
    adapter = AnthropicMessagesAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"fake-png-bytes", media_type="image/png", kind="image")
    ctx = ctx.user(
        [
            Part(type="text", text="Describe this image."),
            Part(type="image", blob=ref),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]

    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Describe this image."}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"


def test_plan_with_file_part() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user(
        [
            Part(type="text", text="Summarize this file."),
            Part(type="file", provider_raw={"file_id": "file_123", "title": "Notebook PDF"}),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Summarize this file."}
    assert content[1]["type"] == "document"
    assert content[1]["source"] == {"type": "file", "file_id": "file_123"}
    assert content[1]["title"] == "Notebook PDF"


def test_plan_with_file_blob_part() -> None:
    adapter = AnthropicMessagesAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"%PDF-1.4 fake content", media_type="application/pdf", kind="document")
    ctx = ctx.user([Part(type="file", blob=ref)])

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content == [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQgZmFrZSBjb250ZW50",
            },
        }
    ]


def test_plan_reports_invalid_part_with_part_level_excluded() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user([Part(type="text", text="Keep"), Part(type="image")])

    plan = adapter.plan(ctx, _spec())
    assert plan.request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Keep"}]}]
    assert any(item.description == "context.messages[0].parts[1]" for item in plan.excluded)


def test_plan_alternating_role_error() -> None:
    adapter = AnthropicMessagesAdapter()
    # Two consecutive user messages without assistant in between
    ctx = Context().user("First").user("Second")
    plan = adapter.plan(ctx, _spec())

    # After merging, consecutive user messages are merged, so no error
    messages = plan.request["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 2


def test_plan_first_message_must_be_user() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context()
    ctx = ctx.append(Message(role="assistant", parts=(Part(type="text", text="Hi"),)))

    plan = adapter.plan(ctx, _spec())
    assert any("first message" in e.lower() for e in plan.errors)


def test_plan_extra_body() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(extra_body={"metadata": {"user_id": "u123"}})
    plan = adapter.plan(ctx, spec)

    assert plan.request["metadata"] == {"user_id": "u123"}


def test_plan_transport_overrides() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        extra_headers={"anthropic-beta": "output-128k-2025-02-19"},
        extra_query={"api-version": "2025-01-01"},
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["extra_headers"] == {"anthropic-beta": "output-128k-2025-02-19"}
    assert plan.request["extra_query"] == {"api-version": "2025-01-01"}
    assert "extra_headers" in plan.included
    assert "extra_query" in plan.included


def test_plan_compaction_passthrough_configuration() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        extra_headers={"anthropic-beta": "context-management-2025-06-27"},
        extra_body={
            "context_management": {
                "edits": [
                    {
                        "type": "compact",
                        "trigger": "manual",
                    }
                ]
            }
        },
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["extra_headers"] == {"anthropic-beta": "context-management-2025-06-27"}
    assert plan.request["context_management"]["edits"][0]["type"] == "compact"
    assert plan.request["context_management"]["edits"][0]["trigger"] == "manual"


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
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"content": []}, spec=bad_spec)


def test_ingest_text_response() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
        "model": "claude-sonnet-4-5-20250929",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 2,
            "cache_creation": {"ephemeral_1h_input_tokens": 1, "ephemeral_5m_input_tokens": 2},
            "server_tool_use": {"web_search_requests": 4},
            "service_tier": "standard",
            "inference_geo": "us",
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx) == 2
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].text == "Hi there!"
    assert last.id == "msg_123"
    assert last.provider == "anthropic"

    assert len(ctx.usage_log) == 1
    assert ctx.usage_log[0].input_tokens == 10
    assert ctx.usage_log[0].output_tokens == 5
    assert ctx.usage_log[0].total_tokens == 15
    assert ctx.usage_log[0].provider_usage["input_tokens"] == 10
    assert ctx.usage_log[0].provider_usage["output_tokens"] == 5
    assert ctx.usage_log[0].provider_usage["cache_creation_input_tokens"] == 3
    assert ctx.usage_log[0].provider_usage["cache_read_input_tokens"] == 2
    assert ctx.usage_log[0].provider_usage["service_tier"] == "standard"
    assert ctx.usage_log[0].provider_usage["inference_geo"] == "us"


def test_ingest_usage_falls_back_to_iterations_for_compaction() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Compact context.")

    response = {
        "id": "msg_compact_usage",
        "content": [{"type": "compaction", "content": "opaque-compaction-payload"}],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "iterations": [
                {"type": "compaction", "input_tokens": 52133, "output_tokens": 234},
            ],
        },
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2.usage_log) == 1
    assert ctx2.usage_log[0].input_tokens == 52133
    assert ctx2.usage_log[0].output_tokens == 234
    assert ctx2.usage_log[0].total_tokens == 52367


def test_ingest_usage_preserves_explicit_total_tokens() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "msg_total",
        "content": [{"type": "text", "text": "Hi"}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 42,
        },
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert ctx2.usage_log[0].input_tokens == 10
    assert ctx2.usage_log[0].output_tokens == 5
    assert ctx2.usage_log[0].total_tokens == 42


def test_ingest_tool_use_response() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "id": "msg_456",
        "content": [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            },
        ],
        "usage": {"input_tokens": 20, "output_tokens": 15},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "text"
    assert last.parts[0].text == "Let me check the weather."
    assert last.parts[1].type == "tool_call"
    assert last.parts[1].tool_call_id == "toolu_abc"
    assert last.parts[1].tool_name == "get_weather"
    assert last.parts[1].tool_args == {"city": "Tokyo"}


def test_ingest_thinking_response() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Solve this problem.")

    response = {
        "id": "msg_789",
        "content": [
            {"type": "thinking", "thinking": "Let me reason about this..."},
            {"type": "text", "text": "The answer is 42."},
        ],
        "usage": {"input_tokens": 30, "output_tokens": 50},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "thinking"
    assert last.parts[0].text == "Let me reason about this..."
    assert last.parts[1].type == "text"
    assert last.parts[1].text == "The answer is 42."


def test_ingest_compaction_response() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Compact context.")
    response = {
        "id": "msg_compact",
        "content": [
            {"type": "compaction", "content": "opaque-compaction-payload"},
        ],
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "compaction"
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get(last.parts[0].blob) == b"opaque-compaction-payload"


def test_plan_compaction_roundtrip_from_blob() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Compact context.")
    response = {
        "id": "msg_compact",
        "content": [
            {"type": "compaction", "content": "opaque-compaction-payload"},
        ],
    }
    ctx = adapter.ingest(ctx, response, spec=_spec())
    ctx = ctx.user("follow up")

    plan = adapter.plan(ctx, _spec())
    assistant_msg = plan.request["messages"][1]
    assert assistant_msg["content"][0] == {"type": "compaction", "content": "opaque-compaction-payload"}


def test_ingest_empty_content() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Hello")

    response = {"id": "msg_empty", "content": []}
    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2) == 1  # No new message added


# =============================================================================
# Thinking Round-Trip
# =============================================================================


def test_ingest_thinking_preserves_signature() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Think hard.")

    response = {
        "id": "msg_think",
        "content": [
            {
                "type": "thinking",
                "thinking": "Step by step...",
                "signature": "WaUjzkypQ2mUEVM36O2T",
            },
            {"type": "text", "text": "The answer."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 30},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None

    thinking_part = last.parts[0]
    assert thinking_part.type == "thinking"
    assert thinking_part.text == "Step by step..."
    # provider_raw preserves the full block including signature.
    assert thinking_part.provider_raw is not None
    assert thinking_part.provider_raw["signature"] == "WaUjzkypQ2mUEVM36O2T"


def test_ingest_redacted_thinking() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Sensitive topic.")

    response = {
        "id": "msg_redact",
        "content": [
            {"type": "redacted_thinking", "data": "encrypted_base64_data"},
            {"type": "text", "text": "I cannot help with that."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2

    redacted = last.parts[0]
    assert redacted.type == "thinking"
    assert redacted.text is None  # No readable text for redacted thinking.
    assert redacted.provider_raw is not None
    assert redacted.provider_raw["type"] == "redacted_thinking"
    assert redacted.provider_raw["data"] == "encrypted_base64_data"


def test_plan_thinking_roundtrip_preserves_signature() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Think hard.")

    # Simulate: ingest a response with thinking + signature.
    response = {
        "id": "msg_1",
        "content": [
            {
                "type": "thinking",
                "thinking": "Step by step...",
                "signature": "abc123signature",
            },
            {"type": "text", "text": "Answer."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    ctx = adapter.ingest(ctx, response, spec=_spec())

    # Now add a follow-up and plan the next request.
    ctx = ctx.user("Follow up.")
    plan = adapter.plan(ctx, _spec())

    messages = plan.request["messages"]
    # messages[0] = user "Think hard."
    # messages[1] = assistant with thinking + text
    # messages[2] = user "Follow up."
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"

    # The thinking block should preserve the signature via provider_raw.
    thinking_block = assistant_msg["content"][0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["thinking"] == "Step by step..."
    assert thinking_block["signature"] == "abc123signature"


def test_plan_redacted_thinking_roundtrip() -> None:
    adapter = AnthropicMessagesAdapter()
    ctx = Context().user("Topic.")

    # Simulate: ingest a response with redacted thinking.
    response = {
        "id": "msg_2",
        "content": [
            {"type": "redacted_thinking", "data": "encrypted_data"},
            {"type": "text", "text": "Response."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }
    ctx = adapter.ingest(ctx, response, spec=_spec())

    ctx = ctx.user("Continue.")
    plan = adapter.plan(ctx, _spec())

    messages = plan.request["messages"]
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"

    # Redacted thinking should be preserved via provider_raw.
    redacted_block = assistant_msg["content"][0]
    assert redacted_block["type"] == "redacted_thinking"
    assert redacted_block["data"] == "encrypted_data"
