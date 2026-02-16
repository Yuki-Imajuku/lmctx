import json
from collections.abc import Mapping

import pytest

from lmctx import Context, Message, Part, ToolSpecification
from lmctx.adapters import BedrockConverseAdapter
from lmctx.blobs import InMemoryBlobStore
from lmctx.plan import LmctxAdapter
from lmctx.spec import Instructions, RunSpec


def _as_str_object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    return {str(key): val for key, val in value.items()}


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {
        "provider": "bedrock",
        "endpoint": "converse",
        "model": "anthropic.claude-sonnet-4-5-20250929-v2:0",
    }
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Protocol Conformance
# =============================================================================


def test_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = BedrockConverseAdapter()
    assert isinstance(adapter, LmctxAdapter)


# =============================================================================
# plan()
# =============================================================================


def test_plan_simple_text() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello").assistant("Hi").user("How are you?")
    plan = adapter.plan(ctx, _spec())

    assert plan.request["modelId"] == "anthropic.claude-sonnet-4-5-20250929-v2:0"
    messages = plan.request["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"text": "Hello"}]
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
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_plan_with_system_instructions() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(instructions=Instructions(system="You are helpful."))
    plan = adapter.plan(ctx, spec)

    system = plan.request["system"]
    assert system == [{"text": "You are helpful."}]
    assert "system instruction" in plan.included


def test_plan_system_message_in_context() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context()
    ctx = ctx.append(Message(role="system", parts=(Part(type="text", text="From context."),)))
    ctx = ctx.user("Hello")

    spec = _spec(instructions=Instructions(system="From RunSpec."))
    plan = adapter.plan(ctx, spec)

    system_texts = [s["text"] for s in plan.request["system"]]
    assert "From RunSpec." in system_texts
    assert "From context." in system_texts


def test_plan_inference_config() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(temperature=0.5, max_output_tokens=100, top_p=0.9)
    plan = adapter.plan(ctx, spec)

    config = plan.request["inferenceConfig"]
    assert config["temperature"] == 0.5
    assert config["maxTokens"] == 100
    assert config["topP"] == 0.9


def test_plan_with_tools() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,))
    plan = adapter.plan(ctx, spec)

    tool_config = plan.request["toolConfig"]
    tools = tool_config["tools"]
    assert len(tools) == 1
    assert tools[0]["toolSpec"]["name"] == "get_weather"
    assert tools[0]["toolSpec"]["inputSchema"]["json"] == tool.input_schema


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        pytest.param("auto", {"auto": {}}, id="string-auto"),
        pytest.param("required", {"any": {}}, id="string-required"),
        pytest.param({"type": "auto"}, {"auto": {}}, id="openai-style-auto"),
        pytest.param(
            {"type": "function", "function": {"name": "get_weather"}},
            {"tool": {"name": "get_weather"}},
            id="openai-style-function",
        ),
        pytest.param({"tool": {"name": "get_weather"}}, {"tool": {"name": "get_weather"}}, id="bedrock-native-tool"),
    ],
)
def test_plan_tool_choice_mapped_to_tool_config(tool_choice: object, expected: dict[str, object]) -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,), tool_choice=tool_choice)
    plan = adapter.plan(ctx, spec)

    assert plan.request["toolConfig"]["toolChoice"] == expected
    assert "tool_choice" in plan.included


def test_plan_tool_choice_name_shortcut_maps_to_tool_choice_tool() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,), tool_choice={"name": "get_weather"})
    plan = adapter.plan(ctx, spec)

    assert plan.request["toolConfig"]["toolChoice"] == {"tool": {"name": "get_weather"}}
    assert "tool_choice" in plan.included


def test_plan_tool_choice_none_reported_as_excluded() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(tool_choice="none")
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "tool_choice" for item in plan.excluded)


@pytest.mark.parametrize(
    ("tool_choice", "reason_fragment"),
    [
        pytest.param("manual", "unsupported string tool_choice value", id="unsupported-string"),
        pytest.param(
            {"type": "function", "function": {}},
            "requires function.name",
            id="function-missing-name",
        ),
        pytest.param({"mode": "auto"}, "format is not recognized", id="unknown-mapping-shape"),
        pytest.param(123, "must be a string or mapping", id="non-mapping"),
    ],
)
def test_plan_invalid_tool_choice_reported_as_excluded(tool_choice: object, reason_fragment: str) -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(tool_choice=tool_choice)
    plan = adapter.plan(ctx, spec)

    excluded_item = next(item for item in plan.excluded if item.description == "tool_choice")
    assert reason_fragment in excluded_item.reason


def test_plan_with_response_schema() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Return JSON.")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    spec = _spec(response_schema=schema)
    plan = adapter.plan(ctx, spec)

    output_config = _as_str_object_dict(plan.request["outputConfig"])
    text_format = _as_str_object_dict(output_config["textFormat"])
    structure = _as_str_object_dict(text_format["structure"])
    json_schema = _as_str_object_dict(structure["jsonSchema"])
    assert text_format["type"] == "json_schema"
    assert json.loads(str(json_schema["schema"])) == {
        **schema,
        "additionalProperties": False,
    }
    assert "response_schema" in plan.included


def test_plan_with_openai_style_response_schema_wrapper() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Return JSON.")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    spec = _spec(
        response_schema={
            "name": "answer_schema",
            "description": "Schema for answer payload",
            "schema": schema,
            "strict": True,
        }
    )
    plan = adapter.plan(ctx, spec)

    output_config = _as_str_object_dict(plan.request["outputConfig"])
    text_format = _as_str_object_dict(output_config["textFormat"])
    structure = _as_str_object_dict(text_format["structure"])
    json_schema = _as_str_object_dict(structure["jsonSchema"])
    assert text_format["type"] == "json_schema"
    assert json_schema["name"] == "answer_schema"
    assert json_schema["description"] == "Schema for answer payload"
    assert json.loads(str(json_schema["schema"])) == {
        **schema,
        "additionalProperties": False,
    }


def test_plan_response_schema_closes_nested_object_schemas() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Return JSON.")
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"y": {"type": "integer"}},
                    "required": ["y"],
                },
            },
        },
        "required": ["outer", "items"],
    }
    spec = _spec(response_schema=schema)
    plan = adapter.plan(ctx, spec)

    output_config = _as_str_object_dict(plan.request["outputConfig"])
    text_format = _as_str_object_dict(output_config["textFormat"])
    structure = _as_str_object_dict(text_format["structure"])
    json_schema = _as_str_object_dict(structure["jsonSchema"])
    parsed = json.loads(str(json_schema["schema"]))
    assert parsed["additionalProperties"] is False
    assert parsed["properties"]["outer"]["additionalProperties"] is False
    assert parsed["properties"]["items"]["items"]["additionalProperties"] is False


def test_plan_tool_call_and_result() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("What's the weather?")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_call_id="tu_1", tool_name="get_weather", tool_args={"city": "Tokyo"}),),
        )
    )
    ctx = ctx.append(
        Message(
            role="tool",
            parts=(Part(type="tool_result", tool_call_id="tu_1", tool_output={"temp": 22}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    messages = plan.request["messages"]

    # user, assistant (toolUse), user (toolResult)
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["toolUse"]["toolUseId"] == "tu_1"
    assert messages[1]["content"][0]["toolUse"]["name"] == "get_weather"

    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["toolResult"]["toolUseId"] == "tu_1"


def test_plan_multimodal_image() -> None:
    adapter = BedrockConverseAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    image_data = b"fake-png-bytes"
    ref = store.put_blob(image_data, media_type="image/png", kind="image")
    ctx = ctx.user(
        [
            Part(type="text", text="Describe this."),
            Part(type="image", blob=ref),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content[0] == {"text": "Describe this."}
    assert content[1]["image"]["format"] == "png"
    assert content[1]["image"]["source"]["bytes"] == image_data


def test_plan_multimodal_image_media_type_is_case_insensitive() -> None:
    adapter = BedrockConverseAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    image_data = b"fake-jpeg-bytes"
    ref = store.put_blob(image_data, media_type="image/JPEG", kind="image")
    ctx = ctx.user([Part(type="image", blob=ref)])

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content[0]["image"]["format"] == "jpeg"


def test_plan_with_file_part() -> None:
    adapter = BedrockConverseAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    pdf_data = b"%PDF-1.4 fake content"
    ref = store.put_blob(pdf_data, media_type="application/pdf", kind="document")
    ctx = ctx.user(
        [
            Part(type="text", text="Summarize this PDF."),
            Part(type="file", blob=ref, provider_raw={"name": "notebook_pdf"}),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    content = plan.request["messages"][0]["content"]
    assert content[0] == {"text": "Summarize this PDF."}
    assert content[1]["document"]["format"] == "pdf"
    assert content[1]["document"]["name"] == "notebook_pdf"
    assert content[1]["document"]["source"]["bytes"] == pdf_data


def test_plan_consecutive_user_messages_merged() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("First").user("Second")
    plan = adapter.plan(ctx, _spec())

    messages = plan.request["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 2


def test_plan_extra_body_deep_merge() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        max_output_tokens=120,
        temperature=0.3,
        extra_body={"inferenceConfig": {"topK": 50}},
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["inferenceConfig"]["maxTokens"] == 120
    assert plan.request["inferenceConfig"]["temperature"] == 0.3
    assert plan.request["inferenceConfig"]["topK"] == 50


def test_plan_transport_overrides_reported_as_excluded() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(extra_headers={"X-Test": "1"}, extra_query={"region": "us-east-1"})
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "extra_headers" for item in plan.excluded)
    assert any(item.description == "extra_query" for item in plan.excluded)


def test_plan_response_modalities_reported_as_excluded() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Generate an image")
    spec = _spec(response_modalities=("image",))
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "response_modalities" for item in plan.excluded)


def test_plan_seed_reported_as_excluded() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    spec = _spec(seed=42)
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "seed" for item in plan.excluded)


def test_plan_omits_invalid_tool_call_without_name() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Tool call")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_call_id="tu_1", tool_args={"city": "Tokyo"}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assert len(plan.request["messages"]) == 1


def test_plan_reports_invalid_part_with_part_level_excluded() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user([Part(type="text", text="Keep"), Part(type="image")])

    plan = adapter.plan(ctx, _spec())
    assert plan.request["messages"] == [{"role": "user", "content": [{"text": "Keep"}]}]
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
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"output": {"message": {"content": []}}}, spec=bad_spec)


def test_ingest_text_response() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hi there!"}],
            }
        },
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
            "cacheReadInputTokens": 2,
            "cacheWriteInputTokens": 3,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 2}],
        },
        "stopReason": "end_turn",
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx) == 2
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].text == "Hi there!"
    assert last.provider == "bedrock"

    assert len(ctx.usage_log) == 1
    assert ctx.usage_log[0].input_tokens == 10
    assert ctx.usage_log[0].output_tokens == 5
    assert ctx.usage_log[0].total_tokens == 15
    assert ctx.usage_log[0].provider_usage["inputTokens"] == 10
    assert ctx.usage_log[0].provider_usage["outputTokens"] == 5
    assert ctx.usage_log[0].provider_usage["totalTokens"] == 15
    assert ctx.usage_log[0].provider_usage["cacheReadInputTokens"] == 2
    assert ctx.usage_log[0].provider_usage["cacheWriteInputTokens"] == 3
    assert ctx.usage_log[0].provider_usage["cacheDetails"] == ({"ttl": "5m", "inputTokens": 2},)


def test_ingest_usage_keeps_only_integer_token_fields() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hi there!"}],
            }
        },
        "usage": {
            "inputTokens": "10",
            "outputTokens": 5.5,
            "totalTokens": None,
            "cacheReadInputTokens": 2,
        },
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx2.usage_log) == 1
    usage = ctx2.usage_log[0]
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.provider_usage["cacheReadInputTokens"] == 2


def test_ingest_usage_derives_total_when_missing() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hi there!"}],
            }
        },
        "usage": {
            "inputTokens": 20,
            "outputTokens": 15,
        },
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx2.usage_log) == 1
    usage = ctx2.usage_log[0]
    assert usage.input_tokens == 20
    assert usage.output_tokens == 15
    assert usage.total_tokens == 35


def test_ingest_tool_use_response() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Let me check."},
                    {
                        "toolUse": {
                            "toolUseId": "tu_abc",
                            "name": "get_weather",
                            "input": {"city": "Tokyo"},
                        }
                    },
                ],
            }
        },
        "usage": {"inputTokens": 20, "outputTokens": 15, "totalTokens": 35},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "text"
    assert last.parts[1].type == "tool_call"
    assert last.parts[1].tool_call_id == "tu_abc"
    assert last.parts[1].tool_name == "get_weather"
    assert last.parts[1].tool_args == {"city": "Tokyo"}


def test_ingest_empty_output() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Hello")

    response = {"output": {"message": {"role": "assistant", "content": []}}}
    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2) == 1


# =============================================================================
# Reasoning Content
# =============================================================================


def test_ingest_reasoning_content() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Think about this.")

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "Let me think step by step...",
                                "signature": "bedrock_sig_123",
                            }
                        }
                    },
                    {"text": "The answer is 42."},
                ],
            }
        },
        "usage": {"inputTokens": 20, "outputTokens": 40, "totalTokens": 60},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2

    thinking = last.parts[0]
    assert thinking.type == "thinking"
    assert thinking.text == "Let me think step by step..."
    assert thinking.provider_raw is not None
    reasoning_content = _as_str_object_dict(thinking.provider_raw["reasoningContent"])
    reasoning_text = _as_str_object_dict(reasoning_content["reasoningText"])
    assert reasoning_text["signature"] == "bedrock_sig_123"

    assert last.parts[1].type == "text"
    assert last.parts[1].text == "The answer is 42."


def test_ingest_redacted_reasoning_content() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Sensitive topic.")

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"redactedContent": "encrypted_bytes"}},
                    {"text": "Response."},
                ],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 15, "totalTokens": 25},
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None

    thinking = last.parts[0]
    assert thinking.type == "thinking"
    assert thinking.text is None  # No text for redacted reasoning.
    assert thinking.provider_raw is not None
    reasoning_content = _as_str_object_dict(thinking.provider_raw["reasoningContent"])
    assert "redactedContent" in reasoning_content


def test_plan_reasoning_content_roundtrip() -> None:
    adapter = BedrockConverseAdapter()
    ctx = Context().user("Think about this.")

    # Simulate: ingest reasoning content with signature.
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "Reasoning...",
                                "signature": "sig_abc",
                            }
                        }
                    },
                    {"text": "Answer."},
                ],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }
    ctx = adapter.ingest(ctx, response, spec=_spec())

    # Plan the next request with the reasoning content in context.
    ctx = ctx.user("Follow up.")
    plan = adapter.plan(ctx, _spec())

    messages = plan.request["messages"]
    # messages[0] = user "Think about this."
    # messages[1] = assistant with reasoning + text
    # messages[2] = user "Follow up."
    assistant_content = messages[1]["content"]
    assert assistant_content[0]["reasoningContent"]["reasoningText"]["text"] == "Reasoning..."
    assert assistant_content[0]["reasoningContent"]["reasoningText"]["signature"] == "sig_abc"
