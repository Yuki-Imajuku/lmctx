import pytest

from lmctx import Context, Cursor, Message, Part, ToolSpecification
from lmctx.adapters import OpenAIResponsesAdapter, OpenAIResponsesCompactAdapter
from lmctx.blobs import BlobReference, InMemoryBlobStore
from lmctx.plan import LmctxAdapter
from lmctx.spec import Instructions, RunSpec


def _spec(**kwargs: object) -> RunSpec:
    defaults: dict[str, object] = {
        "provider": "openai",
        "endpoint": "responses.create",
        "model": "gpt-4o",
    }
    defaults.update(kwargs)
    return RunSpec(**defaults)  # type: ignore[arg-type]


# =============================================================================
# Protocol Conformance
# =============================================================================


def test_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = OpenAIResponsesAdapter()
    assert isinstance(adapter, LmctxAdapter)


def test_compact_conforms_to_lmctx_adapter_protocol() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    assert isinstance(adapter, LmctxAdapter)


# =============================================================================
# plan()
# =============================================================================


def test_plan_simple_text() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello").assistant("Hi").user("How are you?")
    plan = adapter.plan(ctx, _spec())

    assert plan.request["model"] == "gpt-4o"
    input_items = plan.request["input"]
    assert len(input_items) == 3
    assert input_items[0] == {"role": "user", "content": "Hello"}
    assert input_items[1]["type"] == "message"
    assert input_items[1]["role"] == "assistant"
    assert input_items[1]["content"] == [{"type": "output_text", "text": "Hi"}]
    assert input_items[2] == {"role": "user", "content": "How are you?"}


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_plan_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_plan_with_instructions() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(instructions=Instructions(system="You are helpful.", developer="Be concise."))
    plan = adapter.plan(ctx, spec)

    assert plan.request["instructions"] == "You are helpful.\n\nBe concise."
    assert "system instruction" in plan.included
    assert "developer instruction" in plan.included


def test_plan_system_message_in_context() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context()
    ctx = ctx.append(Message(role="system", parts=(Part(type="text", text="From context."),)))
    ctx = ctx.user("Hello")

    spec = _spec(instructions=Instructions(system="From RunSpec."))
    plan = adapter.plan(ctx, spec)

    # RunSpec instructions go in `instructions`, Context system goes in `input`.
    assert plan.request["instructions"] == "From RunSpec."
    input_items = plan.request["input"]
    assert input_items[0] == {"role": "system", "content": "From context."}
    assert input_items[1] == {"role": "user", "content": "Hello"}


def test_plan_tools_flat_format() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("What's the weather?")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(tools=(tool,))
    plan = adapter.plan(ctx, spec)

    tools = plan.request["tools"]
    assert len(tools) == 1
    # Responses API uses flat structure (name at top level, not nested in function).
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == "get_weather"
    assert tools[0]["parameters"] == tool.input_schema
    assert "function" not in tools[0]


def test_plan_tool_call_and_result() -> None:
    adapter = OpenAIResponsesAdapter()
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
    input_items = plan.request["input"]

    # user, function_call, function_call_output
    assert input_items[0]["role"] == "user"
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["call_id"] == "call_1"
    assert input_items[1]["name"] == "get_weather"
    assert input_items[2]["type"] == "function_call_output"
    assert input_items[2]["call_id"] == "call_1"


def test_plan_multimodal_image() -> None:
    adapter = OpenAIResponsesAdapter()
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
    content = plan.request["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Describe this."}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_plan_with_file_part() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user(
        [
            Part(type="text", text="Summarize this file."),
            Part(type="file", provider_raw={"file_id": "file_123", "filename": "notebook.pdf"}),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    content = plan.request["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Summarize this file."}
    assert content[1] == {"type": "input_file", "file_id": "file_123"}


def test_plan_with_file_data_and_filename() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user(
        [
            Part(type="text", text="Summarize this file."),
            Part(type="file", provider_raw={"file_data": "dGVzdA==", "filename": "notebook.pdf"}),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    content = plan.request["input"][0]["content"]
    assert content[1] == {"type": "input_file", "file_data": "dGVzdA==", "filename": "notebook.pdf"}


def test_plan_with_file_blob_part() -> None:
    adapter = OpenAIResponsesAdapter()
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)

    ref = store.put(b"%PDF-1.4 fake content", media_type="application/pdf", kind="document")
    ctx = ctx.user([Part(type="file", blob=ref)])

    plan = adapter.plan(ctx, _spec())
    content = plan.request["input"][0]["content"]
    assert content == [
        {
            "type": "input_file",
            "file_data": "JVBERi0xLjQgZmFrZSBjb250ZW50",
            "filename": "upload.pdf",
        }
    ]


def test_plan_reports_invalid_part_with_part_level_excluded() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user(
        [
            Part(type="text", text="Keep this text."),
            Part(type="image"),
        ]
    )

    plan = adapter.plan(ctx, _spec())
    assert plan.request["input"] == [{"role": "user", "content": "Keep this text."}]
    assert any(item.description == "context.messages[0].parts[1]" for item in plan.excluded)


def test_plan_generation_params() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(temperature=0.5, max_output_tokens=100, top_p=0.9)
    plan = adapter.plan(ctx, spec)

    assert plan.request["temperature"] == 0.5
    assert plan.request["max_output_tokens"] == 100
    assert plan.request["top_p"] == 0.9


def test_plan_reports_seed_as_excluded() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(seed=42)
    plan = adapter.plan(ctx, spec)

    assert any(item.description == "seed" for item in plan.excluded)


def test_plan_with_response_modalities() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Generate an image.")
    spec = _spec(response_modalities=("image",))
    plan = adapter.plan(ctx, spec)

    assert plan.request["modalities"] == ["image"]
    assert "response_modalities" in plan.included


def test_plan_response_schema() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    schema = {"name": "person", "schema": {"type": "object"}}
    spec = _spec(response_schema=schema)
    plan = adapter.plan(ctx, spec)

    assert plan.request["text"] == {"format": {"type": "json_schema", "name": "person", "schema": {"type": "object"}}}
    assert "response_schema" in plan.included


def test_plan_response_schema_from_plain_json_schema() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    spec = _spec(response_schema=schema)
    plan = adapter.plan(ctx, spec)

    assert plan.request["text"] == {"format": {"type": "json_schema", "name": "response", "schema": schema}}
    assert "response_schema" in plan.included


def test_plan_response_schema_explicit_format_defaults_name() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(response_schema={"type": "json_schema", "schema": {"type": "object"}})
    plan = adapter.plan(ctx, spec)

    assert plan.request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "response",
            "schema": {"type": "object"},
        }
    }


def test_plan_response_schema_explicit_format_with_legacy_nested() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        response_schema={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "description": "person payload",
                "strict": True,
                "schema": {"type": "object"},
            },
        }
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "person",
            "description": "person payload",
            "strict": True,
            "schema": {"type": "object", "additionalProperties": False},
        }
    }


def test_plan_response_schema_legacy_nested() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        response_schema={
            "json_schema": {
                "name": "person",
                "description": "person payload",
                "strict": True,
                "schema": {"type": "object"},
            },
        }
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "person",
            "description": "person payload",
            "strict": True,
            "schema": {"type": "object", "additionalProperties": False},
        }
    }


def test_plan_response_schema_strict_closes_nested_object_schemas() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
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
    spec = _spec(response_schema={"name": "payload", "schema": schema, "strict": True})
    plan = adapter.plan(ctx, spec)

    format_obj = plan.request["text"]["format"]
    assert format_obj["schema"]["additionalProperties"] is False
    assert format_obj["schema"]["properties"]["outer"]["additionalProperties"] is False
    assert format_obj["schema"]["properties"]["items"]["items"]["additionalProperties"] is False


def test_plan_response_schema_explicit_format_without_schema_raises() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(response_schema={"type": "json_schema", "name": "person"})

    with pytest.raises(ValueError, match="must include a schema"):
        adapter.plan(ctx, spec)


def test_plan_response_schema_legacy_nested_without_schema_raises() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(response_schema={"json_schema": {"name": "person"}})

    with pytest.raises(ValueError, match="must include a schema"):
        adapter.plan(ctx, spec)


def test_plan_tool_choice_only() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(tool_choice="auto")
    plan = adapter.plan(ctx, spec)

    assert plan.request["tool_choice"] == "auto"


def test_plan_with_cursor_previous_response_id() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.with_cursor(Cursor(last_response_id="resp_abc123"))

    plan = adapter.plan(ctx, _spec())
    assert plan.request["previous_response_id"] == "resp_abc123"
    assert "previous_response_id" in plan.included


def test_plan_extra_body() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    spec = _spec(extra_body={"store": False, "reasoning": {"effort": "high"}})
    plan = adapter.plan(ctx, spec)

    assert plan.request["store"] is False
    assert plan.request["reasoning"] == {"effort": "high"}


def test_plan_transport_overrides() -> None:
    adapter = OpenAIResponsesAdapter()
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
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(type="text", text="Let me check."),
                Part(type="tool_call", tool_call_id="call_1", tool_name="search", tool_args={"q": "test"}),
            ),
        )
    )

    plan = adapter.plan(ctx, _spec())
    input_items = plan.request["input"]

    # user, message (text), function_call
    assert input_items[1]["type"] == "message"
    assert input_items[1]["content"][0]["text"] == "Let me check."
    assert input_items[2]["type"] == "function_call"
    assert input_items[2]["name"] == "search"


def test_plan_omits_invalid_tool_call_without_name() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_call_id="call_1", tool_args={"q": "x"}),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    input_items = plan.request["input"]
    assert len(input_items) == 1


def test_plan_reports_invalid_user_item_without_serializable_content() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().append(Message(role="user", parts=(Part(type="image"),)))

    plan = adapter.plan(ctx, _spec())

    assert plan.request["input"] == []
    assert any(item.description == "context.messages[0]" for item in plan.excluded)
    assert any("responses.create request has no input items" in error for error in plan.errors)


def test_plan_roundtrips_image_generation_call_reference() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().append(
        Message(
            role="assistant",
            parts=(
                Part(
                    type="image",
                    provider_raw={"type": "image_generation_call", "id": "imggen_001"},
                ),
            ),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assert plan.request["input"][0] == {"type": "image_generation_call", "id": "imggen_001"}


def test_plan_preserves_compaction_item() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(
                    type="compaction",
                    provider_raw={
                        "type": "compaction",
                        "id": "cmp_001",
                        "encrypted_content": "encrypted-payload",
                    },
                ),
            ),
        )
    )

    plan = adapter.plan(ctx, _spec())
    input_items = plan.request["input"]

    assert input_items[1] == {
        "type": "compaction",
        "id": "cmp_001",
        "encrypted_content": "encrypted-payload",
    }


def test_plan_compaction_item_from_text_fallback() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="compaction", text="encrypted-fallback"),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    input_items = plan.request["input"]
    assert input_items[1] == {"type": "compaction", "encrypted_content": "encrypted-fallback"}


def test_plan_compaction_item_from_blob() -> None:
    adapter = OpenAIResponsesAdapter()
    store = InMemoryBlobStore()
    ref = store.put(b"encrypted-from-blob", media_type="text/plain", kind="compaction")
    ctx = Context(blob_store=store).user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="compaction", blob=ref),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assert plan.request["input"][1] == {
        "type": "compaction",
        "encrypted_content": "encrypted-from-blob",
    }


def test_plan_excludes_non_utf8_compaction_blob() -> None:
    adapter = OpenAIResponsesAdapter()
    store = InMemoryBlobStore()
    ref = store.put(b"\xff\xfe", media_type="application/octet-stream", kind="compaction")
    ctx = Context(blob_store=store).user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="compaction", blob=ref),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assert plan.request["input"] == [{"role": "user", "content": "Hello"}]
    assert any(item.description == "context.messages[1].parts[0]" for item in plan.excluded)
    assert any("UTF-8" in item.reason for item in plan.excluded)


def test_plan_excludes_missing_compaction_blob() -> None:
    adapter = OpenAIResponsesAdapter()
    store = InMemoryBlobStore()
    missing_ref = BlobReference(
        id="missing-blob-id",
        sha256="0" * 64,
        media_type="text/plain",
        kind="compaction",
        size=0,
    )
    ctx = Context(blob_store=store).user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(Part(type="compaction", blob=missing_ref),),
        )
    )

    plan = adapter.plan(ctx, _spec())
    assert plan.request["input"] == [{"role": "user", "content": "Hello"}]
    assert any(item.description == "context.messages[1].parts[0]" for item in plan.excluded)
    assert any("missing in blob_store" in item.reason for item in plan.excluded)


def test_compact_plan_minimal_request() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    spec = _spec(endpoint="responses.compact")
    plan = adapter.plan(ctx, spec)

    assert plan.request["model"] == "gpt-4o"
    assert plan.request["input"] == [{"role": "user", "content": "Hello"}]
    assert "previous_response_id" not in plan.request


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_compact_plan_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    merged = {"endpoint": "responses.compact"}
    merged.update(overrides)
    bad_spec = _spec(**merged)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.plan(ctx, bad_spec)


def test_compact_plan_includes_instructions() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    spec = _spec(
        endpoint="responses.compact",
        instructions=Instructions(system="System rule", developer="Developer rule"),
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["instructions"] == "System rule\n\nDeveloper rule"
    assert "system instruction" in plan.included
    assert "developer instruction" in plan.included


def test_compact_plan_with_cursor_and_overrides() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.with_cursor(Cursor(last_response_id="resp_prev"))
    spec = _spec(
        endpoint="responses.compact",
        extra_headers={"X-Trace-Id": "trace-1"},
        extra_query={"api-version": "2025-01-01"},
        extra_body={"store": False},
    )
    plan = adapter.plan(ctx, spec)

    assert plan.request["previous_response_id"] == "resp_prev"
    assert plan.request["extra_headers"] == {"X-Trace-Id": "trace-1"}
    assert plan.request["extra_query"] == {"api-version": "2025-01-01"}
    assert plan.request["store"] is False


def test_compact_plan_preserves_compaction_input_items() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    ctx = ctx.append(
        Message(
            role="assistant",
            parts=(
                Part(
                    type="compaction",
                    provider_raw={"type": "compaction", "encrypted_content": "encrypted-payload"},
                ),
            ),
        )
    )
    plan = adapter.plan(ctx, _spec(endpoint="responses.compact"))

    assert plan.request["input"][1] == {
        "type": "compaction",
        "encrypted_content": "encrypted-payload",
    }


def test_compact_plan_excludes_unsupported_fields() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    tool = ToolSpecification(
        name="get_weather",
        description="Get weather info",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = _spec(
        endpoint="responses.compact",
        max_output_tokens=100,
        temperature=0.2,
        top_p=0.8,
        seed=123,
        tools=(tool,),
        tool_choice="auto",
        response_schema={"name": "schema", "schema": {"type": "object"}},
        response_modalities=("text",),
    )
    plan = adapter.plan(ctx, spec)

    excluded = {item.description for item in plan.excluded}
    assert excluded == {
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "tools",
        "tool_choice",
        "response_schema",
        "response_modalities",
    }


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
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")
    bad_spec = _spec(**overrides)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"output": []}, spec=bad_spec)


@pytest.mark.parametrize(
    ("overrides", "error_pattern"),
    [
        ({"provider": "invalid-provider"}, "provider mismatch"),
        ({"endpoint": "invalid.endpoint"}, "endpoint mismatch"),
    ],
)
def test_compact_ingest_rejects_mismatched_runspec_target(overrides: dict[str, str], error_pattern: str) -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")
    merged = {"endpoint": "responses.compact"}
    merged.update(overrides)
    bad_spec = _spec(**merged)

    with pytest.raises(ValueError, match=error_pattern):
        adapter.ingest(ctx, {"output": []}, spec=bad_spec)


def test_ingest_text_response() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "resp_001",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    assert len(ctx) == 2
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].text == "Hi there!"
    assert last.provider == "openai"
    assert last.id == "resp_001"

    # Cursor updated with response ID.
    assert ctx.cursor.last_response_id == "resp_001"

    assert len(ctx.usage_log) == 1
    assert ctx.usage_log[0].input_tokens == 10
    assert ctx.usage_log[0].output_tokens == 5
    assert ctx.usage_log[0].total_tokens == 15
    assert ctx.usage_log[0].provider_usage["input_tokens"] == 10
    assert ctx.usage_log[0].provider_usage["output_tokens"] == 5
    assert ctx.usage_log[0].provider_usage["total_tokens"] == 15
    assert ctx.usage_log[0].provider_usage["input_tokens_details"] == {"cached_tokens": 3}
    assert ctx.usage_log[0].provider_usage["output_tokens_details"] == {"reasoning_tokens": 2}


def test_ingest_function_call_response() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "id": "resp_002",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "tool_call"
    assert last.parts[0].tool_call_id == "call_abc"
    assert last.parts[0].tool_name == "get_weather"
    assert last.parts[0].tool_args == {"city": "Tokyo"}


def test_ingest_function_call_response_with_object_arguments() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "id": "resp_002b",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_obj",
                "name": "get_weather",
                "arguments": {"city": "Tokyo", "unit": "celsius"},
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "tool_call"
    assert last.parts[0].tool_args == {"city": "Tokyo", "unit": "celsius"}


def test_ingest_text_and_function_call() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("What's the weather?")

    response = {
        "id": "resp_003",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "text"
    assert last.parts[0].text == "Let me check."
    assert last.parts[1].type == "tool_call"
    assert last.parts[1].tool_name == "get_weather"


def test_ingest_message_output_image_block() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Generate an image.")
    image_b64 = "aGVsbG8="

    response = {
        "id": "resp_img_001",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_image", "b64_json": image_b64, "media_type": "image/png"}],
            }
        ],
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 1
    assert last.parts[0].type == "image"
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get(last.parts[0].blob) == b"hello"


def test_ingest_image_generation_call_item() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Generate an image.")
    response = {
        "id": "resp_img_002",
        "output": [
            {
                "type": "image_generation_call",
                "id": "imggen_123",
                "result": "aGVsbG8=",
                "media_type": "image/png",
            }
        ],
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    last = ctx2.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 1
    assert last.parts[0].type == "image"
    assert last.parts[0].provider_raw == {
        "type": "image_generation_call",
        "id": "imggen_123",
        "result": "aGVsbG8=",
        "media_type": "image/png",
    }
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get(last.parts[0].blob) == b"hello"


def test_ingest_reasoning_item() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Think first.")

    response = {
        "id": "resp_reasoning",
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "I should compute this carefully."}],
                "status": "completed",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The answer is 42."}],
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 2
    assert last.parts[0].type == "thinking"
    assert last.parts[0].text == "I should compute this carefully."
    assert last.parts[1].type == "text"
    assert last.parts[1].text == "The answer is 42."


def test_ingest_reasoning_item_merges_content_and_summary() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Think first.")

    response = {
        "id": "resp_reasoning_merge",
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "Reasoning content"}],
                "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "thinking"
    assert last.parts[0].text == "Reasoning content\nReasoning summary"


def test_ingest_empty_output() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")

    response = {"id": "resp_004", "output": []}
    ctx2 = adapter.ingest(ctx, response, spec=_spec())
    assert len(ctx2) == 1
    assert ctx2.cursor.last_response_id == "resp_004"


def test_ingest_preserves_existing_cursor_fields() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().with_cursor(Cursor(last_response_id="resp_old", conversation_id="conv_1", session_id="sess_1"))
    ctx = ctx.user("Hello")

    response = {"id": "resp_new", "output": []}
    ctx2 = adapter.ingest(ctx, response, spec=_spec())

    assert ctx2.cursor.last_response_id == "resp_new"
    assert ctx2.cursor.conversation_id == "conv_1"
    assert ctx2.cursor.session_id == "sess_1"


def test_ingest_malformed_arguments() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "resp_005",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_bad",
                "name": "func",
                "arguments": "not-valid-json",
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())

    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].tool_args == {"_raw": "not-valid-json"}


def test_ingest_non_object_function_arguments_fallback() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "resp_005b",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_array",
                "name": "func",
                "arguments": "[1,2,3]",
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].tool_args == {"_raw": "[1,2,3]"}


def test_ingest_compaction_item() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "resp_006",
        "output": [
            {
                "type": "compaction",
                "id": "cmp_001",
                "encrypted_content": "encrypted-payload",
            },
        ],
    }

    ctx = adapter.ingest(ctx, response, spec=_spec())
    last = ctx.last(role="assistant")
    assert last is not None
    assert last.parts[0].type == "compaction"
    assert last.parts[0].provider_raw == response["output"][0]
    assert last.parts[0].blob is not None
    assert ctx.blob_store.get(last.parts[0].blob) == b"encrypted-payload"


def test_compact_ingest_only_compaction_part_and_usage() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().with_cursor(Cursor(last_response_id="resp_old")).user("Hello")

    response = {
        "id": "cmp_resp_001",
        "object": "response.compaction",
        "output": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
            {
                "type": "compaction",
                "id": "cmp_001",
                "encrypted_content": "encrypted-payload",
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 4},
        },
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec(endpoint="responses.compact"))

    last = ctx2.last(role="assistant")
    assert last is not None
    assert len(last.parts) == 1
    assert last.parts[0].type == "compaction"
    assert last.parts[0].blob is not None
    assert ctx2.blob_store.get(last.parts[0].blob) == b"encrypted-payload"

    # Compact ingestion should not overwrite the existing conversation cursor.
    assert ctx2.cursor.last_response_id == "resp_old"

    assert len(ctx2.usage_log) == 1
    assert ctx2.usage_log[0].input_tokens == 10
    assert ctx2.usage_log[0].output_tokens == 2
    assert ctx2.usage_log[0].total_tokens == 12
    assert ctx2.usage_log[0].provider_usage["input_tokens_details"] == {"cached_tokens": 4}


def test_compact_ingest_ignores_invalid_compaction_items() -> None:
    adapter = OpenAIResponsesCompactAdapter()
    ctx = Context().user("Hello")

    response = {
        "id": "cmp_resp_invalid",
        "output": [
            {"type": "compaction", "id": "cmp_bad", "encrypted_content": 123},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ignore"}]},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    ctx2 = adapter.ingest(ctx, response, spec=_spec(endpoint="responses.compact"))
    assert len(ctx2) == 1
    assert len(ctx2.usage_log) == 1


def test_ingest_supports_model_dump_warnings_flag() -> None:
    adapter = OpenAIResponsesAdapter()
    ctx = Context().user("Hello")

    class _FakeResponse:
        def model_dump(self, *, mode: str = "python", warnings: bool = True) -> dict[str, object]:
            assert mode == "python"
            assert warnings is False
            return {
                "id": "resp_fake",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            }

    ctx2 = adapter.ingest(ctx, _FakeResponse(), spec=_spec())
    assert ctx2.cursor.last_response_id == "resp_fake"
    assert ctx2.usage_log[0].provider_usage["input_tokens"] == 1
