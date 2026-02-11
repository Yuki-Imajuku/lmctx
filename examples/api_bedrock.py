"""AWS Bedrock Converse: build requests with lmctx, call with boto3.

Requirements:
    pip install 'lmctx[bedrock]'
    # Configure AWS credentials (e.g. aws configure, environment variables, or IAM role)
"""

from collections.abc import Mapping
from pathlib import Path

import boto3
from pydantic import BaseModel

from lmctx import Context, Message, Part, RunSpec, ToolSpecification, Usage, put_file
from lmctx.adapters import BedrockConverseAdapter
from lmctx.spec import Instructions

# Latest Anthropic Claude model listed in Bedrock table as of 2026-02-11.
MODEL = "us.anthropic.claude-opus-4-6-v1"
ASSETS_DIR = Path(__file__).parent / "assets"
IMAGE_PATH = ASSETS_DIR / "image.jpg"
PDF_PATH = ASSETS_DIR / "notebook.pdf"


class CountryCapital(BaseModel):
    country: str
    capital: str
    reason: str


# ── Setup ──────────────────────────────────────────────────────────

adapter = BedrockConverseAdapter()
client = boto3.client("bedrock-runtime", region_name="us-east-1")


def pydantic_schema(model: type[BaseModel]) -> dict[str, object]:
    return {str(key): value for key, value in model.model_json_schema().items()}


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def extract_invoked_model(response: dict[str, object]) -> str | None:
    """Try to extract the invoked model ID from Bedrock response metadata."""
    response_metadata = _as_str_object_dict(response.get("ResponseMetadata"))
    if response_metadata is not None:
        headers = _as_str_object_dict(response_metadata.get("HTTPHeaders"))
        if headers is not None:
            for key in (
                "x-amzn-bedrock-invoked-model-id",
                "x-amzn-bedrock-model-id",
                "x-amzn-bedrock-model-arn",
            ):
                value = headers.get(key)
                if isinstance(value, str) and value:
                    return value

    trace = _as_str_object_dict(response.get("trace"))
    if trace is not None:
        prompt_router = _as_str_object_dict(trace.get("promptRouter"))
        if prompt_router is not None:
            invoked_model = prompt_router.get("invokedModelId")
            if isinstance(invoked_model, str) and invoked_model:
                return invoked_model

    return None


def verify_called_model(expected_model: str, response: dict[str, object]) -> None:
    """Verify the invoked model if Bedrock returns it."""
    actual_model = extract_invoked_model(response)
    if actual_model is None:
        print(
            "  Model verification: Bedrock response does not expose invoked model ID; "
            f"request used modelId={expected_model}"
        )
        return

    if actual_model != expected_model and expected_model not in actual_model:
        msg = f"Model mismatch: requested={expected_model}, actual={actual_model}"
        raise RuntimeError(msg)

    print(f"  Model verified: requested={expected_model}, actual={actual_model}")


def run_local_tool(tool_name: str | None, tool_args: Mapping[str, object] | None) -> dict[str, object]:
    """Fake local tools used only for demo."""
    args = tool_args or {}
    city = str(args.get("city", ""))

    if tool_name == "get_weather":
        table = {
            "Tokyo": {"city": "Tokyo", "temperature_c": 22, "condition": "sunny"},
            "London": {"city": "London", "temperature_c": 11, "condition": "cloudy"},
        }
        return table.get(city, {"city": city, "temperature_c": None, "condition": "unknown"})

    if tool_name == "get_local_time":
        table = {"Tokyo": "21:00", "London": "12:00"}
        return {"city": city, "local_time": table.get(city, "unknown")}

    return {"error": f"Unknown tool: {tool_name}"}


def append_tool_results(ctx: Context) -> tuple[Context, int]:
    """Execute tool calls in the last assistant message and append tool results."""
    last = ctx.last(role="assistant")
    if last is None:
        return ctx, 0

    tool_results: list[Part] = []
    call_count = 0

    for part in last.parts:
        if part.type != "tool_call":
            continue

        call_count += 1
        output = run_local_tool(part.tool_name, part.tool_args)
        print(f"  Tool call: {part.tool_name}({part.tool_args})")
        tool_results.append(
            Part(
                type="tool_result",
                tool_call_id=part.tool_call_id,
                tool_output=output,
            )
        )

    if tool_results:
        ctx = ctx.append(Message(role="tool", parts=tuple(tool_results)))

    return ctx, call_count


def ensure_assets() -> None:
    missing = [str(path) for path in (IMAGE_PATH, PDF_PATH) if not path.exists()]
    if missing:
        msg = f"Missing assets: {', '.join(missing)}"
        raise FileNotFoundError(msg)


def print_usage_checks(label: str, ctx: Context) -> None:
    print(f"\n=== Usage ({label}) ===")
    if not ctx.usage_log:
        print("  No usage entries recorded.")
        return

    for i, usage in enumerate(ctx.usage_log):
        missing: list[str] = []
        if usage.input_tokens is None:
            missing.append("input")
        if usage.output_tokens is None:
            missing.append("output")
        token_check = "ok" if not missing else f"missing={','.join(missing)}"
        print(
            "  "
            f"Call {i + 1}: input={usage.input_tokens}, output={usage.output_tokens}, "
            f"total={usage.total_tokens}, token_check={token_check}, provider_usage={usage.provider_usage}"
        )


def _stream_usage_to_lmctx(usage_data: object) -> Usage | None:
    usage_map = _as_str_object_dict(usage_data)
    if usage_map is None:
        return None

    input_tokens = usage_map.get("inputTokens")
    output_tokens = usage_map.get("outputTokens")
    total_tokens = usage_map.get("totalTokens")
    if not isinstance(input_tokens, int):
        input_tokens = None
    if not isinstance(output_tokens, int):
        output_tokens = None
    if not isinstance(total_tokens, int):
        total_tokens = None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        provider_usage=usage_map,
    )


# ── 1. Simple conversation ─────────────────────────────────────────

ctx = Context().user("What is the capital of France?")

spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
    instructions=Instructions(system="You are a helpful geography assistant."),
)

plan = adapter.plan(ctx, spec)
print("=== Plan diagnostics ===")
print(f"  Included: {plan.included}")

response = client.converse(**plan.request)
verify_called_model(spec.model, response)
ctx = adapter.ingest(ctx, response, spec=spec)

print("\n=== Simple conversation ===")
for msg in ctx:
    for part in msg.parts:
        if part.type == "text" and part.text:
            print(f"  [{msg.role}] {part.text}")

# ── System message control (English query -> Japanese answer) ─────

ctx_japanese = Context().user("Explain in two sentences why Kyoto is popular with travelers.")
japanese_spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
    instructions=Instructions(system="Answer in Japanese"),
)
japanese_plan = adapter.plan(ctx_japanese, japanese_spec)
japanese_response = client.converse(**japanese_plan.request)
print("\n=== System Message ===")
verify_called_model(japanese_spec.model, japanese_response)
ctx_japanese = adapter.ingest(ctx_japanese, japanese_response, spec=japanese_spec)
for msg in ctx_japanese:
    for part in msg.parts:
        if part.type == "text" and part.text:
            print(f"  [{msg.role}] {part.text}")

# ── Structured output (Pydantic BaseModel) ────────────────────────

ctx_schema = Context().user("Return France's capital as structured JSON.")
schema_spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
    response_schema=pydantic_schema(CountryCapital),
)
schema_plan = adapter.plan(ctx_schema, schema_spec)
schema_response = client.converse(**schema_plan.request)
print("\n=== Structured output run (Pydantic BaseModel) ===")
verify_called_model(schema_spec.model, schema_response)
ctx_schema = adapter.ingest(ctx_schema, schema_response, spec=schema_spec)
schema_last = ctx_schema.last(role="assistant")
if schema_last:
    for part in schema_last.parts:
        if part.type == "text" and part.text:
            parsed = CountryCapital.model_validate_json(part.text)
            print(f"  Structured output: {parsed.model_dump()}")
            break

# ── 2. Tool loop (multiple calls supported) ────────────────────────

weather_tool = ToolSpecification(
    name="get_weather",
    description="Get the current weather for a city",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)

time_tool = ToolSpecification(
    name="get_local_time",
    description="Get local time for a city",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)

ctx2 = Context().user("Compare weather in Tokyo and London, and include Tokyo local time.")

spec2 = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
    tools=(weather_tool, time_tool),
)

print("\n=== Tool loop ===")
for turn in range(4):
    plan2 = adapter.plan(ctx2, spec2)
    response2 = client.converse(**plan2.request)
    print(f"  Turn {turn + 1}")
    verify_called_model(spec2.model, response2)
    ctx2 = adapter.ingest(ctx2, response2, spec=spec2)

    assistant = ctx2.last(role="assistant")
    predicted_tool_calls = 0
    if assistant is not None:
        predicted_tool_calls = sum(1 for part in assistant.parts if part.type == "tool_call")
        for part in assistant.parts:
            if part.type == "thinking" and part.text:
                print(f"  Thinking: {part.text}")
            elif part.type == "text" and part.text:
                label = "Final answer" if predicted_tool_calls == 0 else "Assistant text"
                print(f"  {label}: {part.text}")

    ctx2, tool_call_count = append_tool_results(ctx2)
    if tool_call_count == 0:
        break

# ── 3. Thinking run ────────────────────────────────────────────────

ctx3 = Context().user("Solve 37 * 89 and explain briefly.")
thinking_spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
    extra_body={"additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}}},
)
thinking_plan = adapter.plan(ctx3, thinking_spec)
thinking_response = client.converse(**thinking_plan.request)
print("\n=== Thinking run ===")
verify_called_model(thinking_spec.model, thinking_response)
ctx3 = adapter.ingest(ctx3, thinking_response, spec=thinking_spec)

thinking_last = ctx3.last(role="assistant")
if thinking_last:
    for part in thinking_last.parts:
        if part.type == "thinking" and part.text:
            print(f"  Thinking: {part.text}")
        elif part.type == "text" and part.text:
            print(f"  Answer: {part.text}")

# ── 4. Streaming run ───────────────────────────────────────────────

ctx4 = Context().user("In one sentence, compare Tokyo and London as travel destinations.")
stream_spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
)
stream_plan = adapter.plan(ctx4, stream_spec)
stream_response = client.converse_stream(**stream_plan.request)

print("\n=== Streaming run ===")
verify_called_model(stream_spec.model, stream_response)
stream_usage = None
stream_text_chunks: list[str] = []
for event in stream_response["stream"]:
    if "contentBlockDelta" in event:
        block = event["contentBlockDelta"]
        if isinstance(block, dict):
            delta = block.get("delta")
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    stream_text_chunks.append(text)
                    print(text, end="", flush=True)
    elif "metadata" in event:
        metadata = event["metadata"]
        if isinstance(metadata, dict):
            stream_usage = metadata.get("usage")
print()
if isinstance(stream_usage, dict):
    print(f"  Stream usage: {stream_usage}")
if stream_text_chunks:
    ctx4 = ctx4.assistant("".join(stream_text_chunks))
usage_entry = _stream_usage_to_lmctx(stream_usage)
if usage_entry is not None:
    ctx4 = ctx4.with_usage(usage_entry)

# ── 5. Image input run ─────────────────────────────────────────────

ensure_assets()
ctx5 = Context()
image_ref = put_file(ctx5.blob_store, IMAGE_PATH, kind="image")
ctx5 = ctx5.user(
    [
        Part(type="text", text="Describe this image in one sentence."),
        Part(type="image", blob=image_ref),
    ]
)
image_spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
)
image_plan = adapter.plan(ctx5, image_spec)
image_response = client.converse(**image_plan.request)
print("\n=== Image input run ===")
verify_called_model(image_spec.model, image_response)
ctx5 = adapter.ingest(ctx5, image_response, spec=image_spec)
image_last = ctx5.last(role="assistant")
if image_last:
    for part in image_last.parts:
        if part.type == "text" and part.text:
            print(f"  Image answer: {part.text}")

# ── 6. PDF document run (inline bytes) ─────────────────────────────

print("\n=== PDF document run (inline bytes) ===")
ctx6 = Context()
pdf_ref = put_file(ctx6.blob_store, PDF_PATH, kind="document")
ctx6 = ctx6.user(
    [
        Part(type="text", text="Summarize this PDF in 3 bullet points."),
        Part(type="file", blob=pdf_ref, provider_raw={"name": "notebook_pdf"}),
    ]
)
document_spec = RunSpec(
    provider="bedrock",
    endpoint="converse",
    model=MODEL,
)
document_plan = adapter.plan(ctx6, document_spec)
document_response = client.converse(**document_plan.request)
verify_called_model(document_spec.model, document_response)

ctx6 = adapter.ingest(ctx6, document_response, spec=document_spec)
document_last = ctx6.last(role="assistant")
if document_last:
    for part in document_last.parts:
        if part.type == "text" and part.text:
            print(f"  PDF answer: {part.text}")
            break

# ── 7. Usage tracking (all contexts) ───────────────────────────────

print_usage_checks("simple conversation", ctx)
print_usage_checks("language control run", ctx_japanese)
print_usage_checks("structured output run", ctx_schema)
print_usage_checks("tool loop", ctx2)
print_usage_checks("thinking run", ctx3)
print_usage_checks("streaming run", ctx4)
print_usage_checks("image input run", ctx5)
print_usage_checks("pdf document run", ctx6)
