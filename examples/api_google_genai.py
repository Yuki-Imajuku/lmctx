"""Google GenAI (Gemini): build requests with lmctx, call with the google-genai SDK.

Requirements:
    pip install 'lmctx[google]'
    export GOOGLE_API_KEY=...
"""

from collections.abc import Mapping
from pathlib import Path

from google import genai
from pydantic import BaseModel

from lmctx import Context, Message, Part, RunSpec, ToolSpecification, put_file
from lmctx.adapters import GoogleGenAIAdapter
from lmctx.spec import Instructions

# Google's latest Gemini Flash preview family as of 2026-02-11.
MODEL = "gemini-3-flash-preview"
ASSETS_DIR = Path(__file__).parent / "assets"
IMAGE_PATH = ASSETS_DIR / "image.jpg"
PDF_PATH = ASSETS_DIR / "notebook.pdf"


class CountryCapital(BaseModel):
    country: str
    capital: str
    reason: str


# ── Setup ──────────────────────────────────────────────────────────

adapter = GoogleGenAIAdapter()
client = genai.Client()


def google_output_schema(model: type[BaseModel]) -> dict[str, object]:
    return {str(key): value for key, value in model.model_json_schema().items()}


def verify_called_model(expected_model: str, actual_model: object) -> str:
    """Verify the model returned by API response is compatible with requested model."""
    if not isinstance(actual_model, str) or not actual_model:
        msg = "Google response did not include model_version."
        raise RuntimeError(msg)

    if actual_model != expected_model and not actual_model.startswith(f"{expected_model}-"):
        msg = f"Model mismatch: requested={expected_model}, actual={actual_model}"
        raise RuntimeError(msg)

    return actual_model


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
        print(f"  Function call: {part.tool_name}({part.tool_args})")
        # Google requires tool_name for function_response.
        tool_results.append(
            Part(
                type="tool_result",
                tool_call_id=part.tool_call_id,
                tool_name=part.tool_name,
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


# ── 1. Simple conversation ─────────────────────────────────────────

ctx = Context().user("What is the capital of France?")

spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    instructions=Instructions(system="You are a helpful geography assistant."),
)

plan = adapter.plan(ctx, spec)
print("=== Plan diagnostics ===")
print(f"  Included: {plan.included}")

response = client.models.generate_content(**plan.request)
actual_model = verify_called_model(spec.model, getattr(response, "model_version", None))
print(f"  Model verified: requested={spec.model}, actual={actual_model}")
ctx = adapter.ingest(ctx, response, spec=spec)

print("\n=== Simple conversation ===")
for msg in ctx:
    for part in msg.parts:
        if part.type == "text" and part.text:
            print(f"  [{msg.role}] {part.text}")

# ── System message control (English query -> Japanese answer) ─────

ctx_japanese = Context().user("Explain in two sentences why Kyoto is popular with travelers.")
japanese_spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    instructions=Instructions(system="Answer in Japanese"),
)
japanese_plan = adapter.plan(ctx_japanese, japanese_spec)
japanese_response = client.models.generate_content(**japanese_plan.request)
actual_model = verify_called_model(japanese_spec.model, getattr(japanese_response, "model_version", None))
print("\n=== System Message ===")
print(f"  Model verified: requested={japanese_spec.model}, actual={actual_model}")
ctx_japanese = adapter.ingest(ctx_japanese, japanese_response, spec=japanese_spec)
for msg in ctx_japanese:
    for part in msg.parts:
        if part.type == "text" and part.text:
            print(f"  [{msg.role}] {part.text}")

# ── Structured output (Pydantic BaseModel) ────────────────────────

ctx_schema = Context().user("Return France's capital as structured JSON.")
schema_spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    response_schema=google_output_schema(CountryCapital),
)
schema_plan = adapter.plan(ctx_schema, schema_spec)
schema_response = client.models.generate_content(**schema_plan.request)
actual_model = verify_called_model(schema_spec.model, getattr(schema_response, "model_version", None))
print("\n=== Structured output run (Pydantic BaseModel) ===")
print(f"  Model verified: requested={schema_spec.model}, actual={actual_model}")
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
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    tools=(weather_tool, time_tool),
)

print("\n=== Tool loop ===")
for turn in range(4):
    plan2 = adapter.plan(ctx2, spec2)
    response2 = client.models.generate_content(**plan2.request)
    actual_model = verify_called_model(spec2.model, getattr(response2, "model_version", None))
    print(f"  Turn {turn + 1}: model={actual_model}")
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

# ── 3. Server-side web search tool run ─────────────────────────────

ctx2_server = Context().user(
    "Use Google Search and summarize one major AI-related announcement from this week in 3 bullet points."
)
server_tool_spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    extra_body={"tools": [{"google_search": {}}]},
)
server_tool_plan = adapter.plan(ctx2_server, server_tool_spec)
print("\n=== Server-side web search tool run ===")
config = server_tool_plan.request.get("config", {})
if isinstance(config, dict):
    print(f"  Request tools: {config.get('tools')}")
server_tool_response = client.models.generate_content(**server_tool_plan.request)
actual_model = verify_called_model(server_tool_spec.model, getattr(server_tool_response, "model_version", None))
print(f"  Model verified: requested={server_tool_spec.model}, actual={actual_model}")
ctx2_server = adapter.ingest(ctx2_server, server_tool_response, spec=server_tool_spec)

server_last = ctx2_server.last(role="assistant")
if server_last:
    for part in server_last.parts:
        if part.type == "text" and part.text:
            print(f"  Web search answer: {part.text}")

# ── 4. Thinking run ────────────────────────────────────────────────

ctx3 = Context().user("Solve 37 * 89 and explain briefly.")
thinking_spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    extra_body={"thinking_config": {"include_thoughts": True, "thinking_level": "low"}},
)
thinking_plan = adapter.plan(ctx3, thinking_spec)
thinking_response = client.models.generate_content(**thinking_plan.request)
actual_model = verify_called_model(thinking_spec.model, getattr(thinking_response, "model_version", None))
print("\n=== Thinking run ===")
print(f"  Model verified: requested={thinking_spec.model}, actual={actual_model}")
ctx3 = adapter.ingest(ctx3, thinking_response, spec=thinking_spec)

thinking_last = ctx3.last(role="assistant")
if thinking_last:
    for part in thinking_last.parts:
        if part.type == "thinking" and part.text:
            print(f"  Thinking: {part.text}")
        elif part.type == "text" and part.text:
            print(f"  Answer: {part.text}")

# ── 5. Streaming run ───────────────────────────────────────────────

ctx4 = Context().user("In one sentence, compare Tokyo and London as travel destinations.")
stream_spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
)
stream_plan = adapter.plan(ctx4, stream_spec)

print("\n=== Streaming run ===")
stream_response = None
for chunk in client.models.generate_content_stream(**stream_plan.request):
    stream_response = chunk
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text:
        print(text, end="", flush=True)
print()

if stream_response is None:
    msg = "Google streaming did not return any response chunk."
    raise RuntimeError(msg)

actual_model = verify_called_model(stream_spec.model, getattr(stream_response, "model_version", None))
print(f"  Stream model verified: requested={stream_spec.model}, actual={actual_model}")
ctx4 = adapter.ingest(ctx4, stream_response, spec=stream_spec)

# ── 6. Image input run ─────────────────────────────────────────────

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
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
)
image_plan = adapter.plan(ctx5, image_spec)
image_response = client.models.generate_content(**image_plan.request)
actual_model = verify_called_model(image_spec.model, getattr(image_response, "model_version", None))
print("\n=== Image input run ===")
print(f"  Model verified: requested={image_spec.model}, actual={actual_model}")
ctx5 = adapter.ingest(ctx5, image_response, spec=image_spec)
image_last = ctx5.last(role="assistant")
if image_last:
    for part in image_last.parts:
        if part.type == "text" and part.text:
            print(f"  Image answer: {part.text}")

# ── 7. File upload run (image + PDF) ───────────────────────────────

print("\n=== File upload run (image + PDF) ===")
uploaded_image = client.files.upload(file=IMAGE_PATH)
uploaded_pdf = client.files.upload(file=PDF_PATH)
image_uri = getattr(uploaded_image, "uri", None)
image_mime = getattr(uploaded_image, "mime_type", None)
pdf_uri = getattr(uploaded_pdf, "uri", None)
pdf_mime = getattr(uploaded_pdf, "mime_type", None)

if not all(isinstance(x, str) and x for x in (image_uri, image_mime, pdf_uri, pdf_mime)):
    msg = "Upload returned invalid file metadata."
    raise RuntimeError(msg)

ctx6 = Context().user(
    [
        Part(
            type="text",
            text="Describe the uploaded image briefly and summarize the uploaded PDF in 3 bullet points.",
        ),
        Part(type="file", provider_raw={"file_uri": image_uri, "mime_type": image_mime}),
        Part(type="file", provider_raw={"file_uri": pdf_uri, "mime_type": pdf_mime}),
    ]
)
upload_spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
)
upload_plan = adapter.plan(ctx6, upload_spec)
upload_response = client.models.generate_content(**upload_plan.request)
actual_model = verify_called_model(upload_spec.model, getattr(upload_response, "model_version", None))
print(f"  Model verified: requested={upload_spec.model}, actual={actual_model}")
print(f"  Uploaded image uri: {image_uri}")
print(f"  Uploaded pdf uri: {pdf_uri}")

ctx6 = adapter.ingest(ctx6, upload_response, spec=upload_spec)
upload_last = ctx6.last(role="assistant")
if upload_last:
    for part in upload_last.parts:
        if part.type == "text" and part.text:
            print(f"  Upload answer: {part.text}")
            break

# ── 8. Usage tracking (all contexts) ───────────────────────────────

print_usage_checks("simple conversation", ctx)
print_usage_checks("language control run", ctx_japanese)
print_usage_checks("structured output run", ctx_schema)
print_usage_checks("tool loop", ctx2)
print_usage_checks("server-side web search tool run", ctx2_server)
print_usage_checks("thinking run", ctx3)
print_usage_checks("streaming run", ctx4)
print_usage_checks("image input run", ctx5)
print_usage_checks("file upload run", ctx6)
