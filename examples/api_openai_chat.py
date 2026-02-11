"""OpenRouter Chat Completions: build requests with lmctx, call with OpenAI SDK.

Requirements:
    pip install 'lmctx[openai]'
    export OPENROUTER_API_KEY=...
    # Optional ranking headers:
    # export OPENROUTER_SITE_URL=https://example.com
    # export OPENROUTER_APP_NAME=lmctx-demo
"""

import os
from collections.abc import Mapping
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from lmctx import Context, Message, Part, RunSpec, ToolSpecification, Usage, put_file
from lmctx.adapters import OpenAIChatCompletionsAdapter
from lmctx.spec import Instructions

# OpenRouter Moonshot model as requested.
MODEL = "moonshotai/kimi-k2.5"
OPENROUTER_PROVIDER_SLUG = "moonshotai"
OPENROUTER_PROVIDER_NAME = "Moonshot AI"
ASSETS_DIR = Path(__file__).parent / "assets"
IMAGE_PATH = ASSETS_DIR / "image.jpg"


class CountryCapital(BaseModel):
    country: str
    capital: str
    reason: str


# ── Setup ──────────────────────────────────────────────────────────

adapter = OpenAIChatCompletionsAdapter()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

openrouter_headers: dict[str, str] = {}
if os.getenv("OPENROUTER_SITE_URL"):
    openrouter_headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"]
if os.getenv("OPENROUTER_APP_NAME"):
    openrouter_headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"]


def _normalize_provider_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _to_dict(response: object) -> dict[str, object]:
    mapped = _as_str_object_dict(response)
    if mapped is not None:
        return mapped

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        mapped = _as_str_object_dict(dumped)
        if mapped is not None:
            return mapped

    return {}


def _extract_model_name(response: object, data: dict[str, object]) -> str | None:
    raw_model = data.get("model")
    if isinstance(raw_model, str) and raw_model:
        return raw_model

    fallback_model = getattr(response, "model", None)
    if isinstance(fallback_model, str) and fallback_model:
        return fallback_model

    return None


def _extract_provider_name(response: object, data: dict[str, object]) -> str | None:
    provider_raw: object = data.get("provider", getattr(response, "provider", None))
    if provider_raw is None:
        model_extra = getattr(response, "model_extra", None)
        model_extra_data = _as_str_object_dict(model_extra)
        if model_extra_data is not None:
            provider_raw = model_extra_data.get("provider")

    if isinstance(provider_raw, str):
        return provider_raw

    provider_data = _as_str_object_dict(provider_raw)
    if provider_data is not None:
        candidate = provider_data.get("name") or provider_data.get("slug")
        if isinstance(candidate, str) and candidate:
            return candidate

    return None


def _usage_int(usage_data: Mapping[str, object], key: str) -> int | None:
    value = usage_data.get(key)
    return value if isinstance(value, int) else None


def _stream_usage_to_lmctx(usage_obj: object) -> Usage | None:
    usage_data = _to_dict(usage_obj)
    if not usage_data:
        return None

    input_tokens = _usage_int(usage_data, "prompt_tokens")
    output_tokens = _usage_int(usage_data, "completion_tokens")
    total_tokens = _usage_int(usage_data, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        provider_usage=usage_data,
    )


def openrouter_request_overrides(extra: dict[str, object] | None = None) -> dict[str, object]:
    """Return RunSpec.extra_body patch for OpenRouter-specific routing."""
    overrides: dict[str, object] = {
        "extra_body": {
            "provider": {
                "order": [OPENROUTER_PROVIDER_SLUG],
                "allow_fallbacks": False,
            }
        },
    }
    if extra:
        overrides.update(extra)
    return overrides


def openai_output_schema(model: type[BaseModel], *, name: str) -> dict[str, object]:
    schema = {str(key): value for key, value in model.model_json_schema().items()}
    return {"name": name, "schema": schema, "strict": True}


def verify_openrouter_routing(expected_model: str, expected_provider: str, response: object) -> tuple[str, str]:
    """Verify model/provider from OpenRouter response payload."""
    data = _to_dict(response)

    actual_model = _extract_model_name(response, data)
    if not isinstance(actual_model, str) or not actual_model:
        msg = "OpenRouter response did not include a model name."
        raise RuntimeError(msg)

    if actual_model != expected_model and expected_model not in actual_model:
        msg = f"Model mismatch: requested={expected_model}, actual={actual_model}"
        raise RuntimeError(msg)

    provider_name = _extract_provider_name(response, data)
    if provider_name is None:
        msg = "OpenRouter response did not include provider routing information."
        raise RuntimeError(msg)

    if _normalize_provider_name(expected_provider) not in _normalize_provider_name(provider_name):
        msg = f"Provider mismatch: requested={expected_provider}, actual={provider_name}"
        raise RuntimeError(msg)

    return actual_model, provider_name


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
    missing = [str(path) for path in (IMAGE_PATH,) if not path.exists()]
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
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    instructions=Instructions(system="You are a helpful geography assistant."),
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides(),
)

plan = adapter.plan(ctx, spec)
print("=== Plan diagnostics ===")
print(f"  Included: {plan.included}")

response = client.chat.completions.create(**plan.request)
actual_model, actual_provider = verify_openrouter_routing(spec.model, OPENROUTER_PROVIDER_NAME, response)
print(f"  Model verified: requested={spec.model}, actual={actual_model}")
print(f"  Provider verified: requested={OPENROUTER_PROVIDER_NAME}, actual={actual_provider}")
ctx = adapter.ingest(ctx, response, spec=spec)

print("\n=== Simple conversation ===")
for msg in ctx:
    for part in msg.parts:
        if part.type == "text" and part.text:
            print(f"  [{msg.role}] {part.text}")

# ── System message control (English query -> Japanese answer) ─────

ctx_japanese = Context().user("Explain in two sentences why Kyoto is popular with travelers.")
japanese_spec = RunSpec(
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    instructions=Instructions(system="Answer in Japanese"),
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides(),
)
japanese_plan = adapter.plan(ctx_japanese, japanese_spec)
japanese_response = client.chat.completions.create(**japanese_plan.request)
actual_model, actual_provider = verify_openrouter_routing(
    japanese_spec.model, OPENROUTER_PROVIDER_NAME, japanese_response
)
print("\n=== System Message ===")
print(f"  Model verified: requested={japanese_spec.model}, actual={actual_model}")
print(f"  Provider verified: requested={OPENROUTER_PROVIDER_NAME}, actual={actual_provider}")
ctx_japanese = adapter.ingest(ctx_japanese, japanese_response, spec=japanese_spec)
for msg in ctx_japanese:
    for part in msg.parts:
        if part.type == "text" and part.text:
            print(f"  [{msg.role}] {part.text}")

# ── Structured output (Pydantic BaseModel) ────────────────────────

ctx_schema = Context().user("Return France's capital as structured JSON.")
schema_spec = RunSpec(
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    response_schema=openai_output_schema(CountryCapital, name="country_capital"),
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides(),
)
schema_plan = adapter.plan(ctx_schema, schema_spec)
schema_response = client.chat.completions.create(**schema_plan.request)
actual_model, actual_provider = verify_openrouter_routing(schema_spec.model, OPENROUTER_PROVIDER_NAME, schema_response)
print("\n=== Structured output run (Pydantic BaseModel) ===")
print(f"  Model verified: requested={schema_spec.model}, actual={actual_model}")
print(f"  Provider verified: requested={OPENROUTER_PROVIDER_NAME}, actual={actual_provider}")
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
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    tools=(weather_tool, time_tool),
    tool_choice="auto",
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides(),
)

print("\n=== Tool loop ===")
for turn in range(4):
    plan2 = adapter.plan(ctx2, spec2)
    response2 = client.chat.completions.create(**plan2.request)
    actual_model, actual_provider = verify_openrouter_routing(spec2.model, OPENROUTER_PROVIDER_NAME, response2)
    print(f"  Turn {turn + 1}: model={actual_model}, provider={actual_provider}")
    ctx2 = adapter.ingest(ctx2, response2, spec=spec2)

    assistant = ctx2.last(role="assistant")
    predicted_tool_calls = 0
    if assistant is not None:
        predicted_tool_calls = sum(1 for part in assistant.parts if part.type == "tool_call")
        for part in assistant.parts:
            if part.type == "text" and part.text:
                label = "Final answer" if predicted_tool_calls == 0 else "Assistant text"
                print(f"  {label}: {part.text}")

    ctx2, tool_call_count = append_tool_results(ctx2)
    if tool_call_count == 0:
        break

# ── 3. Thinking run ────────────────────────────────────────────────

ctx3 = Context().user("Solve 37 * 89 and explain briefly.")
thinking_spec = RunSpec(
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides({"reasoning_effort": "medium"}),
)
thinking_plan = adapter.plan(ctx3, thinking_spec)
thinking_response = client.chat.completions.create(**thinking_plan.request)
actual_model, actual_provider = verify_openrouter_routing(
    thinking_spec.model, OPENROUTER_PROVIDER_NAME, thinking_response
)
print("\n=== Thinking run ===")
print(f"  Model verified: requested={thinking_spec.model}, actual={actual_model}")
print(f"  Provider verified: requested={OPENROUTER_PROVIDER_NAME}, actual={actual_provider}")
ctx3 = adapter.ingest(ctx3, thinking_response, spec=thinking_spec)

thinking_last = ctx3.last(role="assistant")
if thinking_last:
    for part in thinking_last.parts:
        if part.type == "text" and part.text:
            print(f"  Answer: {part.text}")

# ── 4. Streaming run ───────────────────────────────────────────────

ctx4 = Context().user("In one sentence, compare Tokyo and London as travel destinations.")
stream_spec = RunSpec(
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides({"stream": True, "stream_options": {"include_usage": True}}),
)
stream_plan = adapter.plan(ctx4, stream_spec)
stream = client.chat.completions.create(**stream_plan.request)

print("\n=== Streaming run ===")
stream_model: str | None = None
stream_text_chunks: list[str] = []
stream_usage: Usage | None = None
for chunk in stream:
    if stream_model is None:
        candidate_model = getattr(chunk, "model", None)
        if isinstance(candidate_model, str) and candidate_model:
            stream_model = candidate_model

    if stream_usage is None:
        usage_entry = _stream_usage_to_lmctx(getattr(chunk, "usage", None))
        if usage_entry is not None:
            stream_usage = usage_entry

    choices = getattr(chunk, "choices", None)
    if not choices:
        continue
    delta = getattr(choices[0], "delta", None)
    text = getattr(delta, "content", None)
    if isinstance(text, str) and text:
        stream_text_chunks.append(text)
        print(text, end="", flush=True)
print()

if stream_text_chunks:
    ctx4 = ctx4.assistant("".join(stream_text_chunks))
if stream_usage is not None:
    ctx4 = ctx4.with_usage(stream_usage)

if stream_model is not None:
    if stream_model != stream_spec.model and stream_spec.model not in stream_model:
        msg = f"Streaming model mismatch: requested={stream_spec.model}, actual={stream_model}"
        raise RuntimeError(msg)
    print(f"  Stream model verified: requested={stream_spec.model}, actual={stream_model}")
else:
    print("  Stream model verification: no model field observed in streamed chunks.")

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
    provider="openai",
    endpoint="chat.completions",
    model=MODEL,
    extra_headers=openrouter_headers,
    extra_body=openrouter_request_overrides(),
)
image_plan = adapter.plan(ctx5, image_spec)
image_response = client.chat.completions.create(**image_plan.request)
actual_model, actual_provider = verify_openrouter_routing(image_spec.model, OPENROUTER_PROVIDER_NAME, image_response)
print("\n=== Image input run ===")
print(f"  Model verified: requested={image_spec.model}, actual={actual_model}")
print(f"  Provider verified: requested={OPENROUTER_PROVIDER_NAME}, actual={actual_provider}")
ctx5 = adapter.ingest(ctx5, image_response, spec=image_spec)
image_last = ctx5.last(role="assistant")
if image_last:
    for part in image_last.parts:
        if part.type == "text" and part.text:
            print(f"  Image answer: {part.text}")

# ── 6. Usage tracking (all contexts) ───────────────────────────────

print_usage_checks("simple conversation", ctx)
print_usage_checks("language control run", ctx_japanese)
print_usage_checks("structured output run", ctx_schema)
print_usage_checks("tool loop", ctx2)
print_usage_checks("thinking run", ctx3)
print_usage_checks("streaming run", ctx4)
print_usage_checks("image input run", ctx5)
