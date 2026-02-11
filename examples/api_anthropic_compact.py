"""Anthropic compaction via messages.create context management edits.

Requirements:
    pip install 'lmctx[anthropic]'
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from collections.abc import Mapping

from anthropic import Anthropic

from lmctx import Context, RunSpec
from lmctx.adapters import AnthropicMessagesAdapter
from lmctx.spec import Instructions

MODEL = "claude-opus-4-6"
COMPACTION_BETA = "compact-2026-01-12"
COMPACTION_TRIGGER_TOKENS = 50000
LONG_CONTEXT_REPEAT = 1300  # Keep this high enough to cross compaction trigger by default.


adapter = AnthropicMessagesAdapter()
client = Anthropic()


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _usage_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _sum_iteration_usage(usage_data: Mapping[str, object], field: str) -> int | None:
    raw_iterations = usage_data.get("iterations")
    if not isinstance(raw_iterations, list):
        return None

    total = 0
    found = False
    for iteration in raw_iterations:
        iteration_data = _as_str_object_dict(iteration)
        if iteration_data is None:
            continue
        value = _usage_int(iteration_data.get(field))
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def verify_called_model(expected_model: str, actual_model: object) -> str:
    """Verify the model returned by API response is compatible with requested model."""
    if not isinstance(actual_model, str) or not actual_model:
        msg = "Anthropic response did not include a model name."
        raise RuntimeError(msg)

    if actual_model != expected_model and not actual_model.startswith(f"{expected_model}-"):
        msg = f"Model mismatch: requested={expected_model}, actual={actual_model}"
        raise RuntimeError(msg)

    return actual_model


def response_to_dict(response: object) -> dict[str, object]:
    mapped = _as_str_object_dict(response)
    if mapped is not None:
        return mapped

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        mapped = _as_str_object_dict(dumped)
        if mapped is not None:
            return mapped

    msg = f"Unexpected response type: {type(response).__name__}. Expected dict or object with model_dump()."
    raise TypeError(msg)


# Keep the prompt moderately long so context-management behavior is easier to inspect.
long_context = " ".join(
    [
        "Tokyo has dense rail transit, mixed zoning, compact neighborhoods, and 24-hour services.",
        "London has layered history, radial rail corridors, and a polycentric urban structure.",
    ]
    * LONG_CONTEXT_REPEAT
)

ctx = Context().user(
    "Read this background and then summarize the key city-level differences:\n\n"
    f"{long_context}\n\n"
    "Summarize Tokyo vs London in 5 bullets."
)

compact_spec = RunSpec(
    provider="anthropic",
    endpoint="messages.create",
    model=MODEL,
    instructions=Instructions(system="You are a concise urban policy analyst."),
    extra_body={
        "betas": [COMPACTION_BETA],
        "context_management": {
            "edits": [
                {
                    "type": "compact_20260112",
                    # Anthropic requires input_tokens triggers to be >= 50,000.
                    "trigger": {"type": "input_tokens", "value": COMPACTION_TRIGGER_TOKENS},
                    "pause_after_compaction": True,
                    "instructions": ("Preserve city-specific facts, transport details, and quantitative comparisons."),
                }
            ]
        },
    },
)

plan = adapter.plan(ctx, compact_spec)
print("=== Plan diagnostics ===")
print(f"  Included: {plan.included}")
print(f"  Warnings: {plan.warnings}")

response = client.beta.messages.create(**plan.request)
actual_model = verify_called_model(compact_spec.model, getattr(response, "model", None))
print(f"  Model verified: requested={compact_spec.model}, actual={actual_model}")
ctx = adapter.ingest(ctx, response, spec=compact_spec)

print("\n=== Assistant text ===")
assistant = ctx.last(role="assistant")
printed_assistant_text = False
if assistant is not None:
    for part in assistant.parts:
        if part.type == "text" and part.text:
            print(f"  {part.text}")
            printed_assistant_text = True
if not printed_assistant_text:
    print("  (No assistant text; compaction-only response.)")

# Anthropic compaction blocks are returned in the provider payload.
raw_response = response_to_dict(response)
content = raw_response.get("content")
compaction_blocks: list[dict[str, object]] = []
if isinstance(content, list):
    for block in content:
        block_data = _as_str_object_dict(block)
        if block_data is not None and block_data.get("type") == "compaction":
            compaction_blocks.append(block_data)

print("\n=== Compaction blocks ===")
print(f"  Count: {len(compaction_blocks)}")
if compaction_blocks:
    first_content = compaction_blocks[0].get("content")
    if isinstance(first_content, str) and first_content:
        preview = first_content[:180].replace("\n", " ")
        print(f"  Preview: {preview}...")

usage = raw_response.get("usage")
usage_data = _as_str_object_dict(usage)
compaction_iterations: list[dict[str, object]] = []
if usage_data is not None:
    raw_iterations = usage_data.get("iterations")
    if isinstance(raw_iterations, list):
        for iteration in raw_iterations:
            iteration_data = _as_str_object_dict(iteration)
            if iteration_data is not None and iteration_data.get("type") == "compaction":
                compaction_iterations.append(iteration_data)

context_management = raw_response.get("context_management")
context_management_data = _as_str_object_dict(context_management)
if context_management_data is not None:
    print(f"  Context management: {context_management_data}")
    applied_edits = context_management_data.get("applied_edits")
    if isinstance(applied_edits, list) and not applied_edits:
        if compaction_blocks or compaction_iterations:
            print(
                "  Note: compaction output was returned but applied_edits is empty "
                "(expected when pause_after_compaction=True)."
            )
        else:
            print(
                "  Note: no compaction output observed. "
                f"Increase LONG_CONTEXT_REPEAT (currently {LONG_CONTEXT_REPEAT}) if input_tokens stayed below trigger."
            )

if usage_data is not None:
    input_tokens = _usage_int(usage_data.get("input_tokens"))
    output_tokens = _usage_int(usage_data.get("output_tokens"))
    if input_tokens in (None, 0):
        input_tokens = _sum_iteration_usage(usage_data, "input_tokens")
    if output_tokens in (None, 0):
        output_tokens = _sum_iteration_usage(usage_data, "output_tokens")
    total_tokens = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    print(f"  Usage summary: input={input_tokens}, output={output_tokens}, total={total_tokens}")

print("\n=== Usage (context) ===")
if not ctx.usage_log:
    print("  No usage entries recorded.")
else:
    for i, usage_entry in enumerate(ctx.usage_log):
        missing: list[str] = []
        if usage_entry.input_tokens is None:
            missing.append("input")
        if usage_entry.output_tokens is None:
            missing.append("output")
        token_check = "ok" if not missing else f"missing={','.join(missing)}"
        print(
            "  "
            f"Call {i + 1}: input={usage_entry.input_tokens}, output={usage_entry.output_tokens}, "
            f"total={usage_entry.total_tokens}, token_check={token_check}, provider_usage={usage_entry.provider_usage}"
        )
