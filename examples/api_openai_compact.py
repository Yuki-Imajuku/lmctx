"""OpenAI Responses compact flow: build with lmctx, call with the OpenAI SDK.

Requirements:
    pip install 'lmctx[openai]'
    export OPENAI_API_KEY=sk-...
"""

from openai import OpenAI

from lmctx import Context
from lmctx.adapters import OpenAIResponsesAdapter, OpenAIResponsesCompactAdapter
from lmctx.spec import Instructions, RunSpec

# -- Setup --

client = OpenAI()
responses_adapter = OpenAIResponsesAdapter()
compact_adapter = OpenAIResponsesCompactAdapter()


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


# -- 1) Build a short conversation with responses.create --

ctx = Context().user("Give me a 3-bullet summary of Tokyo, then a 3-bullet summary of London.")

create_spec = RunSpec(
    provider="openai",
    endpoint="responses.create",
    model="gpt-4o-mini",
    instructions=Instructions(system="You are a concise travel assistant."),
)

plan = responses_adapter.plan(ctx, create_spec)
response = client.responses.create(**plan.request)
ctx = responses_adapter.ingest(ctx, response, spec=create_spec)

print("=== Initial answer ===")
assistant = ctx.last(role="assistant")
if assistant is not None:
    for part in assistant.parts:
        if part.type == "text" and part.text:
            print(part.text)


# -- 2) Compact the accumulated transcript --

compact_spec = RunSpec(
    provider="openai",
    endpoint="responses.compact",
    model="gpt-4o-mini",
)

compact_plan = compact_adapter.plan(ctx, compact_spec)
compact_response = client.responses.compact(**compact_plan.request)
ctx = compact_adapter.ingest(ctx, compact_response, spec=compact_spec)

print("\n=== Compaction ===")
compaction_msg = ctx.last(role="assistant")
if compaction_msg is not None:
    compaction_parts = [part for part in compaction_msg.parts if part.type == "compaction"]
    print(f"Compaction parts ingested: {len(compaction_parts)}")


# -- 3) Continue with responses.create (compaction item is round-tripped) --

ctx = ctx.user("Now compare those two cities in one sentence.")

plan2 = responses_adapter.plan(ctx, create_spec)
input_types = [item.get("type", "message") for item in plan2.request["input"] if isinstance(item, dict)]
print(f"Input item types include: {input_types[-4:]}")

response2 = client.responses.create(**plan2.request)
ctx = responses_adapter.ingest(ctx, response2, spec=create_spec)

print("\n=== Follow-up answer ===")
assistant2 = ctx.last(role="assistant")
if assistant2 is not None:
    for part in assistant2.parts:
        if part.type == "text" and part.text:
            print(part.text)

print_usage_checks("full compact flow", ctx)
