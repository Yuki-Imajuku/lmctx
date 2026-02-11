"""OpenAI Images API: build requests with lmctx, call with the OpenAI SDK.

Requirements:
    pip install 'lmctx[openai]'
    export OPENAI_API_KEY=sk-...
"""

from pathlib import Path

from openai import OpenAI

from lmctx import Context, RunSpec
from lmctx.adapters import OpenAIImagesAdapter
from lmctx.spec import Instructions

MODEL = "gpt-image-1.5"
OUTPUT_DIR = Path(__file__).parent / "outputs"

adapter = OpenAIImagesAdapter()
client = OpenAI()

ctx = Context().user("A cinematic watercolor illustration of Tokyo Tower at sunset.")

spec = RunSpec(
    provider="openai",
    endpoint="images.generate",
    model=MODEL,
    instructions=Instructions(system="Generate high quality, clean images suitable for documentation."),
    extra_body={"size": "1024x1024", "quality": "high"},
)

plan = adapter.plan(ctx, spec)
print("=== Plan diagnostics ===")
print(f"  Included: {plan.included}")

response = client.images.generate(**plan.request)
ctx = adapter.ingest(ctx, response, spec=spec)

assistant = ctx.last(role="assistant")
if assistant is None:
    msg = "No assistant message was ingested."
    raise RuntimeError(msg)

image_parts = [part for part in assistant.parts if part.type == "image" and part.blob is not None]
if not image_parts:
    msg = "No generated image blob was found in the ingested response."
    raise RuntimeError(msg)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
blob = image_parts[0].blob
assert blob is not None
image_bytes = ctx.blob_store.get(blob)
out_path = OUTPUT_DIR / "openai-generated.png"
out_path.write_bytes(image_bytes)

print("\n=== Image generation ===")
print(f"  Saved: {out_path}")
print(f"  Bytes: {len(image_bytes)}")
if ctx.usage_log:
    usage = ctx.usage_log[-1]
    print(
        "  Usage: "
        f"input={usage.input_tokens}, output={usage.output_tokens}, total={usage.total_tokens}, "
        f"provider_usage={usage.provider_usage}"
    )
else:
    print("  Usage: no entry recorded.")
