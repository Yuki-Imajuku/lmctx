"""Google Gemini image generation with lmctx + google-genai SDK.

Requirements:
    pip install 'lmctx[google]'
    export GOOGLE_API_KEY=...
"""

from pathlib import Path

from google import genai

from lmctx import Context, RunSpec
from lmctx.adapters import GoogleGenAIAdapter
from lmctx.spec import Instructions

MODEL = "gemini-3-pro-image-preview"
OUTPUT_DIR = Path(__file__).parent / "outputs"

adapter = GoogleGenAIAdapter()
client = genai.Client()

ctx = Context().user("Create a minimalist flat-design poster of Mt. Fuji with a red sun.")

spec = RunSpec(
    provider="google",
    endpoint="models.generate_content",
    model=MODEL,
    instructions=Instructions(system="Generate clean, high-contrast images suitable for presentations."),
    response_modalities=("IMAGE",),
    extra_body={"image_config": {"aspect_ratio": "1:1"}},
)

plan = adapter.plan(ctx, spec)
print("=== Plan diagnostics ===")
print(f"  Model: {MODEL}")
print(f"  Included: {plan.included}")
print(f"  Excluded: {plan.excluded}")

response = client.models.generate_content(**plan.request)
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
image_bytes = ctx.blob_store.get_blob(blob)
out_path = OUTPUT_DIR / "gemini-generated.png"
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

for part in assistant.parts:
    if part.type == "text" and part.text:
        print(f"  Text note: {part.text}")
