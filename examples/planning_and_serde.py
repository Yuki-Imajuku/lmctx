"""Planning diagnostics, capabilities, and serialization round-trips."""

from lmctx import (
    BlobNotFoundError,
    Context,
    Instructions,
    PlanValidationError,
    RequestPlan,
    RunSpec,
)
from lmctx.adapters import AutoAdapter, OpenAIImagesAdapter
from lmctx.types import Part

# ---- Capability API ----
images_adapter = OpenAIImagesAdapter()
caps = images_adapter.capabilities()
print("[capabilities] openai/images.generate")
print(f"  tool_choice = {caps.level('tool_choice')}")
print(f"  extra_headers = {caps.level('extra_headers')}")

# ---- RequestPlan strict validation ----
ctx = Context().user("Draw a minimal black cat icon.")
spec = RunSpec(
    provider="openai",
    endpoint="images.generate",
    model="gpt-image-1",
    tool_choice="auto",  # intentionally unsupported for images.generate
)
plan = images_adapter.plan(ctx, spec)
print("\n[plan] included:")
for item in plan.included:
    print(f"  - {item}")
print("[plan] unused parameter warnings:")
for warning in plan.unused_parameter_warnings():
    print(f"  - {warning}")

try:
    plan.assert_valid(fail_on_excluded=True)
except PlanValidationError as exc:
    print(f"[plan] strict validation raised: {type(exc).__name__}")

# RequestPlan to_dict/from_dict
plan_round_trip = RequestPlan.from_dict(plan.to_dict())
print(f"[plan serde] request keys = {sorted(plan_round_trip.request)}")

# AutoAdapter capability lookup
auto = AutoAdapter()
resolved = auto.capabilities(spec)
print(f"[auto capabilities] response_modalities = {resolved.level('response_modalities')}")

# ---- Context serialization (blob references vs payload embedding) ----
blob_ctx = Context()
blob_ref = blob_ctx.blob_store.put_blob(
    b"binary-bytes",
    media_type="application/octet-stream",
    kind="file",
)
blob_ctx = blob_ctx.user([Part(type="file", blob=blob_ref)])

ref_only = blob_ctx.to_dict()
restored_ref_only = Context.from_dict(ref_only)
print(f"\n[context serde] ref_only includes blob_payloads = {'blob_payloads' in ref_only}")
try:
    restored_ref_only.blob_store.get_blob(blob_ref)
except BlobNotFoundError:
    print("[context serde] ref-only restore cannot resolve blob bytes (expected)")

portable = blob_ctx.to_dict(include_blob_payloads=True)
restored_portable = Context.from_dict(portable)
restored_bytes = restored_portable.blob_store.get_blob(blob_ref)
print(f"[context serde] portable restore bytes = {restored_bytes!r}")

# ---- RunSpec serialization ----
spec_with_extras = RunSpec(
    provider="openai",
    endpoint="chat.completions",
    model="gpt-4o-mini",
    instructions=Instructions(system="Be concise."),
    extra_headers={"X-Debug": "1"},
    extra_query={"api-version": "2025-01-01"},
)
spec_round_trip = RunSpec.from_dict(spec_with_extras.to_dict())
print(f"\n[runspec serde] endpoint = {spec_round_trip.endpoint}")
print(f"[runspec serde] extra_headers = {dict(spec_round_trip.extra_headers)}")
