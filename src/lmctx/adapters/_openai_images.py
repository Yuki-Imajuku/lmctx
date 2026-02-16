"""OpenAI Images API adapter.

Converts between lmctx Context/RunSpec and the OpenAI ``images.generate`` format.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmctx.adapters._util import _deep_merge, _json_object, _plan_extra_hints, _to_dict, _validate_adapter_spec
from lmctx.plan import AdapterCapabilities, AdapterId, ExcludedItem, RequestPlan
from lmctx.types import Message, Part, Usage

if TYPE_CHECKING:
    from lmctx.context import Context
    from lmctx.spec import RunSpec


def _usage_int(usage_data: Mapping[str, object], key: str) -> int | None:
    """Read an integer usage field."""
    value = usage_data.get(key)
    return value if isinstance(value, int) else None


def _extract_prompt(ctx: Context) -> str:
    """Extract the latest user prompt text from Context."""
    last_user = ctx.last(role="user")
    if last_user is None:
        msg = "OpenAI images.generate requires at least one user message with text."
        raise ValueError(msg)

    texts = [part.text for part in last_user.parts if part.type == "text" and part.text]
    if not texts:
        msg = "OpenAI images.generate requires text content in the latest user message."
        raise ValueError(msg)
    return "\n".join(texts)


def _build_prompt(ctx: Context, spec: RunSpec) -> str:
    """Build prompt text by combining instructions and latest user text."""
    prompt = _extract_prompt(ctx)
    instructions: list[str] = []
    if spec.instructions:
        if spec.instructions.system:
            instructions.append(spec.instructions.system)
        if spec.instructions.developer:
            instructions.append(spec.instructions.developer)
    if not instructions:
        return prompt
    return "\n\n".join((*instructions, prompt))


def _modalities_to_response_format(modalities: tuple[str, ...]) -> str | None:
    """Translate response modalities hints to Images API response_format."""
    if not modalities:
        return None
    normalized = modalities[0].strip().lower()
    if normalized in {"image_url", "url"}:
        return "url"
    if normalized in {"b64_json", "base64", "b64"}:
        return "b64_json"
    return None


def _supports_response_format(model: str) -> bool:
    """Return whether the model supports explicit ``response_format`` selection."""
    normalized = model.strip().lower()
    return normalized.startswith("dall-e-")


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    """Convert a mapping-like object into ``dict[str, object]`` when possible."""
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _unsupported_runspec_items(spec: RunSpec) -> list[ExcludedItem]:
    """Collect RunSpec fields that images.generate cannot honor."""
    excluded: list[ExcludedItem] = []
    if spec.max_output_tokens is not None:
        excluded.append(
            ExcludedItem(
                description="max_output_tokens",
                reason="images.generate does not support text token limits",
            )
        )
    if spec.temperature is not None:
        excluded.append(
            ExcludedItem(
                description="temperature",
                reason="images.generate does not expose sampling temperature",
            )
        )
    if spec.top_p is not None:
        excluded.append(
            ExcludedItem(
                description="top_p",
                reason="images.generate does not expose nucleus sampling controls",
            )
        )
    if spec.seed is not None:
        excluded.append(
            ExcludedItem(
                description="seed",
                reason="images.generate does not support deterministic sampling seed",
            )
        )
    if spec.tools:
        excluded.append(
            ExcludedItem(
                description="tools",
                reason="images.generate does not execute tools",
            )
        )
    if spec.tool_choice is not None:
        excluded.append(
            ExcludedItem(
                description="tool_choice",
                reason="images.generate does not execute tools",
            )
        )
    if spec.response_schema is not None:
        excluded.append(
            ExcludedItem(
                description="response_schema",
                reason="images.generate does not support structured text output",
            )
        )
    return excluded


def _image_part_from_item(ctx: Context, item: dict[str, object]) -> Part | None:
    """Build an image part from an Images API response item."""
    b64_json = item.get("b64_json")
    if isinstance(b64_json, str) and b64_json:
        try:
            data = base64.b64decode(b64_json, validate=True)
        except (binascii.Error, ValueError):
            data = None
        if data is not None:
            blob = ctx.blob_store.put_blob(data, media_type="image/png", kind="image")
            return Part(type="image", blob=blob, provider_raw=item)

    image_url = item.get("url")
    if isinstance(image_url, str) and image_url:
        return Part(type="image", provider_raw=item)
    return None


_CAPABILITIES = AdapterCapabilities(
    id=AdapterId(provider="openai", endpoint="images.generate"),
    fields={
        "instructions": "partial",
        "max_output_tokens": "no",
        "temperature": "no",
        "top_p": "no",
        "seed": "no",
        "tools": "no",
        "tool_choice": "no",
        "response_schema": "no",
        "response_modalities": "partial",
        "extra_body": "yes",
        "extra_headers": "yes",
        "extra_query": "yes",
        "cursor_chaining": "no",
    },
    notes={
        "instructions": "Instructions are prepended to prompt text.",
        "response_modalities": "Only maps to response_format for DALL-E models.",
        "cursor_chaining": "images.generate is stateless in this adapter.",
    },
)


class OpenAIImagesAdapter:
    """Adapter for the OpenAI Images API (``images.generate``)."""

    id = AdapterId(provider="openai", endpoint="images.generate")

    def capabilities(self) -> AdapterCapabilities:
        """Return capability metadata for this adapter."""
        return _CAPABILITIES

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build an OpenAI Images API request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        included: list[str] = [f"{len(ctx)} messages"]
        excluded = _unsupported_runspec_items(spec)
        warnings: list[str] = []

        prompt = _build_prompt(ctx, spec)
        request: dict[str, object] = {
            "model": spec.model,
            "prompt": prompt,
        }

        if spec.instructions:
            if spec.instructions.system:
                included.append("system instruction")
            if spec.instructions.developer:
                included.append("developer instruction")

        if spec.response_modalities:
            response_format = _modalities_to_response_format(spec.response_modalities)
            if response_format is None:
                warnings.append(
                    "response_modalities value is not recognized for images.generate; "
                    "expected one of: url, image_url, b64_json"
                )
            elif _supports_response_format(spec.model):
                request["response_format"] = response_format
                included.append("response_modalities")
            else:
                warnings.append(
                    "response_modalities is ignored for GPT image models because "
                    "images.generate does not accept response_format"
                )
        if spec.extra_headers:
            request["extra_headers"] = dict(spec.extra_headers)
            included.append("extra_headers")
        if spec.extra_query:
            request["extra_query"] = dict(spec.extra_query)
            included.append("extra_query")

        if spec.extra_body:
            request = _deep_merge(request, spec.extra_body)

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            warnings=tuple(warnings),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
        """Parse an OpenAI Images API response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        raw_items = data.get("data")
        if not isinstance(raw_items, list):
            return ctx

        image_parts: list[Part] = []
        for raw_item in raw_items:
            item = _as_str_object_dict(raw_item)
            if item is None:
                continue
            image_part = _image_part_from_item(ctx, item)
            if image_part is not None:
                image_parts.append(image_part)

        if image_parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(image_parts),
                    provider="openai",
                )
            )

        usage_data = data.get("usage")
        if isinstance(usage_data, Mapping):
            ctx = ctx.with_usage(
                Usage(
                    input_tokens=_usage_int(usage_data, "input_tokens"),
                    output_tokens=_usage_int(usage_data, "output_tokens"),
                    total_tokens=_usage_int(usage_data, "total_tokens"),
                    provider_usage=_json_object(usage_data),
                )
            )

        return ctx
