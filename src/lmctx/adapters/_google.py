"""Google GenAI (Gemini) adapter.

Converts between lmctx Context/RunSpec and the Google GenAI SDK format.
Works with both Google AI Studio and Vertex AI via the ``google-genai`` package.

Key differences from OpenAI/Anthropic:
- Assistant role is called ``model``
- Tool calls use ``function_call`` / ``function_response`` format
- System instructions go in ``config.system_instruction``
- Images use ``inline_data`` with base64
- Tools are wrapped in ``function_declarations``
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from lmctx.adapters._util import (
    _deep_merge,
    _json_object,
    _plan_extra_hints,
    _to_dict,
    _to_json_compatible,
    _validate_adapter_spec,
)
from lmctx.plan import AdapterCapabilities, AdapterId, ExcludedItem, RequestPlan
from lmctx.types import Message, Part, Usage

if TYPE_CHECKING:
    from lmctx.blobs import BlobStore
    from lmctx.context import Context
    from lmctx.spec import RunSpec


class _GoogleContentItem(TypedDict):
    role: str
    parts: list[dict[str, object]]


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    """Convert an object to ``dict[str, object]`` when possible."""
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _usage_value(usage: Mapping[str, object], snake: str, camel: str) -> object:
    """Read a usage field that may appear in snake_case or camelCase."""
    if snake in usage:
        return usage[snake]
    return usage.get(camel)


def _usage_int(usage: Mapping[str, object], snake: str, camel: str) -> int | None:
    """Read an integer usage field that may appear in snake_case or camelCase."""
    value = _usage_value(usage, snake, camel)
    return value if isinstance(value, int) else None


def _sum_ints(*values: int | None) -> int | None:
    """Sum known integer values while allowing ``None`` entries."""
    total = 0
    found = False
    for value in values:
        if isinstance(value, int):
            total += value
            found = True
    return total if found else None


def _normalized_usage_tokens(usage: Mapping[str, object]) -> tuple[int | None, int | None, int | None]:
    """Normalize Google usage into lmctx input/output/total token fields.

    Google responses can expose separate thought/tool-use token buckets that may or may not
    be already reflected in prompt/candidate totals depending on endpoint/version.
    This method prefers the interpretation that best matches provider ``total_token_count``.
    """
    prompt_tokens = _usage_int(usage, "prompt_token_count", "promptTokenCount")
    candidate_tokens = _usage_int(usage, "candidates_token_count", "candidatesTokenCount")
    total_tokens = _usage_int(usage, "total_token_count", "totalTokenCount")

    tool_use_prompt_tokens = _usage_int(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount")
    thought_tokens = _usage_int(usage, "thoughts_token_count", "thoughtsTokenCount")

    base_input = prompt_tokens
    base_output = candidate_tokens
    base_total = _sum_ints(base_input, base_output)

    extended_input = _sum_ints(prompt_tokens, tool_use_prompt_tokens)
    extended_output = _sum_ints(candidate_tokens, thought_tokens)
    extended_total = _sum_ints(extended_input, extended_output)

    input_tokens = base_input
    output_tokens = base_output

    if total_tokens is None:
        if extended_input is not None:
            input_tokens = extended_input
        if extended_output is not None:
            output_tokens = extended_output
    else:
        base_delta = abs(total_tokens - base_total) if base_total is not None else None
        extended_delta = abs(total_tokens - extended_total) if extended_total is not None else None
        if extended_delta is not None and (base_delta is None or extended_delta < base_delta):
            input_tokens = extended_input
            output_tokens = extended_output

    resolved_total = total_tokens if total_tokens is not None else _sum_ints(input_tokens, output_tokens)
    return input_tokens, output_tokens, resolved_total


def _usage_payload(usage_data: Mapping[str, object]) -> dict[str, object]:
    """Copy full provider usage payload with stable string keys."""
    return _json_object(usage_data)


def _decode_inline_data_bytes(encoded: object) -> bytes | None:
    """Decode Google ``inline_data.data`` values into raw bytes."""
    if isinstance(encoded, bytes):
        return encoded
    if isinstance(encoded, bytearray):
        return bytes(encoded)
    if isinstance(encoded, memoryview):
        return encoded.tobytes()
    if isinstance(encoded, str) and encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None
    return None


def _inline_data_part_kind(media_type: str | None) -> tuple[str, str]:
    """Map inline payload MIME type to lmctx ``Part.type`` and blob kind."""
    if not isinstance(media_type, str) or not media_type:
        return "file", "file"

    major = media_type.split("/", 1)[0].strip().lower()
    if major in {"image", "audio", "video"}:
        return major, major
    return "file", "file"


def _convert_tool_output(output: object) -> dict[str, object]:
    """Convert tool output to a dict for function_response."""
    if isinstance(output, Mapping):
        return {str(k): _to_json_compatible(v) for k, v in output.items()}
    if isinstance(output, str):
        return {"result": output}
    return {"result": json.dumps(_to_json_compatible(output), ensure_ascii=False)}


def _file_part_to_google(part: Part, store: BlobStore) -> dict[str, object] | None:
    """Convert ``Part(type='file')`` to Google file_data or inline_data."""
    raw = _as_str_object_dict(part.provider_raw)
    if raw is not None:
        uri = raw.get("file_uri")
        if not isinstance(uri, str) or not uri:
            uri = raw.get("uri")
        mime_type = raw.get("mime_type")
        if not isinstance(mime_type, str) or not mime_type:
            mime_type = raw.get("media_type")

        if isinstance(uri, str) and uri and isinstance(mime_type, str) and mime_type:
            return {
                "file_data": {
                    "file_uri": uri,
                    "mime_type": mime_type,
                }
            }

    if part.blob:
        data = store.get(part.blob)
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "inline_data": {
                "mime_type": part.blob.media_type or "application/octet-stream",
                "data": b64,
            }
        }

    return None


def _tool_call_part_from_provider_raw(part: Part) -> dict[str, object] | None:  # noqa: C901
    """Preserve provider tool-call part payload (function call + thought signature) when available."""
    raw = _as_str_object_dict(part.provider_raw)
    if raw is None:
        return None

    fc_raw = raw.get("function_call")
    if fc_raw is None:
        fc_raw = raw.get("functionCall")
    fc = _as_str_object_dict(fc_raw)
    if fc is None:
        return None

    normalized = _to_json_compatible(fc)
    if not isinstance(normalized, dict):
        return None
    function_call = {str(key): value for key, value in normalized.items()}

    if "name" not in function_call and part.tool_name:
        function_call["name"] = part.tool_name
    if "id" not in function_call and part.tool_call_id:
        function_call["id"] = part.tool_call_id
    if "args" not in function_call and "arguments" not in function_call:
        args_value = _to_json_compatible(part.tool_args or {})
        function_call["args"] = args_value if isinstance(args_value, dict) else {"_raw": args_value}

    name = function_call.get("name")
    if not isinstance(name, str) or not name:
        return None

    preserved_part: dict[str, object] = {"function_call": function_call}

    thought_signature = raw.get("thought_signature")
    if thought_signature is None:
        thought_signature = raw.get("thoughtSignature")
    if isinstance(thought_signature, (bytes, str)):
        preserved_part["thought_signature"] = thought_signature

    thought = raw.get("thought")
    if isinstance(thought, bool):
        preserved_part["thought"] = thought

    return preserved_part


def _tool_call_identity(part: Part) -> tuple[str, str] | None:
    """Extract ``(tool_call_id, tool_name)`` from a tool call part when available."""
    if part.tool_call_id and part.tool_name:
        return part.tool_call_id, part.tool_name

    provider_part = _tool_call_part_from_provider_raw(part)
    if provider_part is None:
        return None

    function_call = _as_str_object_dict(provider_part.get("function_call"))
    if function_call is None:
        return None

    call_id = function_call.get("id")
    name = function_call.get("name")
    if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
        return call_id, name
    return None


def _part_to_google(  # noqa: C901, PLR0911, PLR0912
    part: Part,
    role: str,
    store: BlobStore,
    tool_name_by_call_id: Mapping[str, str],
) -> dict[str, object] | None:
    """Convert a single lmctx Part to a Google GenAI part dict."""
    if part.type == "text" and part.text:
        return {"text": part.text}
    if part.type == "image" and part.blob:
        data = store.get(part.blob)
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "inline_data": {
                "mime_type": part.blob.media_type or "application/octet-stream",
                "data": b64,
            },
        }
    if part.type == "image":
        raw = _as_str_object_dict(part.provider_raw)
        if raw is not None and ("inline_data" in raw or "file_data" in raw):
            return raw
    if part.type == "file":
        return _file_part_to_google(part, store)
    if part.type == "tool_call" and role == "assistant":
        if provider_tool_call := _tool_call_part_from_provider_raw(part):
            return provider_tool_call
        if not part.tool_name:
            return None
        args_value = _to_json_compatible(part.tool_args or {})
        fc: dict[str, object] = {
            "name": part.tool_name,
            "args": args_value if isinstance(args_value, dict) else {"_raw": args_value},
        }
        if part.tool_call_id:
            fc["id"] = part.tool_call_id
        return {"function_call": fc}
    if part.type == "tool_result" and role == "tool":
        tool_name = part.tool_name
        if tool_name is None and part.tool_call_id:
            tool_name = tool_name_by_call_id.get(part.tool_call_id)
        if tool_name is None:
            raw = _as_str_object_dict(part.provider_raw)
            if raw is not None:
                fr_raw = raw.get("function_response")
                if fr_raw is None:
                    fr_raw = raw.get("functionResponse")
                fr = _as_str_object_dict(fr_raw)
                if fr is not None:
                    raw_name = fr.get("name")
                    if isinstance(raw_name, str) and raw_name:
                        tool_name = raw_name
        if tool_name is None:
            return None

        fr: dict[str, object] = {"name": tool_name, "response": _convert_tool_output(part.tool_output)}
        if part.tool_call_id:
            fr["id"] = part.tool_call_id
        return {"function_response": fr}
    if part.type == "thinking":
        if part.provider_raw:
            return dict(part.provider_raw)
        return {"text": part.text, "thought": True} if part.text else None
    return None


def _part_exclusion_reason_for_role(  # noqa: C901, PLR0911, PLR0912
    part: Part, role: str, tool_name_by_call_id: Mapping[str, str]
) -> str | None:
    """Return a reason when a part cannot be serialized for the given role."""
    if role in {"user", "assistant"}:
        if part.type == "text":
            return None if part.text else "text part is empty"
        if part.type == "image":
            if part.blob is not None:
                return None
            raw = _as_str_object_dict(part.provider_raw)
            if raw is not None and ("inline_data" in raw or "file_data" in raw):
                return None
            return "image part requires blob or provider_raw inline_data/file_data"
        if part.type == "file":
            if part.blob is not None:
                return None
            raw = _as_str_object_dict(part.provider_raw)
            if raw is None:
                return "file part requires blob or file_uri+mime_type metadata"
            uri = raw.get("file_uri")
            if not isinstance(uri, str) or not uri:
                uri = raw.get("uri")
            mime_type = raw.get("mime_type")
            if not isinstance(mime_type, str) or not mime_type:
                mime_type = raw.get("media_type")
            if isinstance(uri, str) and uri and isinstance(mime_type, str) and mime_type:
                return None
            return "file part requires blob or file_uri+mime_type metadata"
        if part.type == "tool_call" and role == "assistant":
            if _tool_call_part_from_provider_raw(part) is not None:
                return None
            if part.tool_name:
                return None
            return "tool_call part requires tool_name or provider_raw.function_call payload"
        if part.type == "thinking" and role == "assistant":
            if part.provider_raw or part.text:
                return None
            return "thinking part requires provider_raw or text"
        return f"part type '{part.type}' is not supported for role '{role}' in Google adapter"

    if role == "tool":
        if part.type == "tool_result":
            if part.tool_name:
                return None
            if part.tool_call_id and part.tool_call_id in tool_name_by_call_id:
                return None
            raw = _as_str_object_dict(part.provider_raw)
            if raw is not None:
                fr_raw = raw.get("function_response")
                if fr_raw is None:
                    fr_raw = raw.get("functionResponse")
                fr = _as_str_object_dict(fr_raw)
                if fr is not None:
                    raw_name = fr.get("name")
                    if isinstance(raw_name, str) and raw_name:
                        return None
            if not part.tool_call_id:
                return "Google function_response requires tool_name and tool_call_id."
            return (
                "Google function_response requires tool_name; provide tool_name or a prior "
                "tool_call with matching tool_call_id."
            )
        return f"tool role only supports tool_result parts for Google adapter (got '{part.type}')"

    return f"role '{role}' is not supported by Google adapter"


def _normalize_tool_args(args_raw: object) -> dict[str, object]:
    """Normalize function-call args into ``dict[str, object]``."""
    if isinstance(args_raw, Mapping):
        return {str(key): value for key, value in args_raw.items()}
    if args_raw is None:
        return {}
    return {"_raw": args_raw}


def _part_from_google_response(ctx: Context, raw_part: object) -> Part | None:  # noqa: C901
    """Convert a single Google response part into an lmctx Part."""
    raw = _as_str_object_dict(raw_part)
    if raw is None:
        return None

    text_value = raw.get("text")
    if isinstance(text_value, str):
        if raw.get("thought") is True:
            return Part(type="thinking", text=text_value, provider_raw=raw)
        return Part(type="text", text=text_value)

    inline_data_raw = raw.get("inline_data")
    if inline_data_raw is None:
        inline_data_raw = raw.get("inlineData")
    inline_data = _as_str_object_dict(inline_data_raw)
    if inline_data is not None:
        encoded = inline_data.get("data")
        mime_type = inline_data.get("mime_type")
        if not isinstance(mime_type, str) or not mime_type:
            mime_type = inline_data.get("mimeType")
        data = _decode_inline_data_bytes(encoded)
        if data is not None:
            part_type, blob_kind = _inline_data_part_kind(mime_type if isinstance(mime_type, str) else None)
            blob = ctx.blob_store.put(
                data,
                media_type=mime_type if isinstance(mime_type, str) and mime_type else "application/octet-stream",
                kind=blob_kind,
            )
            return Part(type=part_type, blob=blob, provider_raw=raw)

    fc_raw = raw.get("function_call")
    if fc_raw is None:
        fc_raw = raw.get("functionCall")
    fc = _as_str_object_dict(fc_raw)
    if fc is None:
        return None

    args_raw = fc.get("args")
    if args_raw is None:
        args_raw = fc.get("arguments")

    call_id = fc.get("id")
    tool_name = fc.get("name")
    return Part(
        type="tool_call",
        tool_call_id=call_id if isinstance(call_id, str) else None,
        tool_name=tool_name if isinstance(tool_name, str) else None,
        tool_args=_normalize_tool_args(args_raw),
        provider_raw=raw,
    )


def _build_contents(  # noqa: C901, PLR0912
    ctx: Context,
) -> tuple[str | None, list[_GoogleContentItem], tuple[ExcludedItem, ...]]:
    """Build Google system instruction and contents list from Context.

    Google uses ``model`` instead of ``assistant`` and merges consecutive
    same-role messages.
    """
    system_parts: list[str] = []
    merged: list[_GoogleContentItem] = []
    excluded: list[ExcludedItem] = []
    tool_name_by_call_id: dict[str, str] = {}

    for msg_index, msg in enumerate(ctx.messages):
        if msg.role in ("system", "developer"):
            system_texts: list[str] = []
            for part_index, part in enumerate(msg.parts):
                if part.type == "text" and part.text:
                    system_texts.append(part.text)
                elif part.type == "text":
                    excluded.append(
                        ExcludedItem(
                            description=f"context.messages[{msg_index}].parts[{part_index}]",
                            reason=f"{msg.role} text part is empty",
                        )
                    )
                else:
                    excluded.append(
                        ExcludedItem(
                            description=f"context.messages[{msg_index}].parts[{part_index}]",
                            reason=f"{msg.role} role only supports text parts for Google system instruction",
                        )
                    )
            if system_texts:
                system_parts.extend(system_texts)
            else:
                excluded.append(
                    ExcludedItem(
                        description=f"context.messages[{msg_index}]",
                        reason=f"{msg.role} message has no text content for Google system instruction",
                    )
                )
            continue

        if msg.role == "assistant":
            for part in msg.parts:
                if part.type != "tool_call":
                    continue
                identity = _tool_call_identity(part)
                if identity is not None:
                    tool_name_by_call_id[identity[0]] = identity[1]

        # Google uses "model" for assistant, "user" for user and tool.
        google_role = "model" if msg.role == "assistant" else "user"
        parts: list[dict[str, object]] = []
        for part_index, part in enumerate(msg.parts):
            block = _part_to_google(part, msg.role, ctx.blob_store, tool_name_by_call_id)
            if block is None:
                reason = _part_exclusion_reason_for_role(part, msg.role, tool_name_by_call_id)
                if reason is not None:
                    excluded.append(
                        ExcludedItem(
                            description=f"context.messages[{msg_index}].parts[{part_index}]",
                            reason=reason,
                        )
                    )
                continue

            function_call = _as_str_object_dict(block.get("function_call"))
            if function_call is not None:
                call_id = function_call.get("id")
                call_name = function_call.get("name")
                if isinstance(call_id, str) and call_id and isinstance(call_name, str) and call_name:
                    tool_name_by_call_id[call_id] = call_name

            parts.append(block)

        if not parts:
            excluded.append(
                ExcludedItem(
                    description=f"context.messages[{msg_index}]",
                    reason=f"{msg.role} message has no Google-compatible parts",
                )
            )
            continue

        # Merge consecutive same-role messages.
        if merged and merged[-1]["role"] == google_role:
            merged[-1]["parts"].extend(parts)
        else:
            merged.append({"role": google_role, "parts": parts})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, merged, tuple(excluded)


def _merge_system_instruction(spec: RunSpec, ctx_system: str | None) -> str | None:
    """Merge RunSpec instructions and Context system messages."""
    parts: list[str] = []
    if spec.instructions:
        if spec.instructions.system:
            parts.append(spec.instructions.system)
        if spec.instructions.developer:
            parts.append(spec.instructions.developer)
    if ctx_system:
        parts.append(ctx_system)
    return "\n\n".join(parts) if parts else None


def _build_config(spec: RunSpec, system: str | None) -> dict[str, object]:
    """Build the config dict from RunSpec and system instructions."""
    config: dict[str, object] = {}

    system_instruction = _merge_system_instruction(spec, system)
    if system_instruction:
        config["system_instruction"] = system_instruction

    if spec.max_output_tokens is not None:
        config["max_output_tokens"] = spec.max_output_tokens
    if spec.temperature is not None:
        config["temperature"] = spec.temperature
    if spec.top_p is not None:
        config["top_p"] = spec.top_p
    if spec.seed is not None:
        config["seed"] = spec.seed
    if spec.response_modalities:
        config["response_modalities"] = list(spec.response_modalities)
    if spec.response_schema is not None:
        config["response_schema"] = spec.response_schema
        config["response_mime_type"] = "application/json"

    if spec.tools:
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                    for tool in spec.tools
                ],
            },
        ]
    if spec.tool_choice is not None:
        config["tool_config"] = spec.tool_choice

    return config


_CAPABILITIES = AdapterCapabilities(
    id=AdapterId(provider="google", endpoint="models.generate_content"),
    fields={
        "instructions": "yes",
        "max_output_tokens": "yes",
        "temperature": "yes",
        "top_p": "yes",
        "seed": "yes",
        "tools": "yes",
        "tool_choice": "yes",
        "response_schema": "yes",
        "response_modalities": "yes",
        "extra_body": "yes",
        "extra_headers": "no",
        "extra_query": "no",
        "cursor_chaining": "no",
    },
    notes={
        "extra_headers": "Per-request transport headers are not mapped in this adapter.",
        "extra_query": "Per-request query overrides are not mapped in this adapter.",
        "cursor_chaining": "generate_content is stateless in this adapter.",
    },
)


class GoogleGenAIAdapter:
    """Adapter for the Google GenAI (Gemini) API.

    Usage::

        from google import genai
        client = genai.Client(api_key="...")
        plan = adapter.plan(ctx, spec)
        response = client.models.generate_content(**plan.request)
        ctx = adapter.ingest(ctx, response, spec=spec)
    """

    id = AdapterId(provider="google", endpoint="models.generate_content")

    def capabilities(self) -> AdapterCapabilities:
        """Return capability metadata for this adapter."""
        return _CAPABILITIES

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:  # noqa: C901
        """Build a Google GenAI request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        included: list[str] = []
        excluded: list[ExcludedItem] = []
        errors: list[str] = []

        ctx_system, contents, conversion_excluded = _build_contents(ctx)
        excluded.extend(conversion_excluded)
        included.append(f"{len(ctx)} messages")

        config = _build_config(spec, ctx_system)
        if spec.instructions:
            if spec.instructions.system:
                included.append("system instruction")
            if spec.instructions.developer:
                included.append("developer instruction")
        if spec.tools:
            included.append(f"{len(spec.tools)} tools")
        if spec.response_modalities:
            included.append("response_modalities")
        if spec.response_schema is not None:
            included.append("response_schema")

        request: dict[str, object] = {
            "model": spec.model,
            "contents": contents,
        }
        if config:
            request["config"] = config

        if not contents:
            errors.append("Google generate_content requires at least one message with Google-compatible content.")

        # Deep-merge extra_body into config.
        if spec.extra_body:
            config_obj = request.get("config")
            config = {str(key): value for key, value in config_obj.items()} if isinstance(config_obj, Mapping) else {}
            request["config"] = _deep_merge(config, spec.extra_body)

        if spec.extra_headers:
            excluded.append(
                ExcludedItem(
                    description="extra_headers",
                    reason="Google generate_content does not accept per-request extra headers in this adapter",
                )
            )
        if spec.extra_query:
            excluded.append(
                ExcludedItem(
                    description="extra_query",
                    reason="Google generate_content does not accept per-request query overrides in this adapter",
                )
            )

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            errors=tuple(errors),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: object | dict[str, object], *, spec: RunSpec) -> Context:
        """Parse a Google GenAI response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        candidates = data.get("candidates", [])
        if not candidates:
            return ctx

        content = candidates[0].get("content", {})
        raw_parts = content.get("parts", [])
        parts: list[Part] = []

        for raw_part in raw_parts:
            part = _part_from_google_response(ctx, raw_part)
            if part is not None:
                parts.append(part)

        if parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(parts),
                    provider="google",
                )
            )

        # Extract usage (snake_case from SDK, camelCase from REST).
        usage = data.get("usage_metadata") or data.get("usageMetadata")
        if isinstance(usage, Mapping):
            input_tokens, output_tokens, total_tokens = _normalized_usage_tokens(usage)
            ctx = ctx.with_usage(
                Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    provider_usage=_usage_payload(usage),
                )
            )

        return ctx
