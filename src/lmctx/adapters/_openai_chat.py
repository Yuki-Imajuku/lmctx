"""OpenAI Chat Completions adapter.

Converts between lmctx Context/RunSpec and the OpenAI Chat Completions
API format. Also works with OpenAI-compatible endpoints (vLLM, Azure, etc.).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmctx.adapters._util import (
    _deep_merge,
    _json_object,
    _plan_extra_hints,
    _to_dict,
    _to_json_compatible,
    _validate_adapter_spec,
)
from lmctx.plan import AdapterId, ExcludedItem, RequestPlan
from lmctx.types import Message, Part, Usage

if TYPE_CHECKING:
    from lmctx.blobs import BlobReference, BlobStore
    from lmctx.context import Context
    from lmctx.spec import RunSpec


def _blob_to_data_url(store: BlobStore, blob: BlobReference) -> str:
    data = store.get(blob)
    b64 = base64.b64encode(data).decode("ascii")
    media_type = blob.media_type or "application/octet-stream"
    return f"data:{media_type};base64,{b64}"


def _convert_tool_output(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(_to_json_compatible(output), ensure_ascii=False)


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    """Convert an object to ``dict[str, object]`` when possible."""
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _file_block_to_openai(part: Part, store: BlobStore) -> dict[str, object] | None:
    """Convert ``Part(type='file')`` to a Chat Completions file content block."""
    file_obj: dict[str, object] = {}

    if part.text:
        file_obj["file_id"] = part.text

    raw = _as_str_object_dict(part.provider_raw)
    if raw is not None:
        for key in ("file_id", "file_data", "filename"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                file_obj[key] = value

    if part.blob and "file_id" not in file_obj and "file_data" not in file_obj:
        data = store.get(part.blob)
        file_obj["file_data"] = base64.b64encode(data).decode("ascii")
        if "filename" not in file_obj:
            file_obj["filename"] = "upload.bin"

    return {"type": "file", "file": file_obj} if file_obj else None


def _usage_int(usage_data: Mapping[str, object], key: str) -> int | None:
    """Read an integer usage field."""
    value = usage_data.get(key)
    return value if isinstance(value, int) else None


def _usage_payload(usage_data: Mapping[str, object]) -> dict[str, object]:
    """Copy full provider usage payload with stable string keys."""
    return _json_object(usage_data)


def _tool_results_to_openai(msg: Message) -> list[dict[str, object]]:
    """Convert tool result parts into individual OpenAI tool messages."""
    return [
        {
            "role": "tool",
            "tool_call_id": part.tool_call_id,
            "content": _convert_tool_output(part.tool_output),
        }
        for part in msg.parts
        if part.type == "tool_result" and part.tool_call_id is not None
    ]


def _extract_reasoning_part(message_data: dict[str, object]) -> Part | None:
    """Extract provider reasoning payload from assistant message when present."""
    reasoning_payload: dict[str, object] = {}
    for key in ("reasoning_content", "reasoning", "reasoning_details"):
        value = message_data.get(key)
        if value is not None:
            reasoning_payload[key] = value

    if not reasoning_payload:
        return None

    thinking_text = reasoning_payload.get("reasoning_content")
    if not isinstance(thinking_text, str):
        thinking_text = reasoning_payload.get("reasoning")

    return Part(
        type="thinking",
        text=thinking_text if isinstance(thinking_text, str) else None,
        provider_raw=reasoning_payload,
    )


def _normalize_tool_arguments(raw_arguments: object) -> dict[str, object]:
    """Normalize provider tool arguments into ``dict[str, object]``."""
    if isinstance(raw_arguments, Mapping):
        return {str(key): value for key, value in raw_arguments.items()}

    if isinstance(raw_arguments, str):
        try:
            parsed_args = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": raw_arguments}
        if isinstance(parsed_args, dict):
            return {str(key): value for key, value in parsed_args.items()}
        return {"_raw": raw_arguments}

    if raw_arguments is None:
        return {}
    return {"_raw": _to_json_compatible(raw_arguments)}


def _extract_tool_call_parts(message_data: dict[str, object]) -> list[Part]:
    """Extract tool calls from assistant message."""
    parts: list[Part] = []
    raw_tool_calls = message_data.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return parts

    for raw_tc in raw_tool_calls:
        tc = _as_str_object_dict(raw_tc)
        if tc is None:
            continue

        func = _as_str_object_dict(tc.get("function")) or {}
        tool_args = _normalize_tool_arguments(func.get("arguments"))

        tool_call_id = tc.get("id")
        tool_name = func.get("name")
        parts.append(
            Part(
                type="tool_call",
                tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                tool_name=tool_name if isinstance(tool_name, str) else None,
                tool_args=tool_args,
                provider_raw=tc,
            )
        )

    return parts


def _assistant_to_openai(msg: Message) -> dict[str, object]:
    """Convert an assistant message to OpenAI format."""
    oai_msg: dict[str, object] = {"role": "assistant"}

    text_parts = [p for p in msg.parts if p.type == "text" and p.text]
    tool_call_parts = [p for p in msg.parts if p.type == "tool_call" and p.tool_call_id and p.tool_name]
    thinking_parts = [p for p in msg.parts if p.type == "thinking"]

    if text_parts:
        texts = [p.text for p in text_parts if p.text]
        oai_msg["content"] = texts[0] if len(texts) == 1 else " ".join(texts)
    else:
        oai_msg["content"] = None

    if tool_call_parts:
        oai_msg["tool_calls"] = [
            {
                "id": p.tool_call_id,
                "type": "function",
                "function": {
                    "name": p.tool_name,
                    "arguments": json.dumps(_to_json_compatible(p.tool_args or {}), ensure_ascii=False),
                },
            }
            for p in tool_call_parts
        ]

    for part in thinking_parts:
        raw = part.provider_raw
        if isinstance(raw, Mapping):
            reasoning_content = raw.get("reasoning_content")
            if reasoning_content is not None:
                oai_msg["reasoning_content"] = reasoning_content
            reasoning = raw.get("reasoning")
            if reasoning is not None:
                oai_msg["reasoning"] = reasoning
            reasoning_details = raw.get("reasoning_details")
            if reasoning_details is not None:
                oai_msg["reasoning_details"] = reasoning_details
        elif part.text:
            oai_msg["reasoning_content"] = part.text
        else:
            continue
        # Preserve the first reasoning payload block only.
        break

    return oai_msg


def _content_to_openai(msg: Message, store: BlobStore) -> dict[str, object]:
    """Convert a user/system/developer message to OpenAI format."""
    content_blocks: list[dict[str, object]] = []
    for part in msg.parts:
        if part.type == "text" and part.text:
            content_blocks.append({"type": "text", "text": part.text})
        elif part.type == "image" and part.blob:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _blob_to_data_url(store, part.blob)},
                }
            )
        elif part.type == "file":
            file_block = _file_block_to_openai(part, store)
            if file_block is not None:
                content_blocks.append(file_block)

    # Simplify single text-only messages to a plain string.
    if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
        text_value = content_blocks[0].get("text")
        content: str | list[dict[str, object]] = text_value if isinstance(text_value, str) else ""
    else:
        content = content_blocks

    return {"role": msg.role, "content": content}


def _message_to_openai(msg: Message, store: BlobStore) -> list[dict[str, object]]:
    """Convert a single lmctx Message to one or more OpenAI message dicts."""
    if msg.role == "tool":
        return _tool_results_to_openai(msg)
    if msg.role == "assistant":
        return [_assistant_to_openai(msg)]
    return [_content_to_openai(msg, store)]


def _part_exclusion_reason_for_role(part: Part, role: str) -> str | None:  # noqa: C901, PLR0911, PLR0912
    """Return a reason when a part cannot be serialized for the given role."""
    if role in {"user", "system", "developer"}:
        if part.type == "text":
            return None if part.text else "text part is empty"
        if part.type == "image":
            return None if part.blob is not None else "image part requires blob"
        if part.type == "file":
            raw = _as_str_object_dict(part.provider_raw)
            if part.blob is not None:
                return None
            if isinstance(part.text, str) and part.text:
                return None
            if raw is not None:
                for key in ("file_id", "file_data", "filename"):
                    value = raw.get(key)
                    if isinstance(value, str) and value:
                        return None
            return "file part requires blob, file_id, or file_data metadata"
        return f"part type '{part.type}' is not supported for role '{role}' in chat.completions"

    if role == "assistant":
        if part.type == "text":
            return None if part.text else "assistant text part is empty"
        if part.type == "tool_call":
            if part.tool_call_id and part.tool_name:
                return None
            return "assistant tool_call part requires tool_call_id and tool_name"
        if part.type == "thinking":
            if isinstance(part.provider_raw, Mapping):
                return None
            if part.text:
                return None
            return "assistant thinking part requires provider_raw or text"
        return f"assistant role does not support part type '{part.type}' in chat.completions"

    if role == "tool":
        if part.type == "tool_result":
            return None if part.tool_call_id else "tool_result part is missing tool_call_id"
        return f"tool role only supports tool_result parts in chat.completions (got '{part.type}')"

    return f"role '{role}' is not supported by chat.completions"


def _validate_planned_message(message: dict[str, object]) -> str | None:  # noqa: C901, PLR0911
    """Validate a converted Chat Completions message payload."""
    role = message.get("role")

    if role in {"user", "system", "developer"}:
        content = message.get("content")
        if isinstance(content, str) and content:
            return None
        if isinstance(content, list) and content:
            return None
        return f"{role} message has no serializable content"

    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return None

        content = message.get("content")
        if isinstance(content, str) and content:
            return None
        if isinstance(content, list) and content:
            return None
        for key in ("reasoning_content", "reasoning", "reasoning_details"):
            if message.get(key) is not None:
                return None
        return "assistant message has neither content nor tool_calls"

    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return "tool message is missing tool_call_id"
        content = message.get("content")
        if not isinstance(content, str):
            return "tool message content must be a string"
        return None

    return "message role is not supported by chat.completions"


def _apply_generation_params(request: dict[str, object], spec: RunSpec) -> None:
    """Apply optional generation parameters from RunSpec to the request dict."""
    if spec.max_output_tokens is not None:
        request["max_tokens"] = spec.max_output_tokens
    if spec.temperature is not None:
        request["temperature"] = spec.temperature
    if spec.top_p is not None:
        request["top_p"] = spec.top_p
    if spec.seed is not None:
        request["seed"] = spec.seed
    if spec.response_modalities:
        request["modalities"] = list(spec.response_modalities)


def _parse_message_content_block(block: dict[str, object]) -> list[Part]:
    """Parse a single assistant content block into lmctx parts."""
    block_type = block.get("type")
    if block_type in {"text", "output_text"}:
        text = block.get("text")
        if isinstance(text, str):
            return [Part(type="text", text=text)]
        return []

    if block_type in {"image", "image_url"}:
        return [Part(type="image", provider_raw=block)]
    return []


def _extract_text_parts(message_data: dict[str, object]) -> list[Part]:
    """Extract assistant text/image content from message payload."""
    content = message_data.get("content")
    if isinstance(content, str):
        return [Part(type="text", text=content)] if content else []

    if isinstance(content, list):
        parts: list[Part] = []
        for raw_block in content:
            block = _as_str_object_dict(raw_block)
            if block is None:
                continue
            parts.extend(_parse_message_content_block(block))
        return parts

    return []


def _apply_tools(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply tool definitions and tool_choice from RunSpec to the request dict."""
    if spec.tools:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in spec.tools
        ]
        included.append(f"{len(spec.tools)} tools")
    if spec.tool_choice is not None:
        request["tool_choice"] = spec.tool_choice


def _apply_transport_overrides(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply optional transport-level overrides accepted by OpenAI SDK."""
    if spec.extra_headers:
        request["extra_headers"] = dict(spec.extra_headers)
        included.append("extra_headers")
    if spec.extra_query:
        request["extra_query"] = dict(spec.extra_query)
        included.append("extra_query")


class OpenAIChatCompletionsAdapter:
    """Adapter for OpenAI Chat Completions API (and compatible endpoints)."""

    id = AdapterId(provider="openai", endpoint="chat.completions")

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:  # noqa: C901, PLR0912
        """Build an OpenAI Chat Completions request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        messages: list[dict[str, object]] = []
        included: list[str] = []
        excluded: list[ExcludedItem] = []
        errors: list[str] = []

        # Instructions as system/developer messages.
        if spec.instructions:
            if spec.instructions.system:
                messages.append({"role": "system", "content": spec.instructions.system})
                included.append("system instruction")
            if spec.instructions.developer:
                messages.append({"role": "developer", "content": spec.instructions.developer})
                included.append("developer instruction")

        # Convert context messages.
        for message_index, msg in enumerate(ctx.messages):
            for part_index, part in enumerate(msg.parts):
                part_error = _part_exclusion_reason_for_role(part, msg.role)
                if part_error is not None:
                    excluded.append(
                        ExcludedItem(
                            description=f"context.messages[{message_index}].parts[{part_index}]",
                            reason=part_error,
                        )
                    )

            converted_messages = _message_to_openai(msg, ctx.blob_store)
            if not converted_messages:
                excluded.append(
                    ExcludedItem(
                        description=f"context.messages[{message_index}]",
                        reason=f"{msg.role} message has no OpenAI-compatible parts",
                    )
                )
                errors.append(f"context.messages[{message_index}] produced no Chat Completions payload.")
                continue

            valid_count = 0
            for converted_index, converted in enumerate(converted_messages):
                validation_error = _validate_planned_message(converted)
                if validation_error is not None:
                    description = f"context.messages[{message_index}]"
                    if len(converted_messages) > 1:
                        description = f"{description}[{converted_index}]"
                    excluded.append(ExcludedItem(description=description, reason=validation_error))
                    continue

                messages.append(converted)
                valid_count += 1

            if valid_count == 0:
                errors.append(f"context.messages[{message_index}] produced no valid Chat Completions payload.")
        included.append(f"{len(ctx)} messages")

        request: dict[str, object] = {"model": spec.model, "messages": messages}

        _apply_generation_params(request, spec)
        if spec.response_modalities:
            included.append("response_modalities")
        _apply_tools(request, spec, included)
        _apply_transport_overrides(request, spec, included)

        # Structured output.
        if spec.response_schema is not None:
            request["response_format"] = {"type": "json_schema", "json_schema": spec.response_schema}
            included.append("response_schema")

        # Deep-merge extra_body.
        if spec.extra_body:
            request = _deep_merge(request, spec.extra_body)

        if not messages:
            errors.append("chat.completions request has no messages after conversion.")

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            errors=tuple(errors),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: object | dict[str, object], *, spec: RunSpec) -> Context:
        """Parse an OpenAI Chat Completions response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        choices = data.get("choices", [])
        if not choices:
            return ctx

        message_data = choices[0].get("message", {})
        parts: list[Part] = []

        parts.extend(_extract_text_parts(message_data))

        reasoning_part = _extract_reasoning_part(message_data)
        if reasoning_part is not None:
            parts.append(reasoning_part)

        parts.extend(_extract_tool_call_parts(message_data))

        if parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(parts),
                    id=data.get("id"),
                    provider="openai",
                )
            )

        # Extract usage.
        usage_data = data.get("usage")
        if isinstance(usage_data, Mapping):
            ctx = ctx.with_usage(
                Usage(
                    input_tokens=_usage_int(usage_data, "prompt_tokens"),
                    output_tokens=_usage_int(usage_data, "completion_tokens"),
                    total_tokens=_usage_int(usage_data, "total_tokens"),
                    provider_usage=_usage_payload(usage_data),
                )
            )

        return ctx
