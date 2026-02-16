"""Anthropic Messages API adapter.

Converts between lmctx Context/RunSpec and the Anthropic Messages API format.

Key Anthropic constraints handled here:
- ``system`` is a separate parameter, not a message
- Messages must alternate between user and assistant roles
- Tool results go inside user messages as ``tool_result`` content blocks
- ``max_tokens`` is required
"""

from __future__ import annotations

import base64
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
    from lmctx.blobs import BlobReference, BlobStore
    from lmctx.context import Context
    from lmctx.spec import RunSpec

_DEFAULT_MAX_TOKENS = 4096


class _AnthropicMessage(TypedDict):
    role: str
    content: list[dict[str, object]]


def _usage_payload(usage_data: Mapping[str, object]) -> dict[str, object]:
    """Copy full provider usage payload with stable string keys."""
    return _json_object(usage_data)


def _usage_int(value: object) -> int | None:
    """Return integer usage values when present."""
    return value if isinstance(value, int) else None


def _sum_iteration_usage(usage_data: Mapping[str, object], field: str) -> int | None:
    """Fallback for compaction usage where top-level counts can be zero."""
    iterations = usage_data.get("iterations")
    if not isinstance(iterations, list):
        return None

    total = 0
    found = False
    for item in iterations:
        if not isinstance(item, Mapping):
            continue
        mapped_item = {str(key): val for key, val in item.items()}
        value = mapped_item.get(field)
        if isinstance(value, int):
            total += value
            found = True
    return total if found else None


def _blob_to_base64(store: BlobStore, blob: BlobReference) -> str:
    data = store.get_blob(blob)
    return base64.b64encode(data).decode("ascii")


def _convert_tool_output(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(_to_json_compatible(output), ensure_ascii=False)


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    """Convert an object to ``dict[str, object]`` when possible."""
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _file_block_to_anthropic(part: Part, store: BlobStore) -> dict[str, object] | None:
    """Convert ``Part(type='file')`` to an Anthropic document block."""
    raw = _as_str_object_dict(part.provider_raw)
    file_id = raw.get("file_id") if raw is not None else None

    block: dict[str, object] | None = None
    if isinstance(file_id, str) and file_id:
        block = {
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        }
    elif part.blob is not None:
        block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": part.blob.media_type or "application/octet-stream",
                "data": _blob_to_base64(store, part.blob),
            },
        }

    if block is None:
        return None

    if raw is None:
        return block

    title = raw.get("title")
    if isinstance(title, str) and title:
        block["title"] = title

    context = raw.get("context")
    if isinstance(context, str) and context:
        block["context"] = context

    citations = _as_str_object_dict(raw.get("citations"))
    if citations is not None:
        block["citations"] = citations

    return block


def _parts_to_content(
    parts: tuple[Part, ...], role: str, store: BlobStore
) -> tuple[list[dict[str, object]], tuple[tuple[int, str], ...]]:
    """Convert lmctx Parts to Anthropic content blocks."""
    blocks: list[dict[str, object]] = []
    excluded_parts: list[tuple[int, str]] = []
    for part_index, part in enumerate(parts):
        if part.type == "text" and part.text:
            blocks.append({"type": "text", "text": part.text})
        elif part.type == "text":
            excluded_parts.append((part_index, "text part is empty"))
        elif part.type == "image" and part.blob:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.blob.media_type or "application/octet-stream",
                        "data": _blob_to_base64(store, part.blob),
                    },
                }
            )
        elif part.type == "image":
            excluded_parts.append((part_index, "image part requires blob"))
        elif part.type == "file" and role == "user":
            block = _file_block_to_anthropic(part, store)
            if block is not None:
                blocks.append(block)
            else:
                excluded_parts.append((part_index, "file part requires file_id metadata or blob content"))
        elif part.type == "tool_call" and role == "assistant" and part.tool_call_id and part.tool_name:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.tool_args or {},
                }
            )
        elif part.type == "tool_call" and role == "assistant":
            excluded_parts.append((part_index, "tool_call part requires tool_call_id and tool_name"))
        elif part.type == "tool_result" and role == "tool" and part.tool_call_id:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": part.tool_call_id,
                    "content": _convert_tool_output(part.tool_output),
                }
            )
        elif part.type == "tool_result" and role == "tool":
            excluded_parts.append((part_index, "tool_result part requires tool_call_id"))
        elif part.type == "thinking":
            # Use provider_raw to preserve signature/redacted data for round-trip.
            if part.provider_raw and part.provider_raw.get("type") in ("thinking", "redacted_thinking"):
                blocks.append(dict(part.provider_raw))
            elif part.text:
                blocks.append({"type": "thinking", "thinking": part.text})
            else:
                excluded_parts.append((part_index, "thinking part requires provider_raw or text"))
        elif part.type == "compaction":
            if part.provider_raw and part.provider_raw.get("type") == "compaction":
                blocks.append(dict(part.provider_raw))
            elif part.blob:
                raw_text = store.get_blob(part.blob).decode("utf-8", errors="replace")
                blocks.append({"type": "compaction", "content": raw_text})
            elif part.text:
                blocks.append({"type": "compaction", "content": part.text})
            else:
                excluded_parts.append((part_index, "compaction part requires provider_raw, blob, or text"))
        else:
            excluded_parts.append((part_index, f"part type '{part.type}' is not supported for role '{role}'"))
    return blocks, tuple(excluded_parts)


def _build_messages(ctx: Context) -> tuple[str | None, list[_AnthropicMessage], tuple[ExcludedItem, ...]]:
    """Build the Anthropic system string and message list from a Context.

    Handles role alternation by merging consecutive same-role messages.
    Tool messages are treated as user role for Anthropic.
    """
    system_parts: list[str] = []
    merged: list[_AnthropicMessage] = []
    excluded: list[ExcludedItem] = []

    for message_index, msg in enumerate(ctx.messages):
        if msg.role in ("system", "developer"):
            texts: list[str] = []
            for part_index, part in enumerate(msg.parts):
                if part.type == "text" and part.text:
                    texts.append(part.text)
                elif part.type == "text":
                    excluded.append(
                        ExcludedItem(
                            description=f"context.messages[{message_index}].parts[{part_index}]",
                            reason=f"{msg.role} text part is empty",
                        )
                    )
                else:
                    excluded.append(
                        ExcludedItem(
                            description=f"context.messages[{message_index}].parts[{part_index}]",
                            reason=f"{msg.role} role only supports text parts for Anthropic system field",
                        )
                    )
            if texts:
                system_parts.extend(texts)
            else:
                excluded.append(
                    ExcludedItem(
                        description=f"context.messages[{message_index}]",
                        reason=f"{msg.role} message has no text content for Anthropic system field",
                    )
                )
            continue

        anthropic_role = "user" if msg.role == "tool" else msg.role
        content, part_excluded = _parts_to_content(msg.parts, msg.role, ctx.blob_store)
        excluded.extend(
            ExcludedItem(
                description=f"context.messages[{message_index}].parts[{part_index}]",
                reason=reason,
            )
            for part_index, reason in part_excluded
        )

        if not content:
            excluded.append(
                ExcludedItem(
                    description=f"context.messages[{message_index}]",
                    reason=f"{msg.role} message has no Anthropic-compatible parts",
                )
            )
            continue

        if merged and merged[-1]["role"] == anthropic_role:
            merged[-1]["content"].extend(content)
        else:
            merged.append({"role": anthropic_role, "content": content})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, merged, tuple(excluded)


def _merge_system(spec: RunSpec, ctx_system: str | None) -> str | None:
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


def _validate_messages(messages: list[_AnthropicMessage]) -> list[str]:
    """Check Anthropic message constraints and return errors."""
    errors: list[str] = []

    if messages and messages[0]["role"] != "user":
        errors.append(f"Anthropic requires the first message to be 'user', got '{messages[0]['role']}'")

    errors.extend(
        f"Anthropic requires alternating roles, but found consecutive "
        f"'{messages[i]['role']}' at positions {i - 1} and {i}"
        for i in range(1, len(messages))
        if messages[i]["role"] == messages[i - 1]["role"]
    )

    return errors


def _apply_tools(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply tool definitions and tool_choice from RunSpec to the request dict."""
    if spec.tools:
        request["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in spec.tools
        ]
        included.append(f"{len(spec.tools)} tools")
    if spec.tool_choice is not None:
        request["tool_choice"] = spec.tool_choice


def _apply_structured_output(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply response schema settings as Anthropic ``output_config.format``."""
    if spec.response_schema is None:
        return
    schema = _to_json_compatible(spec.response_schema)
    if not isinstance(schema, dict):
        msg = "response_schema must be a JSON object for Anthropic output_config.format.schema"
        raise TypeError(msg)
    request["output_config"] = {
        "format": {
            "type": "json_schema",
            "schema": _close_anthropic_object_schemas(schema),
        }
    }
    included.append("response_schema")


def _close_anthropic_object_schemas(value: object) -> object:
    """Recursively ensure object schemas set ``additionalProperties=False``.

    Anthropic requires every ``type: object`` schema node used with
    ``output_config.format.schema`` to explicitly set
    ``additionalProperties`` to ``false``.
    """
    if isinstance(value, Mapping):
        closed: dict[str, object] = {str(key): _close_anthropic_object_schemas(item) for key, item in value.items()}
        schema_type = closed.get("type")
        if schema_type == "object":
            closed["additionalProperties"] = False
        return closed
    if isinstance(value, list):
        return [_close_anthropic_object_schemas(item) for item in value]
    return value


def _apply_transport_overrides(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply optional transport-level overrides accepted by Anthropic SDK."""
    if spec.extra_headers:
        request["extra_headers"] = dict(spec.extra_headers)
        included.append("extra_headers")
    if spec.extra_query:
        request["extra_query"] = dict(spec.extra_query)
        included.append("extra_query")


_CAPABILITIES = AdapterCapabilities(
    id=AdapterId(provider="anthropic", endpoint="messages.create"),
    fields={
        "instructions": "yes",
        "max_output_tokens": "yes",
        "temperature": "yes",
        "top_p": "yes",
        "seed": "no",
        "tools": "yes",
        "tool_choice": "yes",
        "response_schema": "yes",
        "response_modalities": "no",
        "extra_body": "yes",
        "extra_headers": "yes",
        "extra_query": "yes",
        "cursor_chaining": "no",
    },
    notes={
        "seed": "messages.create does not support deterministic sampling seed.",
        "response_modalities": "messages.create does not expose output modalities.",
        "cursor_chaining": "messages.create is stateless in this adapter.",
    },
)


class AnthropicMessagesAdapter:
    """Adapter for the Anthropic Messages API."""

    id = AdapterId(provider="anthropic", endpoint="messages.create")

    def capabilities(self) -> AdapterCapabilities:
        """Return capability metadata for this adapter."""
        return _CAPABILITIES

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build an Anthropic Messages API request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        included: list[str] = []
        excluded: list[ExcludedItem] = []
        warnings: list[str] = []

        ctx_system, messages, conversion_excluded = _build_messages(ctx)
        excluded.extend(conversion_excluded)
        included.append(f"{len(ctx)} messages")

        combined_system = _merge_system(spec, ctx_system)
        if spec.instructions:
            if spec.instructions.system:
                included.append("system instruction")
            if spec.instructions.developer:
                included.append("developer instruction")

        request: dict[str, object] = {
            "model": spec.model,
            "max_tokens": spec.max_output_tokens or _DEFAULT_MAX_TOKENS,
            "messages": messages,
        }

        if combined_system:
            request["system"] = combined_system

        if spec.temperature is not None:
            request["temperature"] = spec.temperature
        if spec.top_p is not None:
            request["top_p"] = spec.top_p

        if spec.max_output_tokens is None:
            warnings.append(f"max_output_tokens not set; defaulting to {_DEFAULT_MAX_TOKENS}")
        if spec.response_modalities:
            warnings.append("response_modalities is not supported by messages.create; ignoring the field")
        if spec.seed is not None:
            excluded.append(
                ExcludedItem(
                    description="seed",
                    reason="Anthropic messages.create does not support deterministic sampling seed",
                )
            )

        _apply_tools(request, spec, included)
        _apply_structured_output(request, spec, included)
        _apply_transport_overrides(request, spec, included)

        errors = _validate_messages(messages)
        if not messages:
            errors.append("Anthropic messages.create requires at least one valid user/assistant message.")

        if spec.extra_body:
            request = _deep_merge(request, spec.extra_body)

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            warnings=tuple(warnings),
            errors=tuple(errors),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: object | dict[str, object], *, spec: RunSpec) -> Context:
        """Parse an Anthropic Messages API response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        content_blocks = data.get("content", [])
        parts: list[Part] = []

        for raw_block in content_blocks:
            block = _as_str_object_dict(raw_block)
            if block is None:
                continue
            block_type = block.get("type")

            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(Part(type="text", text=text))
            elif block_type == "tool_use":
                tool_call_id = block.get("id")
                tool_name = block.get("name")
                tool_args = _as_str_object_dict(block.get("input"))
                parts.append(
                    Part(
                        type="tool_call",
                        tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
                        tool_name=tool_name if isinstance(tool_name, str) else None,
                        tool_args=tool_args,
                        provider_raw=block,
                    )
                )
            elif block_type == "thinking":
                thinking_text = block.get("thinking")
                parts.append(
                    Part(
                        type="thinking",
                        text=thinking_text if isinstance(thinking_text, str) else None,
                        provider_raw=block,
                    )
                )
            elif block_type == "redacted_thinking":
                parts.append(Part(type="thinking", provider_raw=block))
            elif block_type == "compaction":
                content = block.get("content")
                if not isinstance(content, str):
                    content = block.get("encrypted_content")
                if isinstance(content, str):
                    blob = ctx.blob_store.put_blob(
                        content.encode("utf-8"),
                        media_type="application/octet-stream",
                        kind="compaction",
                    )
                    parts.append(Part(type="compaction", blob=blob, provider_raw=block))
                else:
                    parts.append(Part(type="compaction", provider_raw=block))

        if parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(parts),
                    id=data.get("id"),
                    provider="anthropic",
                )
            )

        usage_data = data.get("usage")
        if isinstance(usage_data, Mapping):
            input_tokens = _usage_int(usage_data.get("input_tokens"))
            output_tokens = _usage_int(usage_data.get("output_tokens"))
            total_tokens = _usage_int(usage_data.get("total_tokens"))

            if input_tokens in (None, 0):
                iter_input = _sum_iteration_usage(usage_data, "input_tokens")
                if iter_input is not None:
                    input_tokens = iter_input
            if output_tokens in (None, 0):
                iter_output = _sum_iteration_usage(usage_data, "output_tokens")
                if iter_output is not None:
                    output_tokens = iter_output

            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

            ctx = ctx.with_usage(
                Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    provider_usage=_usage_payload(usage_data),
                )
            )

        return ctx
