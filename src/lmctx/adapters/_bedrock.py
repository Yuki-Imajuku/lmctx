"""AWS Bedrock Converse adapter.

Converts between lmctx Context/RunSpec and the AWS Bedrock Converse API format.
Uses ``boto3``'s ``bedrock-runtime`` client.

Key differences from other adapters:
- Uses camelCase keys (``modelId``, ``inferenceConfig``, ``toolConfig``)
- System messages go in a separate ``system`` parameter as content blocks
- Tool definitions use ``toolSpec`` with nested ``inputSchema.json``
- Tool results use ``toolResult`` blocks with ``toolUseId``
- Images use raw bytes (not base64) via ``image.source.bytes``
"""

from __future__ import annotations

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


class _BedrockMessage(TypedDict):
    role: str
    content: list[dict[str, object]]


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    """Convert an object to ``dict[str, object]`` when possible."""
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _usage_payload(usage_data: Mapping[str, object]) -> dict[str, object]:
    """Copy full provider usage payload with stable string keys."""
    return _json_object(usage_data)


def _usage_int(usage_data: Mapping[str, object], key: str) -> int | None:
    """Read an integer usage field."""
    value = usage_data.get(key)
    return value if isinstance(value, int) else None


def _media_type_to_format(media_type: str | None) -> str:
    """Convert MIME type to Bedrock image format string."""
    if media_type is None:
        return "png"
    suffix = media_type.rsplit("/", 1)[-1].lower()
    return {"jpeg": "jpeg", "jpg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "png")


def _media_type_to_document_format(media_type: str | None) -> str:
    """Convert MIME type to Bedrock document format string."""
    if media_type is None:
        return "txt"

    suffix = media_type.rsplit("/", 1)[-1].lower()
    return {
        "pdf": "pdf",
        "plain": "txt",
        "txt": "txt",
        "markdown": "md",
        "md": "md",
        "html": "html",
        "csv": "csv",
    }.get(suffix, "txt")


def _sanitize_document_name(name: str) -> str:
    """Make a Bedrock-safe document name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "document"


def _convert_tool_output(output: object) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(_to_json_compatible(output), ensure_ascii=False)


def _normalize_tool_choice_string(tool_choice: str) -> tuple[dict[str, object] | None, str | None]:
    """Normalize string tool_choice modes to Bedrock toolChoice objects."""
    mode = tool_choice.strip().lower()
    if mode == "auto":
        return {"auto": {}}, None
    if mode in {"required", "any"}:
        return {"any": {}}, None
    if mode == "none":
        return None, "Bedrock Converse toolChoice does not support 'none'."
    return None, f"unsupported string tool_choice value: {tool_choice!r}"


def _normalize_tool_choice_mapping(
    tool_choice: Mapping[str, object],
) -> tuple[dict[str, object] | None, str | None]:
    """Normalize mapping tool_choice forms to Bedrock toolChoice objects."""
    choice = {str(key): item for key, item in tool_choice.items()}

    if any(key in choice for key in ("auto", "any", "tool")):
        return _json_object(choice), None

    raw_type = choice.get("type")
    if isinstance(raw_type, str):
        if raw_type.lower() == "function":
            function_obj = _as_str_object_dict(choice.get("function"))
            function_name = function_obj.get("name") if function_obj is not None else None
            if isinstance(function_name, str) and function_name:
                return {"tool": {"name": function_name}}, None
            return None, "tool_choice type=function requires function.name."
        return _normalize_tool_choice_string(raw_type)

    function_name = choice.get("name")
    if isinstance(function_name, str) and function_name:
        return {"tool": {"name": function_name}}, None

    return None, "tool_choice format is not recognized for Bedrock Converse."


def _normalize_tool_choice(tool_choice: object) -> tuple[dict[str, object] | None, str | None]:
    """Normalize RunSpec tool_choice into Bedrock ``toolConfig.toolChoice`` format."""
    if isinstance(tool_choice, str):
        return _normalize_tool_choice_string(tool_choice)

    choice = _as_str_object_dict(tool_choice)
    if choice is None:
        return None, "tool_choice must be a string or mapping."
    return _normalize_tool_choice_mapping(choice)


def _parts_to_bedrock(
    parts: tuple[Part, ...], role: str, store: BlobStore
) -> tuple[list[dict[str, object]], tuple[tuple[int, str], ...]]:
    """Convert lmctx Parts to Bedrock content blocks."""
    blocks: list[dict[str, object]] = []
    excluded_parts: list[tuple[int, str]] = []
    for part_index, part in enumerate(parts):
        if part.type == "text" and part.text:
            blocks.append({"text": part.text})
        elif part.type == "text":
            excluded_parts.append((part_index, "text part is empty"))
        elif part.type == "image" and part.blob:
            data = store.get_blob(part.blob)
            blocks.append(
                {
                    "image": {
                        "format": _media_type_to_format(part.blob.media_type),
                        "source": {"bytes": data},
                    },
                }
            )
        elif part.type == "image":
            excluded_parts.append((part_index, "image part requires blob"))
        elif part.type == "file" and part.blob and role == "user":
            data = store.get_blob(part.blob)
            raw = _as_str_object_dict(part.provider_raw)
            raw_name = raw.get("name") if raw is not None else None
            name = raw_name if isinstance(raw_name, str) and raw_name else "document"
            blocks.append(
                {
                    "document": {
                        "format": _media_type_to_document_format(part.blob.media_type),
                        "name": _sanitize_document_name(name),
                        "source": {"bytes": data},
                    }
                }
            )
        elif part.type == "file" and role == "user":
            excluded_parts.append((part_index, "file part requires blob for Bedrock document input"))
        elif part.type == "tool_call" and role == "assistant" and part.tool_call_id and part.tool_name:
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": part.tool_call_id,
                        "name": part.tool_name,
                        "input": part.tool_args or {},
                    },
                }
            )
        elif part.type == "tool_call" and role == "assistant":
            excluded_parts.append((part_index, "tool_call part requires tool_call_id and tool_name"))
        elif part.type == "tool_result" and role == "tool" and part.tool_call_id:
            blocks.append(
                {
                    "toolResult": {
                        "toolUseId": part.tool_call_id,
                        "content": [{"text": _convert_tool_output(part.tool_output)}],
                        "status": "success",
                    },
                }
            )
        elif part.type == "tool_result" and role == "tool":
            excluded_parts.append((part_index, "tool_result part requires tool_call_id"))
        elif part.type == "thinking" and role == "assistant":
            # Preserve reasoning content with signature for round-trip.
            if part.provider_raw and "reasoningContent" in part.provider_raw:
                blocks.append(dict(part.provider_raw))
            elif part.text:
                blocks.append({"reasoningContent": {"reasoningText": {"text": part.text}}})
            else:
                excluded_parts.append((part_index, "thinking part requires provider_raw or text"))
        else:
            excluded_parts.append((part_index, f"part type '{part.type}' is not supported for role '{role}'"))
    return blocks, tuple(excluded_parts)


def _build_messages(ctx: Context) -> tuple[list[dict[str, object]], list[_BedrockMessage], tuple[ExcludedItem, ...]]:
    """Build Bedrock system blocks and message list from Context.

    Tool messages are folded into user messages. Consecutive same-role
    messages are merged.
    """
    system_blocks: list[dict[str, object]] = []
    merged: list[_BedrockMessage] = []
    excluded: list[ExcludedItem] = []

    for message_index, msg in enumerate(ctx.messages):
        if msg.role in ("system", "developer"):
            for part_index, part in enumerate(msg.parts):
                if part.type == "text" and part.text:
                    system_blocks.append({"text": part.text})
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
                            reason=f"{msg.role} role only supports text parts for Bedrock system blocks",
                        )
                    )
            continue

        bedrock_role = "user" if msg.role == "tool" else msg.role
        content, part_excluded = _parts_to_bedrock(msg.parts, msg.role, ctx.blob_store)
        excluded.extend(
            ExcludedItem(
                description=f"context.messages[{message_index}].parts[{part_index}]",
                reason=reason,
            )
            for part_index, reason in part_excluded
        )

        if not content:
            continue

        if merged and merged[-1]["role"] == bedrock_role:
            merged[-1]["content"].extend(content)
        else:
            merged.append({"role": bedrock_role, "content": content})

    return system_blocks, merged, tuple(excluded)


def _build_inference_config(spec: RunSpec) -> dict[str, object]:
    """Build the inferenceConfig dict from RunSpec."""
    config: dict[str, object] = {}
    if spec.max_output_tokens is not None:
        config["maxTokens"] = spec.max_output_tokens
    if spec.temperature is not None:
        config["temperature"] = spec.temperature
    if spec.top_p is not None:
        config["topP"] = spec.top_p
    return config


def _extract_bedrock_schema(
    response_schema: Mapping[str, object],
) -> tuple[Mapping[str, object], str | None, str | None]:
    """Normalize ``RunSpec.response_schema`` into Bedrock schema components."""
    wrapped_schema = _as_str_object_dict(response_schema.get("schema"))
    has_wrapper_hints = any(key in response_schema for key in ("name", "description", "strict"))
    has_plain_schema_hints = any(key in response_schema for key in ("type", "$schema", "properties", "required"))

    schema_mapping: Mapping[str, object]
    schema_name: str | None = None
    schema_description: str | None = None
    if wrapped_schema is not None and (has_wrapper_hints or not has_plain_schema_hints):
        schema_mapping = wrapped_schema
        raw_name = response_schema.get("name")
        if isinstance(raw_name, str) and raw_name:
            schema_name = raw_name
        raw_description = response_schema.get("description")
        if isinstance(raw_description, str) and raw_description:
            schema_description = raw_description
    else:
        schema_mapping = response_schema

    return schema_mapping, schema_name, schema_description


def _ensure_closed_object_schemas(value: object) -> object:
    """Recursively set ``additionalProperties: false`` for object schemas when missing.

    Bedrock structured output validation requires object schemas to explicitly set
    ``additionalProperties``.
    """
    if isinstance(value, Mapping):
        normalized = {str(key): _ensure_closed_object_schemas(item) for key, item in value.items()}
        raw_type = normalized.get("type")
        is_object = raw_type == "object" or (
            isinstance(raw_type, list) and any(isinstance(item, str) and item == "object" for item in raw_type)
        )
        if is_object and "additionalProperties" not in normalized:
            normalized["additionalProperties"] = False
        return normalized
    if isinstance(value, list):
        return [_ensure_closed_object_schemas(item) for item in value]
    if isinstance(value, tuple):
        return [_ensure_closed_object_schemas(item) for item in value]
    return value


def _apply_structured_output(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply response schema as Bedrock Converse ``outputConfig.textFormat``."""
    if spec.response_schema is None:
        return

    schema_mapping, schema_name, schema_description = _extract_bedrock_schema(spec.response_schema)
    normalized_schema = _ensure_closed_object_schemas(_to_json_compatible(schema_mapping))
    json_schema: dict[str, object] = {
        "schema": json.dumps(normalized_schema, ensure_ascii=False),
    }
    if schema_name is not None:
        json_schema["name"] = schema_name
    if schema_description is not None:
        json_schema["description"] = schema_description

    request["outputConfig"] = {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": json_schema,
            },
        }
    }
    included.append("response_schema")


_CAPABILITIES = AdapterCapabilities(
    id=AdapterId(provider="bedrock", endpoint="converse"),
    fields={
        "instructions": "yes",
        "max_output_tokens": "yes",
        "temperature": "yes",
        "top_p": "yes",
        "seed": "no",
        "tools": "yes",
        "tool_choice": "partial",
        "response_schema": "yes",
        "response_modalities": "no",
        "extra_body": "yes",
        "extra_headers": "no",
        "extra_query": "no",
        "cursor_chaining": "no",
    },
    notes={
        "seed": "Deterministic seed support is model-specific and not mapped.",
        "tool_choice": (
            "Mapped to toolConfig.toolChoice. Supported by model family (for example Claude 3 and Amazon Nova)."
        ),
        "response_modalities": "Output modality controls are model-specific and not mapped.",
        "extra_headers": "Per-request transport headers are not mapped in this adapter.",
        "extra_query": "Per-request query overrides are not mapped in this adapter.",
        "cursor_chaining": "converse is stateless in this adapter.",
    },
)


class BedrockConverseAdapter:
    """Adapter for the AWS Bedrock Converse API.

    Usage::

        import boto3
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        plan = adapter.plan(ctx, spec)
        response = client.converse(**plan.request)
        ctx = adapter.ingest(ctx, response, spec=spec)
    """

    id = AdapterId(provider="bedrock", endpoint="converse")

    def capabilities(self) -> AdapterCapabilities:
        """Return capability metadata for this adapter."""
        return _CAPABILITIES

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build an AWS Bedrock Converse request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        included: list[str] = []
        excluded: list[ExcludedItem] = []

        system_blocks, messages, conversion_excluded = _build_messages(ctx)
        excluded.extend(conversion_excluded)
        included.append(f"{len(ctx)} messages")

        # Merge RunSpec instructions into system blocks.
        extra_system: list[dict[str, object]] = []
        if spec.instructions:
            if spec.instructions.system:
                extra_system.append({"text": spec.instructions.system})
                included.append("system instruction")
            if spec.instructions.developer:
                extra_system.append({"text": spec.instructions.developer})
                included.append("developer instruction")

        all_system = extra_system + system_blocks

        request: dict[str, object] = {
            "modelId": spec.model,
            "messages": messages,
        }

        if all_system:
            request["system"] = all_system

        inference_config = _build_inference_config(spec)
        if inference_config:
            request["inferenceConfig"] = inference_config

        if spec.tools:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": {"json": tool.input_schema},
                        },
                    }
                    for tool in spec.tools
                ],
            }
            included.append(f"{len(spec.tools)} tools")

        if spec.tool_choice is not None:
            normalized_choice, reason = _normalize_tool_choice(spec.tool_choice)
            if normalized_choice is not None:
                tool_config_obj = request.get("toolConfig")
                tool_config = (
                    {str(key): value for key, value in tool_config_obj.items()}
                    if isinstance(tool_config_obj, Mapping)
                    else {}
                )
                tool_config["toolChoice"] = normalized_choice
                request["toolConfig"] = tool_config
                included.append("tool_choice")
            elif reason is not None:
                excluded.append(
                    ExcludedItem(
                        description="tool_choice",
                        reason=reason,
                    )
                )

        _apply_structured_output(request, spec, included)

        # Bedrock extra_body is deep-merged to preserve nested defaults.
        if spec.extra_body:
            request = _deep_merge(request, spec.extra_body)

        if spec.extra_headers:
            excluded.append(
                ExcludedItem(
                    description="extra_headers",
                    reason="Bedrock Converse does not accept per-request extra headers in this adapter",
                )
            )
        if spec.extra_query:
            excluded.append(
                ExcludedItem(
                    description="extra_query",
                    reason="Bedrock Converse does not accept per-request query overrides in this adapter",
                )
            )
        if spec.response_modalities:
            excluded.append(
                ExcludedItem(
                    description="response_modalities",
                    reason=(
                        "Bedrock Converse output modality controls are model-specific and not mapped by this adapter"
                    ),
                )
            )
        if spec.seed is not None:
            excluded.append(
                ExcludedItem(
                    description="seed",
                    reason="Bedrock Converse deterministic seed is model-specific and not mapped by this adapter",
                )
            )

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: dict[str, object], *, spec: RunSpec) -> Context:
        """Parse a Bedrock Converse response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        output = _as_str_object_dict(data.get("output")) or {}
        message_data = _as_str_object_dict(output.get("message")) or {}
        content_blocks_obj = message_data.get("content")
        content_blocks = content_blocks_obj if isinstance(content_blocks_obj, list) else []

        parts: list[Part] = []
        for raw_block in content_blocks:
            block = _as_str_object_dict(raw_block)
            if block is None:
                continue
            if "text" in block:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(Part(type="text", text=text))
            elif "toolUse" in block:
                tu = _as_str_object_dict(block["toolUse"]) or {}
                tool_use_id = tu.get("toolUseId")
                tool_name = tu.get("name")
                tool_input = _as_str_object_dict(tu.get("input"))
                parts.append(
                    Part(
                        type="tool_call",
                        tool_call_id=tool_use_id if isinstance(tool_use_id, str) else None,
                        tool_name=tool_name if isinstance(tool_name, str) else None,
                        tool_args=tool_input,
                        provider_raw=block,
                    )
                )
            elif "reasoningContent" in block:
                rc = _as_str_object_dict(block["reasoningContent"]) or {}
                reasoning_text = _as_str_object_dict(rc.get("reasoningText")) or {}
                reasoning = reasoning_text.get("text")
                parts.append(
                    Part(
                        type="thinking",
                        text=reasoning if isinstance(reasoning, str) else None,
                        provider_raw=block,
                    )
                )

        if parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(parts),
                    provider="bedrock",
                )
            )

        usage_data = data.get("usage")
        if isinstance(usage_data, Mapping):
            input_tokens = _usage_int(usage_data, "inputTokens")
            output_tokens = _usage_int(usage_data, "outputTokens")
            total_tokens = _usage_int(usage_data, "totalTokens")
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
