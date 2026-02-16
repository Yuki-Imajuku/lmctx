"""OpenAI Responses API adapter.

Converts between lmctx Context/RunSpec and the OpenAI Responses API format.
Also works with Azure OpenAI Responses endpoints.

The Responses API is distinct from Chat Completions:
- Uses ``input`` (mixed item array) instead of ``messages``
- System prompt goes in ``instructions`` parameter
- Tool definitions use a flat structure (``name`` at top level)
- Tool results use ``function_call_output`` items (not ``role: "tool"``)
- Content types: ``input_text``, ``input_image``, ``output_text``
- Server-side state via ``previous_response_id``
- Structured output via ``text.format`` (not ``response_format``)
"""

from __future__ import annotations

import base64
import binascii
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
from lmctx.errors import BlobIntegrityError, BlobNotFoundError
from lmctx.plan import AdapterCapabilities, AdapterId, ExcludedItem, RequestPlan
from lmctx.types import Cursor, Message, Part, Usage

if TYPE_CHECKING:
    from lmctx.blobs import BlobReference, BlobStore
    from lmctx.context import Context
    from lmctx.spec import RunSpec


def _blob_to_data_url(store: BlobStore, blob: BlobReference) -> str:
    """Encode blob as a data URL for inline image input."""
    data = store.get_blob(blob)
    b64 = base64.b64encode(data).decode("ascii")
    media_type = blob.media_type or "application/octet-stream"
    return f"data:{media_type};base64,{b64}"


def _convert_tool_output(output: object) -> str:
    """Convert tool output to a JSON string for function_call_output."""
    if isinstance(output, str):
        return output
    return json.dumps(_to_json_compatible(output), ensure_ascii=False)


def _as_str_object_dict(value: object) -> dict[str, object] | None:
    """Convert an object to ``dict[str, object]`` when possible."""
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _default_file_name(part: Part) -> str:
    """Return a stable fallback filename for inline file payloads."""
    if part.blob is None or not isinstance(part.blob.media_type, str):
        return "upload.bin"
    suffix = part.blob.media_type.split("/", 1)[-1].lower()
    mapped_suffix = {"plain": "txt", "jpeg": "jpg"}.get(suffix, suffix)
    if mapped_suffix.isalnum():
        return f"upload.{mapped_suffix}"
    return "upload.bin"


def _file_input_from_part(part: Part, store: BlobStore) -> dict[str, object] | None:
    """Convert ``Part(type='file')`` to a Responses API ``input_file`` item."""
    raw = _as_str_object_dict(part.provider_raw)
    file_id = part.text if isinstance(part.text, str) and part.text else None
    if file_id is None and raw is not None:
        raw_file_id = raw.get("file_id")
        if isinstance(raw_file_id, str) and raw_file_id:
            file_id = raw_file_id
    if file_id is not None:
        # OpenAI Responses treats file_id and filename as mutually exclusive.
        return {"type": "input_file", "file_id": file_id}

    if raw is not None:
        file_data = raw.get("file_data")
        if isinstance(file_data, str) and file_data:
            item: dict[str, object] = {"type": "input_file", "file_data": file_data}
            filename = raw.get("filename")
            if isinstance(filename, str) and filename:
                item["filename"] = filename
            return item

        file_url = raw.get("file_url")
        if isinstance(file_url, str) and file_url:
            return {"type": "input_file", "file_url": file_url}

        filename = raw.get("filename")
        if isinstance(filename, str) and filename:
            return {"type": "input_file", "filename": filename}

    if part.blob is not None:
        data = store.get_blob(part.blob)
        encoded = base64.b64encode(data).decode("ascii")
        item = {"type": "input_file", "file_data": encoded}
        filename = None
        if raw is not None:
            raw_filename = raw.get("filename")
            if isinstance(raw_filename, str) and raw_filename:
                filename = raw_filename
        item["filename"] = filename or _default_file_name(part)
        return item

    return None


def _usage_int(usage_data: Mapping[str, object], key: str) -> int | None:
    """Read an integer usage field."""
    value = usage_data.get(key)
    return value if isinstance(value, int) else None


def _collect_reasoning_text(blocks: object, block_type: str) -> list[str]:
    """Collect text fields from typed reasoning blocks."""
    if not isinstance(blocks, list):
        return []

    texts: list[str] = []
    for block in blocks:
        block_dict = _as_str_object_dict(block)
        if block_dict is None:
            continue
        if block_dict.get("type") != block_type:
            continue
        text = block_dict.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


def _extract_reasoning_text(item: dict[str, object]) -> str | None:
    """Extract human-readable reasoning text from a reasoning output item."""
    texts = _collect_reasoning_text(item.get("content"), "reasoning_text")
    texts.extend(_collect_reasoning_text(item.get("summary"), "summary_text"))
    return "\n".join(texts) if texts else None


def _usage_payload(usage_data: Mapping[str, object]) -> dict[str, object]:
    """Copy full provider usage payload with stable string keys."""
    return _json_object(usage_data)


def _parse_compaction_item(ctx: Context, item: dict[str, object]) -> Part | None:
    """Convert a Responses API compaction item into a compact Part."""
    encrypted_content = item.get("encrypted_content")
    if not isinstance(encrypted_content, str):
        return None
    blob = ctx.blob_store.put_blob(
        encrypted_content.encode("utf-8"),
        media_type="application/octet-stream",
        kind="compaction",
    )
    return Part(type="compaction", blob=blob, provider_raw=item)


def _decode_base64_payload(raw_payload: object) -> bytes | None:
    """Decode a base64 payload, returning ``None`` when decoding fails."""
    if not isinstance(raw_payload, str) or not raw_payload:
        return None
    try:
        return base64.b64decode(raw_payload, validate=True)
    except (binascii.Error, ValueError):
        return None


def _part_from_image_payload(
    ctx: Context,
    payload: object,
    *,
    media_type: str | None,
    provider_raw: dict[str, object],
) -> Part | None:
    """Create an ``image`` part from a base64 payload."""
    decoded = _decode_base64_payload(payload)
    if decoded is None:
        return None
    blob = ctx.blob_store.put_blob(decoded, media_type=media_type, kind="image")
    return Part(type="image", blob=blob, provider_raw=provider_raw)


def _parse_message_content_block(ctx: Context, block: dict[str, object]) -> list[Part]:
    """Parse a message content block from Responses output."""
    block_type = block.get("type")
    if block_type == "output_text":
        text = block.get("text")
        if isinstance(text, str):
            return [Part(type="text", text=text)]
        return []

    if block_type == "output_image":
        media_type = block.get("media_type")
        image_part = _part_from_image_payload(
            ctx,
            block.get("b64_json") or block.get("image_base64") or block.get("result"),
            media_type=media_type if isinstance(media_type, str) else "image/png",
            provider_raw=block,
        )
        if image_part is not None:
            return [image_part]
        image_url = block.get("image_url")
        if isinstance(image_url, str) and image_url:
            return [Part(type="image", provider_raw=block)]
        image_url_obj = _as_str_object_dict(image_url)
        if image_url_obj is not None:
            url_value = image_url_obj.get("url")
            if isinstance(url_value, str) and url_value:
                return [Part(type="image", provider_raw=block)]
    return []


def _parse_message_item(ctx: Context, item: dict[str, object]) -> list[Part]:
    """Parse a ``type=message`` output item into lmctx parts."""
    parts: list[Part] = []
    content_blocks = item.get("content")
    if not isinstance(content_blocks, list):
        return parts

    for block in content_blocks:
        content_block = _as_str_object_dict(block)
        if content_block is None:
            continue
        parts.extend(_parse_message_content_block(ctx, content_block))
    return parts


def _parse_function_call_item(item: dict[str, object]) -> Part:
    """Parse a ``type=function_call`` output item into a tool call part."""
    tool_args = _normalize_tool_arguments(item.get("arguments"))

    call_id = item.get("call_id")
    tool_call_id = call_id if isinstance(call_id, str) else None
    name = item.get("name")
    tool_name = name if isinstance(name, str) else None

    return Part(
        type="tool_call",
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
        provider_raw=item,
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


def _parse_image_generation_call_item(ctx: Context, item: dict[str, object]) -> Part | None:
    """Parse ``type=image_generation_call`` output item into an image part."""
    media_type = item.get("media_type")
    image_part = _part_from_image_payload(
        ctx,
        item.get("result") or item.get("b64_json") or item.get("image_base64"),
        media_type=media_type if isinstance(media_type, str) else "image/png",
        provider_raw=item,
    )
    if image_part is not None:
        return image_part
    image_url = item.get("image_url")
    if isinstance(image_url, str) and image_url:
        return Part(type="image", provider_raw=item)
    return None


def _parse_output_item(ctx: Context, item: dict[str, object]) -> list[Part]:
    """Parse a single output item from Responses API into lmctx parts."""
    item_type = item.get("type")

    if item_type == "message":
        return _parse_message_item(ctx, item)
    if item_type == "function_call":
        return [_parse_function_call_item(item)]
    if item_type == "image_generation_call":
        image_part = _parse_image_generation_call_item(ctx, item)
        return [image_part] if image_part is not None else []
    if item_type == "reasoning":
        return [Part(type="thinking", text=_extract_reasoning_text(item), provider_raw=item)]
    if item_type == "compaction":
        compaction_part = _parse_compaction_item(ctx, item)
        return [compaction_part] if compaction_part is not None else []
    return []


def _user_to_responses(msg: Message, store: BlobStore) -> dict[str, object]:
    """Convert a user message to Responses API input format."""
    content: list[dict[str, object]] = []
    for part in msg.parts:
        if part.type == "text" and part.text:
            content.append({"type": "input_text", "text": part.text})
        elif part.type == "image" and part.blob:
            url = _blob_to_data_url(store, part.blob)
            content.append({"type": "input_image", "image_url": url})
        elif part.type == "file":
            file_item = _file_input_from_part(part, store)
            if file_item is not None:
                content.append(file_item)

    # Simplify single text-only messages to a plain string.
    if len(content) == 1 and content[0]["type"] == "input_text":
        text_value = content[0].get("text")
        return {"role": "user", "content": text_value if isinstance(text_value, str) else ""}
    return {"role": "user", "content": content}


def _assistant_to_responses(msg: Message, store: BlobStore) -> list[dict[str, object]]:
    """Convert an assistant message to Responses API input items.

    Text parts become a message item; tool calls become function_call items.
    """
    items: list[dict[str, object]] = []
    text_parts = [p for p in msg.parts if p.type == "text" and p.text]
    image_parts = [p for p in msg.parts if p.type == "image"]
    tool_call_parts = [p for p in msg.parts if p.type == "tool_call" and p.tool_call_id and p.tool_name]

    content: list[dict[str, object]] = [{"type": "output_text", "text": p.text} for p in text_parts if p.text]
    for image_part in image_parts:
        raw = _as_str_object_dict(image_part.provider_raw)
        if raw is None:
            continue
        image_type = raw.get("type")
        if image_type == "output_image":
            content.append(raw)

    if content:
        items.append({"type": "message", "role": "assistant", "content": content})

    items.extend(
        {
            "type": "function_call",
            "call_id": p.tool_call_id,
            "name": p.tool_name,
            "arguments": json.dumps(_to_json_compatible(p.tool_args or {}), ensure_ascii=False),
        }
        for p in tool_call_parts
    )

    for image_part in image_parts:
        raw = _as_str_object_dict(image_part.provider_raw)
        if raw is None:
            continue
        if raw.get("type") != "image_generation_call":
            continue
        image_call_id = raw.get("id")
        if isinstance(image_call_id, str) and image_call_id:
            items.append({"type": "image_generation_call", "id": image_call_id})

    for part in msg.parts:
        if part.type != "compaction":
            continue

        item: dict[str, object] | None = None

        if part.provider_raw and part.provider_raw.get("type") == "compaction":
            encrypted_content = part.provider_raw.get("encrypted_content")
            if isinstance(encrypted_content, str):
                item = {"type": "compaction", "encrypted_content": encrypted_content}
                compaction_id = part.provider_raw.get("id")
                if isinstance(compaction_id, str):
                    item["id"] = compaction_id
        elif part.blob:
            try:
                encrypted_content = store.get_blob(part.blob).decode("utf-8")
            except (BlobIntegrityError, BlobNotFoundError, UnicodeDecodeError):
                encrypted_content = None
            if encrypted_content is not None:
                item = {"type": "compaction", "encrypted_content": encrypted_content}
        elif part.text:
            item = {"type": "compaction", "encrypted_content": part.text}

        if item is not None:
            items.append(item)

    return items


def _tool_results_to_responses(msg: Message) -> list[dict[str, object]]:
    """Convert tool result parts to function_call_output items."""
    return [
        {
            "type": "function_call_output",
            "call_id": p.tool_call_id,
            "output": _convert_tool_output(p.tool_output),
        }
        for p in msg.parts
        if p.type == "tool_result" and p.tool_call_id
    ]


def _message_to_responses(msg: Message, store: BlobStore) -> list[dict[str, object]]:
    """Convert a single lmctx Message to Responses API input items."""
    if msg.role == "user":
        return [_user_to_responses(msg, store)]
    if msg.role == "assistant":
        return _assistant_to_responses(msg, store)
    if msg.role == "tool":
        return _tool_results_to_responses(msg)
    # system/developer messages from Context become input items.
    if msg.role in ("system", "developer"):
        return [{"role": msg.role, "content": p.text} for p in msg.parts if p.type == "text" and p.text]
    return []


def _validate_input_item(item: dict[str, object]) -> str | None:
    """Validate a converted Responses API input item."""
    item_type = item.get("type")
    if isinstance(item_type, str):
        if item_type == "message":
            content = item.get("content")
            if isinstance(content, list) and content:
                return None
            return "message item has empty content blocks"
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                return "function_call item is missing call_id"
            name = item.get("name")
            if not isinstance(name, str) or not name:
                return "function_call item is missing name"
            return None
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                return "function_call_output item is missing call_id"
            output = item.get("output")
            if not isinstance(output, str):
                return "function_call_output item output must be a string"
            return None
        if item_type == "image_generation_call":
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                return "image_generation_call item is missing id"
            return None
        if item_type == "compaction":
            encrypted_content = item.get("encrypted_content")
            if not isinstance(encrypted_content, str) or not encrypted_content:
                return "compaction item is missing encrypted_content"
            return None
        return f"unsupported input item type: {item_type!r}"

    role = item.get("role")
    if isinstance(role, str):
        content = item.get("content")
        if role == "user":
            if isinstance(content, str) and content:
                return None
            if isinstance(content, list) and content:
                return None
            return "user input item has no serializable content"
        if role in {"system", "developer"}:
            if isinstance(content, str) and content:
                return None
            return f"{role} input item must be non-empty text"
        return f"unsupported role input item: {role!r}"

    return "input item is missing role or type"


def _part_exclusion_reason_for_role(part: Part, role: str, store: BlobStore) -> str | None:
    """Return a reason when a part cannot be serialized for the given role."""
    if role == "user":
        if part.type == "text":
            return None if part.text else "user text part is empty"
        if part.type == "image":
            return None if part.blob is not None else "user image part requires blob"
        if part.type == "file":
            if part.blob is not None:
                return None
            raw = _as_str_object_dict(part.provider_raw)
            if isinstance(part.text, str) and part.text:
                return None
            if raw is not None:
                if isinstance(raw.get("file_id"), str) and raw.get("file_id"):
                    return None
                if isinstance(raw.get("file_data"), str) and raw.get("file_data"):
                    return None
                if isinstance(raw.get("file_url"), str) and raw.get("file_url"):
                    return None
                if isinstance(raw.get("filename"), str) and raw.get("filename"):
                    return None
            return "user file part requires blob or file_id/file_data/file_url/filename metadata"
        return f"user role does not support part type '{part.type}' for responses.create"

    if role == "assistant":
        if part.type == "text":
            return None if part.text else "assistant text part is empty"
        if part.type == "tool_call":
            if part.tool_call_id and part.tool_name:
                return None
            return "assistant tool_call part requires both tool_call_id and tool_name"
        if part.type == "image":
            raw = _as_str_object_dict(part.provider_raw)
            if raw is not None:
                raw_type = raw.get("type")
                if raw_type in {"output_image", "image_generation_call"}:
                    return None
            return "assistant image part requires provider_raw from output_image/image_generation_call"
        if part.type == "compaction":
            raw = _as_str_object_dict(part.provider_raw)
            if raw is not None and raw.get("type") == "compaction" and isinstance(raw.get("encrypted_content"), str):
                return None
            if part.blob is not None:
                try:
                    store.get_blob(part.blob).decode("utf-8")
                except BlobNotFoundError:
                    return "assistant compaction blob is missing in blob_store"
                except BlobIntegrityError:
                    return "assistant compaction blob failed integrity validation"
                except UnicodeDecodeError:
                    return "assistant compaction blob must decode as UTF-8 text"
                return None
            if part.text:
                return None
            return "assistant compaction part requires encrypted_content, blob, or text"
        return f"assistant role does not support part type '{part.type}' for responses.create input"

    if role == "tool":
        if part.type == "tool_result":
            return None if part.tool_call_id else "tool_result part is missing tool_call_id"
        return f"tool role only supports tool_result parts for responses.create input (got '{part.type}')"

    if role in {"system", "developer"}:
        if part.type == "text":
            return None if part.text else f"{role} text part is empty"
        return f"{role} role only supports text parts for responses.create input (got '{part.type}')"

    return f"role '{role}' is not supported by responses.create input conversion"


def _build_input_items(ctx: Context) -> tuple[list[dict[str, object]], tuple[ExcludedItem, ...], tuple[str, ...]]:
    """Build validated Responses input items with diagnostics."""
    input_items: list[dict[str, object]] = []
    excluded: list[ExcludedItem] = []
    errors: list[str] = []

    for message_index, msg in enumerate(ctx.messages):
        for part_index, part in enumerate(msg.parts):
            part_error = _part_exclusion_reason_for_role(part, msg.role, ctx.blob_store)
            if part_error is not None:
                excluded.append(
                    ExcludedItem(
                        description=f"context.messages[{message_index}].parts[{part_index}]",
                        reason=part_error,
                    )
                )

        converted_items = _message_to_responses(msg, ctx.blob_store)
        if not converted_items:
            excluded.append(
                ExcludedItem(
                    description=f"context.messages[{message_index}]",
                    reason=f"{msg.role} message has no Responses-compatible parts",
                )
            )
            errors.append(f"context.messages[{message_index}] produced no Responses input items.")
            continue

        valid_count = 0
        for item_index, item in enumerate(converted_items):
            validation_error = _validate_input_item(item)
            if validation_error is not None:
                description = f"context.messages[{message_index}]"
                if len(converted_items) > 1:
                    description = f"{description}[{item_index}]"
                excluded.append(ExcludedItem(description=description, reason=validation_error))
                continue

            input_items.append(item)
            valid_count += 1

        if valid_count == 0:
            errors.append(f"context.messages[{message_index}] produced no valid Responses input items.")

    return input_items, tuple(excluded), tuple(errors)


def _build_instructions(spec: RunSpec) -> str | None:
    """Build the instructions parameter from RunSpec."""
    parts: list[str] = []
    if spec.instructions:
        if spec.instructions.system:
            parts.append(spec.instructions.system)
        if spec.instructions.developer:
            parts.append(spec.instructions.developer)
    return "\n\n".join(parts) if parts else None


def _apply_generation_params(request: dict[str, object], spec: RunSpec) -> None:
    """Apply optional generation parameters from RunSpec to the request dict."""
    if spec.max_output_tokens is not None:
        request["max_output_tokens"] = spec.max_output_tokens
    if spec.temperature is not None:
        request["temperature"] = spec.temperature
    if spec.top_p is not None:
        request["top_p"] = spec.top_p
    if spec.response_modalities:
        request["modalities"] = list(spec.response_modalities)


def _apply_tools(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply tool definitions from RunSpec (flat format for Responses API)."""
    if spec.tools:
        request["tools"] = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in spec.tools
        ]
        included.append(f"{len(spec.tools)} tools")
    if spec.tool_choice is not None:
        request["tool_choice"] = spec.tool_choice


def _ensure_response_format_name(format_obj: dict[str, object]) -> dict[str, object]:
    """Ensure ``text.format`` has a valid ``name`` field."""
    if not isinstance(format_obj.get("name"), str) or not str(format_obj["name"]):
        format_obj["name"] = "response"
    return format_obj


def _ensure_closed_object_schemas(value: object) -> object:
    """Recursively set ``additionalProperties: false`` for object schemas when missing.

    OpenAI structured outputs with ``strict=true`` require object schemas to
    explicitly declare ``additionalProperties``.
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


def _apply_strict_schema_constraints(format_obj: dict[str, object]) -> dict[str, object]:
    """Apply strict-mode schema normalization required by OpenAI."""
    if format_obj.get("strict") is not True:
        return format_obj

    schema = format_obj.get("schema")
    if not isinstance(schema, Mapping):
        return format_obj

    normalized = dict(format_obj)
    normalized_schema = {str(key): value for key, value in schema.items()}
    normalized["schema"] = _ensure_closed_object_schemas(_to_json_compatible(normalized_schema))
    return normalized


def _response_format_from_explicit_format(schema_obj: dict[str, object]) -> dict[str, object] | None:
    """Parse ``{"type":"json_schema", ...}`` format-shaped payloads."""
    if schema_obj.get("type") != "json_schema":
        return None

    format_obj: dict[str, object] = dict(schema_obj)
    legacy_nested = _as_str_object_dict(format_obj.pop("json_schema", None))
    if legacy_nested is not None:
        for key in ("name", "schema", "description", "strict"):
            if key not in format_obj and key in legacy_nested:
                format_obj[key] = legacy_nested[key]

    if "schema" not in format_obj:
        msg = "response_schema with type='json_schema' must include a schema."
        raise ValueError(msg)
    return _ensure_response_format_name(format_obj)


def _response_format_from_schema_wrapper(schema_obj: dict[str, object]) -> dict[str, object] | None:
    """Parse OpenAI-style wrapper payloads like ``{name, schema, strict}``."""
    wrapped_schema = _as_str_object_dict(schema_obj.get("schema"))
    if wrapped_schema is None:
        return None

    format_obj: dict[str, object] = {
        "type": "json_schema",
        "schema": _json_object(wrapped_schema),
        "name": schema_obj.get("name") if isinstance(schema_obj.get("name"), str) else "response",
    }
    description = schema_obj.get("description")
    if isinstance(description, str) and description:
        format_obj["description"] = description
    strict = schema_obj.get("strict")
    if isinstance(strict, bool):
        format_obj["strict"] = strict
    return format_obj


def _response_format_from_legacy_nested(schema_obj: dict[str, object]) -> dict[str, object] | None:
    """Parse legacy nested payloads like ``{json_schema: {...}}``."""
    legacy_nested = _as_str_object_dict(schema_obj.get("json_schema"))
    if legacy_nested is None:
        return None

    format_obj: dict[str, object] = {"type": "json_schema"}
    for key in ("name", "schema", "description", "strict"):
        value = legacy_nested.get(key)
        if value is not None:
            format_obj[key] = value
    if "schema" not in format_obj:
        msg = "response_schema.json_schema must include a schema."
        raise ValueError(msg)
    return _ensure_response_format_name(format_obj)


def _normalized_text_format(response_schema: Mapping[str, object]) -> dict[str, object]:
    """Normalize ``RunSpec.response_schema`` into Responses API ``text.format`` payload."""
    schema_obj = _json_object(response_schema)

    explicit_format = _response_format_from_explicit_format(schema_obj)
    if explicit_format is not None:
        return _apply_strict_schema_constraints(explicit_format)

    wrapped_schema = _response_format_from_schema_wrapper(schema_obj)
    if wrapped_schema is not None:
        return _apply_strict_schema_constraints(wrapped_schema)

    legacy_nested = _response_format_from_legacy_nested(schema_obj)
    if legacy_nested is not None:
        return _apply_strict_schema_constraints(legacy_nested)

    return {
        "type": "json_schema",
        "name": "response",
        "schema": schema_obj,
    }


def _apply_schema_and_cursor(request: dict[str, object], spec: RunSpec, ctx: Context, included: list[str]) -> None:
    """Apply structured output and cursor settings to the request dict."""
    if spec.response_schema is not None:
        request["text"] = {"format": _normalized_text_format(spec.response_schema)}
        included.append("response_schema")

    _apply_cursor(request, ctx, included)


def _apply_cursor(request: dict[str, object], ctx: Context, included: list[str]) -> None:
    """Apply ``previous_response_id`` from the Context cursor."""
    if ctx.cursor and ctx.cursor.last_response_id:
        request["previous_response_id"] = ctx.cursor.last_response_id
        included.append("previous_response_id")


def _apply_transport_overrides(request: dict[str, object], spec: RunSpec, included: list[str]) -> None:
    """Apply optional transport-level overrides accepted by OpenAI SDK."""
    if spec.extra_headers:
        request["extra_headers"] = dict(spec.extra_headers)
        included.append("extra_headers")
    if spec.extra_query:
        request["extra_query"] = dict(spec.extra_query)
        included.append("extra_query")


_RESPONSES_CAPABILITIES = AdapterCapabilities(
    id=AdapterId(provider="openai", endpoint="responses.create"),
    fields={
        "instructions": "yes",
        "max_output_tokens": "yes",
        "temperature": "yes",
        "top_p": "yes",
        "seed": "no",
        "tools": "yes",
        "tool_choice": "yes",
        "response_schema": "yes",
        "response_modalities": "yes",
        "extra_body": "yes",
        "extra_headers": "yes",
        "extra_query": "yes",
        "cursor_chaining": "yes",
    },
    notes={
        "seed": "responses.create does not support deterministic sampling seed.",
        "cursor_chaining": "Uses Context.cursor.last_response_id as previous_response_id.",
    },
)

_RESPONSES_COMPACT_CAPABILITIES = AdapterCapabilities(
    id=AdapterId(provider="openai", endpoint="responses.compact"),
    fields={
        "instructions": "yes",
        "max_output_tokens": "no",
        "temperature": "no",
        "top_p": "no",
        "seed": "no",
        "tools": "no",
        "tool_choice": "no",
        "response_schema": "no",
        "response_modalities": "no",
        "extra_body": "yes",
        "extra_headers": "yes",
        "extra_query": "yes",
        "cursor_chaining": "yes",
    },
    notes={
        "cursor_chaining": "Uses Context.cursor.last_response_id as previous_response_id.",
    },
)


class OpenAIResponsesAdapter:
    """Adapter for the OpenAI Responses API.

    Usage::

        from openai import OpenAI
        client = OpenAI()
        plan = adapter.plan(ctx, spec)
        response = client.responses.create(**plan.request)
        ctx = adapter.ingest(ctx, response, spec=spec)
    """

    id = AdapterId(provider="openai", endpoint="responses.create")

    def capabilities(self) -> AdapterCapabilities:
        """Return capability metadata for this adapter."""
        return _RESPONSES_CAPABILITIES

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build an OpenAI Responses API request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        included: list[str] = []
        excluded: list[ExcludedItem] = []
        errors: list[str] = []

        instructions = _build_instructions(spec)
        if spec.instructions:
            if spec.instructions.system:
                included.append("system instruction")
            if spec.instructions.developer:
                included.append("developer instruction")

        input_items, conversion_excluded, conversion_errors = _build_input_items(ctx)
        excluded.extend(conversion_excluded)
        errors.extend(conversion_errors)
        included.append(f"{len(ctx)} messages")

        request: dict[str, object] = {"model": spec.model, "input": input_items}

        if instructions:
            request["instructions"] = instructions

        _apply_generation_params(request, spec)
        if spec.response_modalities:
            included.append("response_modalities")
        _apply_tools(request, spec, included)
        _apply_schema_and_cursor(request, spec, ctx, included)
        _apply_transport_overrides(request, spec, included)
        if spec.seed is not None:
            excluded.append(
                ExcludedItem(
                    description="seed",
                    reason="responses.create does not support deterministic sampling seed",
                )
            )

        if spec.extra_body:
            request = _deep_merge(request, spec.extra_body)

        if not input_items and instructions is None:
            errors.append("responses.create request has no input items or instructions after conversion.")

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            errors=tuple(errors),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
        """Parse an OpenAI Responses API response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        output_items = data.get("output", [])
        parts: list[Part] = []

        for output_item in output_items:
            item = _as_str_object_dict(output_item)
            if item is None:
                continue
            parts.extend(_parse_output_item(ctx, item))

        if parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(parts),
                    id=data.get("id"),
                    provider="openai",
                )
            )

        # Store response ID in cursor for conversation chaining.
        response_id = data.get("id")
        if response_id:
            ctx = ctx.with_cursor(
                Cursor(
                    last_response_id=response_id,
                    conversation_id=ctx.cursor.conversation_id,
                    session_id=ctx.cursor.session_id,
                )
            )

        # Extract usage.
        usage_data = data.get("usage")
        if isinstance(usage_data, Mapping):
            ctx = ctx.with_usage(
                Usage(
                    input_tokens=_usage_int(usage_data, "input_tokens"),
                    output_tokens=_usage_int(usage_data, "output_tokens"),
                    total_tokens=_usage_int(usage_data, "total_tokens"),
                    provider_usage=_usage_payload(usage_data),
                )
            )

        return ctx


def _compact_excluded_items(spec: RunSpec) -> tuple[ExcludedItem, ...]:
    """List RunSpec fields ignored by ``responses.compact``."""
    excluded: list[ExcludedItem] = []

    if spec.max_output_tokens is not None:
        excluded.append(
            ExcludedItem(
                description="max_output_tokens",
                reason="responses.compact ignores generation limits",
            )
        )
    if spec.temperature is not None:
        excluded.append(
            ExcludedItem(
                description="temperature",
                reason="responses.compact ignores sampling parameters",
            )
        )
    if spec.top_p is not None:
        excluded.append(
            ExcludedItem(
                description="top_p",
                reason="responses.compact ignores sampling parameters",
            )
        )
    if spec.seed is not None:
        excluded.append(
            ExcludedItem(
                description="seed",
                reason="responses.compact ignores deterministic sampling",
            )
        )
    if spec.tools:
        excluded.append(
            ExcludedItem(
                description="tools",
                reason="responses.compact does not execute tools",
            )
        )
    if spec.tool_choice is not None:
        excluded.append(
            ExcludedItem(
                description="tool_choice",
                reason="responses.compact does not execute tools",
            )
        )
    if spec.response_schema is not None:
        excluded.append(
            ExcludedItem(
                description="response_schema",
                reason="responses.compact does not support text formatting",
            )
        )
    if spec.response_modalities:
        excluded.append(
            ExcludedItem(
                description="response_modalities",
                reason="responses.compact does not generate modal output",
            )
        )

    return tuple(excluded)


class OpenAIResponsesCompactAdapter:
    """Adapter for the OpenAI Responses Compact API (``responses.compact``)."""

    id = AdapterId(provider="openai", endpoint="responses.compact")

    def capabilities(self) -> AdapterCapabilities:
        """Return capability metadata for this adapter."""
        return _RESPONSES_COMPACT_CAPABILITIES

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build an OpenAI Responses Compact request from Context and RunSpec."""
        _validate_adapter_spec(self.id, spec)
        included: list[str] = []
        excluded: list[ExcludedItem] = []
        errors: list[str] = []

        instructions = _build_instructions(spec)
        if spec.instructions:
            if spec.instructions.system:
                included.append("system instruction")
            if spec.instructions.developer:
                included.append("developer instruction")

        input_items, conversion_excluded, conversion_errors = _build_input_items(ctx)
        excluded.extend(conversion_excluded)
        errors.extend(conversion_errors)
        included.append(f"{len(ctx)} messages")

        request: dict[str, object] = {"model": spec.model, "input": input_items}
        if instructions:
            request["instructions"] = instructions

        _apply_cursor(request, ctx, included)
        _apply_transport_overrides(request, spec, included)

        if spec.extra_body:
            request = _deep_merge(request, spec.extra_body)

        if not input_items and instructions is None:
            errors.append("responses.compact request has no input items or instructions after conversion.")

        excluded.extend(_compact_excluded_items(spec))

        return RequestPlan(
            request=request,
            included=tuple(included),
            excluded=tuple(excluded),
            errors=tuple(errors),
            extra=_plan_extra_hints(spec),
        )

    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
        """Parse an OpenAI Responses Compact response into the Context."""
        _validate_adapter_spec(self.id, spec)
        data = _to_dict(response)

        output_items = data.get("output", [])
        parts: list[Part] = []
        for output_item in output_items:
            item = _as_str_object_dict(output_item)
            if item is None:
                continue
            if item.get("type") != "compaction":
                continue
            compaction_part = _parse_compaction_item(ctx, item)
            if compaction_part is not None:
                parts.append(compaction_part)

        if parts:
            ctx = ctx.append(
                Message(
                    role="assistant",
                    parts=tuple(parts),
                    id=data.get("id"),
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
                    provider_usage=_usage_payload(usage_data),
                )
            )

        return ctx
