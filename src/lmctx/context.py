"""Context: the append-only conversation log."""

from __future__ import annotations

import base64
import binascii
import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from lmctx.blobs import BlobReference, BlobStore, InMemoryBlobStore
from lmctx.errors import BlobIntegrityError, BlobNotFoundError, ContextError
from lmctx.serde import as_str_object_dict, optional_int, optional_string, require_int, to_plain_data
from lmctx.types import Cursor, Message, Part, Role, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

PipeResultT = TypeVar("PipeResultT")
_ROLE_VALUES = frozenset({"system", "developer", "user", "assistant", "tool"})


def _normalize_content(content: str | Part | Sequence[Part]) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (Part(type="text", text=content),)
    if isinstance(content, Part):
        return (content,)

    parts: list[Part] = []
    for index, item in enumerate(content):
        if not isinstance(item, Part):
            msg = f"content sequence items must be Part instances; got {type(item).__name__} at index {index}."
            raise TypeError(msg)
        parts.append(item)
    return tuple(parts)


class _SerializedBlobStore:
    """BlobStore used for Context.from_dict blob payload hydration."""

    def __init__(self, blobs_by_id: Mapping[str, bytes]) -> None:
        """Initialize with preloaded blob bytes keyed by BlobReference.id."""
        self._blobs = {str(blob_id): data for blob_id, data in blobs_by_id.items()}

    def put(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        kind: str = "file",
    ) -> BlobReference:
        """Store bytes and return a BlobReference."""
        digest = hashlib.sha256(data).hexdigest()
        blob_id = uuid.uuid4().hex
        self._blobs[blob_id] = data
        return BlobReference(
            id=blob_id,
            sha256=digest,
            media_type=media_type,
            kind=kind,
            size=len(data),
        )

    def get(self, ref: BlobReference) -> bytes:
        """Retrieve bytes and verify SHA-256 integrity."""
        data = self._blobs.get(ref.id)
        if data is None:
            raise BlobNotFoundError(ref.id)
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            raise BlobIntegrityError(ref.id, ref.sha256, actual)
        return data

    def contains(self, ref: BlobReference) -> bool:
        """Check whether a blob exists."""
        return ref.id in self._blobs


def _blob_ref_to_dict(ref: BlobReference) -> dict[str, object]:
    """Serialize BlobReference to a plain dictionary."""
    return {
        "id": ref.id,
        "sha256": ref.sha256,
        "media_type": ref.media_type,
        "kind": ref.kind,
        "size": ref.size,
    }


def _blob_ref_from_dict(value: object, *, field_name: str) -> BlobReference:
    """Deserialize BlobReference from a plain dictionary."""
    data = as_str_object_dict(value, field_name=field_name)

    blob_id = data.get("id")
    if not isinstance(blob_id, str) or not blob_id:
        msg = f"{field_name}.id must be a non-empty string."
        raise TypeError(msg)

    sha256 = data.get("sha256")
    if not isinstance(sha256, str) or not sha256:
        msg = f"{field_name}.sha256 must be a non-empty string."
        raise TypeError(msg)

    media_type = optional_string(data.get("media_type"), field_name=f"{field_name}.media_type")

    kind = data.get("kind")
    if not isinstance(kind, str) or not kind:
        msg = f"{field_name}.kind must be a non-empty string."
        raise TypeError(msg)

    size = require_int(data.get("size"), field_name=f"{field_name}.size")

    return BlobReference(
        id=blob_id,
        sha256=sha256,
        media_type=media_type,
        kind=kind,
        size=size,
    )


def _part_to_dict(part: Part) -> dict[str, object]:
    """Serialize Part to a plain dictionary."""
    payload: dict[str, object] = {"type": part.type}
    if part.text is not None:
        payload["text"] = part.text
    if part.json is not None:
        payload["json"] = to_plain_data(part.json)
    if part.blob is not None:
        payload["blob"] = _blob_ref_to_dict(part.blob)
    if part.tool_call_id is not None:
        payload["tool_call_id"] = part.tool_call_id
    if part.tool_name is not None:
        payload["tool_name"] = part.tool_name
    if part.tool_args is not None:
        payload["tool_args"] = to_plain_data(part.tool_args)
    if part.tool_output is not None:
        payload["tool_output"] = to_plain_data(part.tool_output)
    if part.provider_raw is not None:
        payload["provider_raw"] = to_plain_data(part.provider_raw)
    return payload


def _part_from_dict(value: object, *, field_name: str) -> Part:
    """Deserialize Part from a plain dictionary."""
    data = as_str_object_dict(value, field_name=field_name)

    part_type = data.get("type")
    if not isinstance(part_type, str) or not part_type:
        msg = f"{field_name}.type must be a non-empty string."
        raise TypeError(msg)

    text = optional_string(data.get("text"), field_name=f"{field_name}.text")

    json_value = data.get("json")
    json_payload = (
        {str(key): to_plain_data(item) for key, item in json_value.items()} if isinstance(json_value, Mapping) else None
    )
    if json_value is not None and json_payload is None:
        msg = f"{field_name}.json must be a mapping or None."
        raise TypeError(msg)

    blob_value = data.get("blob")
    blob = _blob_ref_from_dict(blob_value, field_name=f"{field_name}.blob") if blob_value is not None else None

    tool_call_id = optional_string(data.get("tool_call_id"), field_name=f"{field_name}.tool_call_id")
    tool_name = optional_string(data.get("tool_name"), field_name=f"{field_name}.tool_name")

    tool_args_value = data.get("tool_args")
    tool_args = (
        {str(key): to_plain_data(item) for key, item in tool_args_value.items()}
        if isinstance(tool_args_value, Mapping)
        else None
    )
    if tool_args_value is not None and tool_args is None:
        msg = f"{field_name}.tool_args must be a mapping or None."
        raise TypeError(msg)

    provider_raw_value = data.get("provider_raw")
    provider_raw = (
        {str(key): to_plain_data(item) for key, item in provider_raw_value.items()}
        if isinstance(provider_raw_value, Mapping)
        else None
    )
    if provider_raw_value is not None and provider_raw is None:
        msg = f"{field_name}.provider_raw must be a mapping or None."
        raise TypeError(msg)

    return Part(
        type=part_type,
        text=text,
        json=json_payload,
        blob=blob,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_output=to_plain_data(data.get("tool_output")),
        provider_raw=provider_raw,
    )


def _message_to_dict(message: Message) -> dict[str, object]:
    """Serialize Message to a plain dictionary."""
    return {
        "role": message.role,
        "parts": [_part_to_dict(part) for part in message.parts],
        "id": message.id,
        "provider": message.provider,
        "turn_id": message.turn_id,
        "timestamp": message.timestamp.isoformat(),
    }


def _message_from_dict(value: object, *, field_name: str) -> Message:
    """Deserialize Message from a plain dictionary."""
    data = as_str_object_dict(value, field_name=field_name)

    role_value = data.get("role")
    if not isinstance(role_value, str) or role_value not in _ROLE_VALUES:
        msg = f"{field_name}.role must be one of {sorted(_ROLE_VALUES)!r}."
        raise TypeError(msg)
    role = role_value

    parts_value = data.get("parts")
    if not isinstance(parts_value, list | tuple):
        msg = f"{field_name}.parts must be a sequence."
        raise TypeError(msg)
    parts = tuple(
        _part_from_dict(part_value, field_name=f"{field_name}.parts[{index}]")
        for index, part_value in enumerate(parts_value)
    )

    message_id = optional_string(data.get("id"), field_name=f"{field_name}.id")
    provider = optional_string(data.get("provider"), field_name=f"{field_name}.provider")
    turn_id = optional_string(data.get("turn_id"), field_name=f"{field_name}.turn_id")

    timestamp_value = data.get("timestamp")
    if timestamp_value is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(timestamp_value, str):
        try:
            timestamp = datetime.fromisoformat(timestamp_value)
        except ValueError as exc:
            msg = f"{field_name}.timestamp must be an ISO-8601 datetime string."
            raise ValueError(msg) from exc
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        msg = f"{field_name}.timestamp must be a string or None."
        raise TypeError(msg)

    return Message(
        role=role,  # type: ignore[arg-type]
        parts=parts,
        id=message_id,
        provider=provider,
        turn_id=turn_id,
        timestamp=timestamp,
    )


def _cursor_to_dict(cursor: Cursor) -> dict[str, object]:
    """Serialize Cursor to a plain dictionary."""
    return {
        "last_response_id": cursor.last_response_id,
        "conversation_id": cursor.conversation_id,
        "session_id": cursor.session_id,
    }


def _cursor_from_dict(value: object) -> Cursor:
    """Deserialize Cursor from a plain dictionary."""
    if value is None:
        return Cursor()
    data = as_str_object_dict(value, field_name="Context.cursor")
    return Cursor(
        last_response_id=optional_string(data.get("last_response_id"), field_name="Context.cursor.last_response_id"),
        conversation_id=optional_string(data.get("conversation_id"), field_name="Context.cursor.conversation_id"),
        session_id=optional_string(data.get("session_id"), field_name="Context.cursor.session_id"),
    )


def _usage_to_dict(usage: Usage) -> dict[str, object]:
    """Serialize Usage to a plain dictionary."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "provider_usage": to_plain_data(usage.provider_usage),
    }


def _usage_from_dict(value: object, *, field_name: str) -> Usage:
    """Deserialize Usage from a plain dictionary."""
    data = as_str_object_dict(value, field_name=field_name)

    provider_usage_value = data.get("provider_usage")
    provider_usage = (
        {str(key): to_plain_data(item) for key, item in provider_usage_value.items()}
        if isinstance(provider_usage_value, Mapping)
        else {}
    )
    if provider_usage_value is not None and not isinstance(provider_usage_value, Mapping):
        msg = f"{field_name}.provider_usage must be a mapping or None."
        raise TypeError(msg)

    return Usage(
        input_tokens=optional_int(data.get("input_tokens"), field_name=f"{field_name}.input_tokens"),
        output_tokens=optional_int(data.get("output_tokens"), field_name=f"{field_name}.output_tokens"),
        total_tokens=optional_int(data.get("total_tokens"), field_name=f"{field_name}.total_tokens"),
        provider_usage=provider_usage,
    )


def _collect_blob_references(messages: tuple[Message, ...]) -> tuple[BlobReference, ...]:
    """Collect unique BlobReferences from messages."""
    seen_ids: set[str] = set()
    refs: list[BlobReference] = []
    for message in messages:
        for part in message.parts:
            if part.blob is None or part.blob.id in seen_ids:
                continue
            seen_ids.add(part.blob.id)
            refs.append(part.blob)
    return tuple(refs)


def _serialize_blob_payloads(ctx: Context) -> list[dict[str, object]]:
    """Serialize referenced blob bytes for portable Context round-trips."""
    payloads: list[dict[str, object]] = []
    for ref in _collect_blob_references(ctx.messages):
        try:
            data = ctx.blob_store.get(ref)
        except (BlobNotFoundError, BlobIntegrityError) as exc:
            msg = f"Cannot serialize blob payload for ref '{ref.id}': {exc}"
            raise ContextError(msg) from exc
        payloads.append(
            {
                "ref": _blob_ref_to_dict(ref),
                "data_b64": base64.b64encode(data).decode("ascii"),
            }
        )
    return payloads


def _deserialize_blob_store(value: object) -> BlobStore:
    """Deserialize a blob payload list into a hydrated BlobStore."""
    if not isinstance(value, list | tuple):
        msg = "Context.blob_payloads must be a sequence."
        raise TypeError(msg)

    blobs_by_id: dict[str, bytes] = {}
    for index, payload_value in enumerate(value):
        payload = as_str_object_dict(payload_value, field_name=f"Context.blob_payloads[{index}]")
        ref = _blob_ref_from_dict(payload.get("ref"), field_name=f"Context.blob_payloads[{index}].ref")

        data_b64 = payload.get("data_b64")
        if not isinstance(data_b64, str):
            msg = f"Context.blob_payloads[{index}].data_b64 must be a base64 string."
            raise TypeError(msg)
        try:
            blob_bytes = base64.b64decode(data_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            msg = f"Context.blob_payloads[{index}].data_b64 must be valid base64."
            raise ValueError(msg) from exc

        actual = hashlib.sha256(blob_bytes).hexdigest()
        if actual != ref.sha256:
            msg = (
                f"Context.blob_payloads[{index}] sha256 mismatch for blob {ref.id!r}: "
                f"expected {ref.sha256}, got {actual}"
            )
            raise ValueError(msg)
        blobs_by_id[ref.id] = blob_bytes

    return _SerializedBlobStore(blobs_by_id)


@dataclass(frozen=True, slots=True)
class Context:
    """Append-only conversation log.

    By default, every mutation method returns a new Context instance.
    Set ``inplace=True`` to mutate the current snapshot and return ``None``.
    The blob_store is intentionally shared across snapshots so that
    all Context instances from the same chain can resolve the same blobs.
    """

    messages: tuple[Message, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)
    usage_log: tuple[Usage, ...] = ()
    blob_store: BlobStore = field(default_factory=InMemoryBlobStore)
    __hash__ = None

    def __post_init__(self) -> None:
        """Normalize containers so runtime behavior matches type hints."""
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "usage_log", tuple(self.usage_log))

    # --- Internal state transition helper ---

    def _next(
        self,
        *,
        messages: tuple[Message, ...] | None = None,
        cursor: Cursor | None = None,
        usage_log: tuple[Usage, ...] | None = None,
        inplace: bool = False,
    ) -> Context | None:
        """Build the next snapshot or mutate the current one when ``inplace``."""
        next_messages = self.messages if messages is None else messages
        next_cursor = self.cursor if cursor is None else cursor
        next_usage_log = self.usage_log if usage_log is None else usage_log

        if inplace:
            object.__setattr__(self, "messages", next_messages)
            object.__setattr__(self, "cursor", next_cursor)
            object.__setattr__(self, "usage_log", next_usage_log)
            return None

        return Context(
            messages=next_messages,
            cursor=next_cursor,
            usage_log=next_usage_log,
            blob_store=self.blob_store,
        )

    # --- Append operations ---

    @overload
    def append(self, message: Message, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def append(self, message: Message, *, inplace: Literal[True]) -> None: ...

    @overload
    def append(self, message: Message, *, inplace: bool) -> Context | None: ...

    def append(self, message: Message, *, inplace: bool = False) -> Context | None:
        """Append a Message.

        Returns a new Context by default. If ``inplace=True``, mutates ``self`` and returns ``None``.
        """
        return self._next(messages=(*self.messages, message), inplace=inplace)

    @overload
    def extend(self, messages: Sequence[Message], *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def extend(self, messages: Sequence[Message], *, inplace: Literal[True]) -> None: ...

    @overload
    def extend(self, messages: Sequence[Message], *, inplace: bool) -> Context | None: ...

    def extend(self, messages: Sequence[Message], *, inplace: bool = False) -> Context | None:
        """Append multiple messages at once."""
        if not messages:
            return self._next(inplace=inplace)
        return self._next(messages=(*self.messages, *messages), inplace=inplace)

    @overload
    def user(self, content: str | Part | Sequence[Part], *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def user(self, content: str | Part | Sequence[Part], *, inplace: Literal[True]) -> None: ...

    @overload
    def user(self, content: str | Part | Sequence[Part], *, inplace: bool) -> Context | None: ...

    def user(self, content: str | Part | Sequence[Part], *, inplace: bool = False) -> Context | None:
        """Append a user message."""
        return self.append(Message(role="user", parts=_normalize_content(content)), inplace=inplace)

    @overload
    def assistant(self, content: str | Part | Sequence[Part], *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def assistant(self, content: str | Part | Sequence[Part], *, inplace: Literal[True]) -> None: ...

    @overload
    def assistant(self, content: str | Part | Sequence[Part], *, inplace: bool) -> Context | None: ...

    def assistant(self, content: str | Part | Sequence[Part], *, inplace: bool = False) -> Context | None:
        """Append an assistant message."""
        return self.append(Message(role="assistant", parts=_normalize_content(content)), inplace=inplace)

    @overload
    def with_cursor(self, cursor: Cursor, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def with_cursor(self, cursor: Cursor, *, inplace: Literal[True]) -> None: ...

    @overload
    def with_cursor(self, cursor: Cursor, *, inplace: bool) -> Context | None: ...

    def with_cursor(self, cursor: Cursor, *, inplace: bool = False) -> Context | None:
        """Return a Context with an updated Cursor."""
        return self._next(cursor=cursor, inplace=inplace)

    @overload
    def with_usage(self, usage: Usage, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def with_usage(self, usage: Usage, *, inplace: Literal[True]) -> None: ...

    @overload
    def with_usage(self, usage: Usage, *, inplace: bool) -> Context | None: ...

    def with_usage(self, usage: Usage, *, inplace: bool = False) -> Context | None:
        """Return a Context with an appended Usage entry."""
        return self._next(usage_log=(*self.usage_log, usage), inplace=inplace)

    @overload
    def clear(self, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def clear(self, *, inplace: Literal[True]) -> None: ...

    @overload
    def clear(self, *, inplace: bool) -> Context | None: ...

    def clear(self, *, inplace: bool = False) -> Context | None:
        """Clear messages, cursor, and usage log."""
        return self._next(messages=(), cursor=Cursor(), usage_log=(), inplace=inplace)

    def clone(self) -> Context:
        """Return a shallow clone of the Context snapshot."""
        return Context(
            messages=self.messages,
            cursor=self.cursor,
            usage_log=self.usage_log,
            blob_store=self.blob_store,
        )

    def to_dict(self, *, include_blob_payloads: bool = False) -> dict[str, object]:
        """Serialize Context to a plain dictionary.

        By default, only BlobReference metadata is serialized. This keeps payloads
        small but requires an external BlobStore to resolve blobs later.

        Set ``include_blob_payloads=True`` to embed referenced blob bytes as
        base64 for portable cross-process persistence. This increases serialized
        size (base64 overhead plus duplicated binary content in the dict).
        """
        payload: dict[str, object] = {
            "messages": [_message_to_dict(message) for message in self.messages],
            "cursor": _cursor_to_dict(self.cursor),
            "usage_log": [_usage_to_dict(usage) for usage in self.usage_log],
        }
        if include_blob_payloads:
            payload["blob_payloads"] = _serialize_blob_payloads(self)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        blob_store: BlobStore | None = None,
    ) -> Context:
        """Deserialize Context from a plain dictionary.

        Blob restore behavior:
        - ``blob_store`` provided: use that store for blob references.
        - ``blob_payloads`` provided: hydrate an internal serialized blob store.
        - neither provided: use a new empty InMemoryBlobStore.
        """
        messages_value = value.get("messages", ())
        if not isinstance(messages_value, list | tuple):
            msg = "Context.messages must be a sequence."
            raise TypeError(msg)
        messages = tuple(
            _message_from_dict(message_value, field_name=f"Context.messages[{index}]")
            for index, message_value in enumerate(messages_value)
        )

        cursor = _cursor_from_dict(value.get("cursor"))

        usage_log_value = value.get("usage_log", ())
        if not isinstance(usage_log_value, list | tuple):
            msg = "Context.usage_log must be a sequence."
            raise TypeError(msg)
        usage_log = tuple(
            _usage_from_dict(usage_value, field_name=f"Context.usage_log[{index}]")
            for index, usage_value in enumerate(usage_log_value)
        )

        blob_payloads = value.get("blob_payloads")
        if blob_store is not None and blob_payloads is not None:
            msg = (
                "Context.from_dict cannot accept both blob_store and blob_payloads. "
                "Pass blob_store=None to hydrate serialized blob payloads."
            )
            raise ValueError(msg)
        if blob_store is None:
            blob_store = _deserialize_blob_store(blob_payloads) if blob_payloads is not None else InMemoryBlobStore()

        return cls(
            messages=messages,
            cursor=cursor,
            usage_log=usage_log,
            blob_store=blob_store,
        )

    def pipe(
        self,
        func: Callable[..., PipeResultT],
        /,
        *args: object,
        **kwargs: object,
    ) -> PipeResultT:
        """Apply a callable to the Context and return its output."""
        return func(self, *args, **kwargs)

    # --- Query operations ---

    def last(self, *, role: Role | None = None) -> Message | None:
        """Return the last message, optionally filtered by role."""
        for message in reversed(self.messages):
            if role is None or message.role == role:
                return message
        return None

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages in chronological order."""
        return iter(self.messages)
