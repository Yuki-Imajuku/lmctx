# Data Model

This document describes the canonical types used by `lmctx`.

## Object Graph

```text
Context
  messages: tuple[Message, ...]
    Message
      role: Role
      parts: tuple[Part, ...]
      id/provider/turn_id/timestamp
  cursor: Cursor
  usage_log: tuple[Usage, ...]
  blob_store: BlobStore
```

## `Role`

```python
Role = Literal["system", "developer", "user", "assistant", "tool"]
```

Roles are normalized across providers even when provider-native role names differ.

## `Part`

`Part` is the basic content unit.

```python
@dataclass(frozen=True, slots=True)
class Part:
    type: str
    text: str | None = None
    json: Mapping[str, object] | None = None
    blob: BlobReference | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: Mapping[str, object] | None = None
    tool_output: object | None = None
    provider_raw: Mapping[str, object] | None = None
```

Typical `type` values include:
- `text`, `json`
- `image`, `audio`, `file`
- `tool_call`, `tool_result`
- `thinking`, `compaction`

`provider_raw` is optional raw payload storage used for round-trip fidelity/debugging.

## `Message`

```python
@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    parts: tuple[Part, ...]
    id: str | None = None
    provider: str | None = None
    turn_id: str | None = None
    timestamp: datetime = datetime.now(timezone.utc)
```

A single assistant message may contain multiple parts (for example text + tool calls + thinking).

## `ToolSpecification`

```python
@dataclass(frozen=True, slots=True)
class ToolSpecification:
    name: str
    description: str
    input_schema: Mapping[str, object]
```

`input_schema` is JSON-schema-like and normalized to an immutable mapping.

## `BlobReference` and `BlobStore`

```python
@dataclass(frozen=True, slots=True)
class BlobReference:
    id: str
    sha256: str
    media_type: str | None
    kind: str
    size: int
```

```python
@dataclass(frozen=True, slots=True)
class BlobEntry:
    ref: BlobReference
    created_at: datetime
    last_accessed_at: datetime | None = None

@dataclass(frozen=True, slots=True)
class PruneReport:
    deleted: tuple[BlobEntry, ...]
    bytes_freed: int
    examined: int
    remaining: int
    dry_run: bool

class BlobStore(Protocol):
    def put_blob(self, data: bytes, *, media_type: str | None = None, kind: str = "file") -> BlobReference: ...
    def get_blob(self, ref: BlobReference) -> bytes: ...
    def has_blob(self, ref: BlobReference) -> bool: ...
    def delete_blob(self, ref_or_id: BlobReference | str) -> bool: ...
    def list_blobs(
        self,
        *,
        kind: str | None = None,
        media_type: str | None = None,
    ) -> tuple[BlobEntry, ...]: ...
    def prune_blobs(
        self,
        *,
        older_than: datetime | None = None,
        max_bytes: int | None = None,
        kind: str | None = None,
        media_type: str | None = None,
        dry_run: bool = False,
    ) -> PruneReport: ...
```

`prune_blobs(older_than=None, max_bytes=None)` is a no-op and returns an empty `PruneReport`.

Built-ins:
- `InMemoryBlobStore`
- `FileBlobStore`

Both verify SHA-256 integrity in `get_blob()` and implement operational lifecycle APIs (`delete_blob/list_blobs/prune_blobs`).

`put_file(store, path, kind=None)` reads a local file, infers media type/kind, and stores it.

## `Cursor`

```python
@dataclass(frozen=True, slots=True)
class Cursor:
    last_response_id: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
```

Cursor stores provider conversation state when applicable.

## `Usage`

```python
@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_usage: Mapping[str, object] = MappingProxyType({})
```

`provider_usage` keeps raw provider-specific usage payloads in immutable form.

## `Context`

```python
@dataclass(frozen=True, slots=True)
class Context:
    messages: tuple[Message, ...] = ()
    cursor: Cursor = Cursor()
    usage_log: tuple[Usage, ...] = ()
    blob_store: BlobStore = InMemoryBlobStore()
```

Mutation methods:
- `append`, `extend`
- `user`, `assistant`
- `with_cursor`, `with_usage`
- `clear`

By default these return a new `Context`; with `inplace=True` they mutate and return `None`.

Query/helpers:
- `last(role=None)`
- `clone()`
- `pipe(func, *args, **kwargs)`
- iterable / `len()` support

Serialization note:
- `Context.to_dict()` stores blob references by default, not blob bytes.
- To persist blobs across process boundaries, either:
  - call `to_dict(include_blob_payloads=True)`, or
  - restore with a persistent external `blob_store` via `Context.from_dict(..., blob_store=...)`.
- Embedding blob payloads is larger because bytes are base64-encoded into the serialized object.

## `Instructions` and `RunSpec`

```python
@dataclass(frozen=True, slots=True)
class Instructions:
    system: str | None = None
    developer: str | None = None
```

```python
@dataclass(frozen=True, slots=True)
class RunSpec:
    provider: str
    endpoint: str
    model: str
    api_version: str | None = None
    instructions: Instructions | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    tools: tuple[ToolSpecification, ...] = ()
    tool_choice: object | None = None
    response_schema: Mapping[str, object] | None = None
    response_modalities: tuple[str, ...] = ()
    extra_body: Mapping[str, object] = MappingProxyType({})
    extra_headers: Mapping[str, str] = MappingProxyType({})
    extra_query: Mapping[str, str] = MappingProxyType({})
    base_url: str | None = None
```

`RunSpec` describes call configuration, not conversation content.

## `RequestPlan`, `ExcludedItem`, `AdapterId`

```python
@dataclass(frozen=True, slots=True)
class AdapterId:
    provider: str
    endpoint: str
    api_version: str | None = None
```

```python
@dataclass(frozen=True, slots=True)
class ExcludedItem:
    description: str
    reason: str
```

```python
@dataclass(frozen=True, slots=True)
class RequestPlan:
    request: dict[str, Any]
    included: tuple[str, ...] = ()
    excluded: tuple[ExcludedItem, ...] = ()
    must_roundtrip: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    token_estimate: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
```

`request` and `extra` are normalized to plain Python containers in `__post_init__`.

## Immutability Rules

Core dataclasses are frozen.
Mutable mappings/sequences passed into types are normalized to immutable structures where relevant.
This prevents accidental mutation of snapshots after construction.
