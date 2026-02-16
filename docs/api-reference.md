# API Reference

Quick reference for the public API exported by `lmctx`.
For detailed behavior and invariants, combine this page with [`data-model.md`](data-model.md) and [`adapters.md`](adapters.md).

## Import Surfaces

```python
import lmctx
from lmctx import Context, RunSpec, AutoAdapter
from lmctx.adapters import OpenAIResponsesAdapter
```

### Root exports (`from lmctx import ...`)

- `AdapterId`, `CapabilityLevel`, `AdapterCapabilities`, `ExcludedItem`, `RequestPlan`, `LmctxAdapter`
- `AutoAdapter`
- `Context`
- `Part`, `Message`, `Role`, `Cursor`, `Usage`, `ToolSpecification`
- `Instructions`, `RunSpec`
- `BlobReference`, `BlobStore`, `InMemoryBlobStore`, `FileBlobStore`, `put_file`
- `LmctxError`, `PlanValidationError`, `ContextError`, `BlobNotFoundError`, `BlobIntegrityError`

### Adapter exports (`from lmctx.adapters import ...`)

- `AutoAdapter`
- `OpenAIResponsesAdapter`
- `OpenAIResponsesCompactAdapter`
- `OpenAIChatCompletionsAdapter`
- `OpenAIImagesAdapter`
- `AnthropicMessagesAdapter`
- `GoogleGenAIAdapter`
- `BedrockConverseAdapter`

## Core Conversation API

### `Context`

`Context` is an append-only log with immutable-by-default updates.

```python
@dataclass(frozen=True, slots=True)
class Context:
    messages: tuple[Message, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)
    usage_log: tuple[Usage, ...] = ()
    blob_store: BlobStore = field(default_factory=InMemoryBlobStore)
```

Mutation methods support both immutable and in-place modes:

- `append(message, *, inplace=False)`
- `extend(messages, *, inplace=False)`
- `user(content, *, inplace=False)`
- `assistant(content, *, inplace=False)`
- `with_cursor(cursor, *, inplace=False)`
- `with_usage(usage, *, inplace=False)`
- `clear(*, inplace=False)`

Behavior:

- `inplace=False` (default): returns a new `Context`
- `inplace=True`: mutates current instance and returns `None`

Query/helper methods:

- `last(*, role=None) -> Message | None`
- `clone() -> Context`
- `to_dict(*, include_blob_payloads=False) -> dict[str, object]`
- `from_dict(value, *, blob_store=None) -> Context` (classmethod)
- `pipe(func, *args, **kwargs) -> Any`
- `__len__() -> int`
- `__iter__() -> Iterator[Message]`

Serialization notes:

- `to_dict()` default (`include_blob_payloads=False`) stores only `BlobReference` metadata, not bytes.
- For cross-process replay with `InMemoryBlobStore`, use `include_blob_payloads=True` or provide a persistent `blob_store` to `from_dict(...)`.
- `include_blob_payloads=True` is heavier because blob bytes are base64-embedded in the serialized payload.

### `Message` and `Part`

```python
@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    parts: tuple[Part, ...]
    id: str | None = None
    provider: str | None = None
    turn_id: str | None = None
    timestamp: datetime
```

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

Common `Part.type` values:

- `text`, `json`
- `image`, `audio`, `file`
- `tool_call`, `tool_result`
- `thinking`, `compaction`

## Blob API

### `BlobReference`

```python
@dataclass(frozen=True, slots=True)
class BlobReference:
    id: str
    sha256: str
    media_type: str | None
    kind: str
    size: int
```

### `BlobStore` protocol

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

- `InMemoryBlobStore()`
- `FileBlobStore(root)`
- `put_file(store, path, kind=None)` helper

## Planning API

### `Instructions` and `RunSpec`

```python
@dataclass(frozen=True, slots=True)
class Instructions:
    system: str | None = None
    developer: str | None = None
```

Helpers:

- `to_dict() -> dict[str, object]`
- `from_dict(value) -> Instructions` (classmethod)

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
    extra_body: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    extra_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    extra_query: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    base_url: str | None = None
```

Helpers:

- `to_dict() -> dict[str, object]`
- `from_dict(value) -> RunSpec` (classmethod)

### `RequestPlan`

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

`request` is the payload you pass to SDK calls. `extra` carries non-request hints such as `base_url`.

Helpers:

- `to_dict() -> dict[str, object]`
- `from_dict(value) -> RequestPlan` (classmethod)

Validation helpers:

- `unused_parameter_warnings() -> tuple[str, ...]`
- `warning_messages(include_unused_parameters=True) -> tuple[str, ...]`
- `assert_valid(*, fail_on_warnings=False, fail_on_excluded=False) -> None`

### `AdapterCapabilities`

```python
CapabilityLevel = Literal["yes", "partial", "no"]

@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    id: AdapterId
    fields: Mapping[str, CapabilityLevel]
    notes: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
```

- `level(field_name) -> CapabilityLevel | None`
- `is_supported(field_name, *, allow_partial=True) -> bool`

### Adapter protocol

```python
class LmctxAdapter(Protocol):
    id: AdapterId
    def capabilities(self) -> AdapterCapabilities: ...
    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan: ...
    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context: ...
```

### `AutoAdapter`

`AutoAdapter` resolves adapters by `(provider, endpoint, api_version)`.

- `plan(ctx, spec) -> RequestPlan`
- `ingest(ctx, response, *, spec) -> Context`
- `capabilities(spec) -> AdapterCapabilities`
- `resolve(spec) -> LmctxAdapter`
- `register(adapter, *, replace=False) -> None`
- `available_ids() -> tuple[AdapterId, ...]`
- `available_capabilities() -> tuple[AdapterCapabilities, ...]`

## Errors

All custom errors inherit from `LmctxError`.

- `ContextError`
- `PlanValidationError`
- `BlobNotFoundError`
- `BlobIntegrityError`

Adapter validation and payload mismatch errors may also raise `ValueError` or `TypeError`.
