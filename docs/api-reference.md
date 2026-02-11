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

- `AdapterId`, `ExcludedItem`, `RequestPlan`, `LmctxAdapter`
- `AutoAdapter`
- `Context`
- `Part`, `Message`, `Role`, `Cursor`, `Usage`, `ToolSpecification`
- `Instructions`, `RunSpec`
- `BlobReference`, `BlobStore`, `InMemoryBlobStore`, `FileBlobStore`, `put_file`
- `LmctxError`, `ContextError`, `BlobNotFoundError`, `BlobIntegrityError`

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
- `pipe(func, *args, **kwargs) -> Any`
- `__len__() -> int`
- `__iter__() -> Iterator[Message]`

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
class BlobStore(Protocol):
    def put(self, data: bytes, *, media_type: str | None = None, kind: str = "file") -> BlobReference: ...
    def get(self, ref: BlobReference) -> bytes: ...
    def contains(self, ref: BlobReference) -> bool: ...
```

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

### Adapter protocol

```python
class LmctxAdapter(Protocol):
    id: AdapterId
    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan: ...
    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context: ...
```

### `AutoAdapter`

`AutoAdapter` resolves adapters by `(provider, endpoint, api_version)`.

- `plan(ctx, spec) -> RequestPlan`
- `ingest(ctx, response, *, spec) -> Context`
- `resolve(spec) -> LmctxAdapter`
- `register(adapter, *, replace=False) -> None`
- `available_ids() -> tuple[AdapterId, ...]`

## Errors

All custom errors inherit from `LmctxError`.

- `ContextError`
- `BlobNotFoundError`
- `BlobIntegrityError`

Adapter validation and payload mismatch errors may also raise `ValueError` or `TypeError`.
