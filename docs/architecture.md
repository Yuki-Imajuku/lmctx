# Architecture

This document describes how `lmctx` is structured and how data flows through the library.

## Purpose and Boundaries

`lmctx` is a context kernel for LLM APIs.

It is responsible for:
- Converting `Context + RunSpec` into provider payloads (`adapter.plan`)
- Converting provider responses back into normalized `Context` entries (`adapter.ingest`)
- Preserving opaque/binary payloads through `BlobStore` references

It is intentionally not responsible for:
- Executing HTTP calls
- Running tools
- Retry logic, rate limits, or agent orchestration

## Lifecycle

```text
Context + RunSpec
  -> adapter.plan(...)
  -> RequestPlan(request=provider payload, diagnostics...)
  -> SDK call in user code
  -> adapter.ingest(...)
  -> new Context (assistant parts + usage + optional cursor updates)
```

## Module Layout

```text
src/lmctx/
  errors.py
  blobs/
    _reference.py
    _store.py
    _memory.py
    _file.py
    _helpers.py
  types.py
  context.py
  spec.py
  plan.py
  adapters/
    _util.py
    _openai_responses.py
    _openai_chat.py
    _openai_images.py
    _anthropic.py
    _google.py
    _bedrock.py
    _auto.py
```

Dependency direction is one-way from foundational types to adapters.
Adapters depend on everything above them; core types do not depend on adapters.

## Core Building Blocks

### `Context`

`Context` is an append-only conversation log (`messages`, `cursor`, `usage_log`, `blob_store`).

- Default mutation style is immutable: methods return a new snapshot.
- `inplace=True` is supported explicitly and returns `None`.
- All snapshots in one chain share the same `blob_store` instance.

### `Part` / `Message`

`Message` is `(role, parts)`.
`Part` is the normalized content unit across providers (`text`, `image`, `tool_call`, `tool_result`, `thinking`, `compaction`, `file`, ...).

### `RunSpec`

`RunSpec` describes how to call a model (provider, endpoint, model, generation knobs, tools, structured output, escape hatches).

### `RequestPlan`

`RequestPlan` is the output of planning and includes:
- `request`: provider payload you can pass to SDK
- `included`: what was mapped
- `excluded`: what was dropped and why
- `warnings` and `errors`: planning diagnostics
- `extra`: non-request hints (for example `base_url`, `api_version`)

## Adapter Contract

Each adapter satisfies `LmctxAdapter`:

```python
class LmctxAdapter(Protocol):
    id: AdapterId
    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan: ...
    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context: ...
```

Design implications:
- Providers are pluggable without inheritance requirements.
- Adapter validation is strict (`provider`/`endpoint` mismatch raises `ValueError`).
- Conversion losses are explicit through `RequestPlan.excluded`.

## Auto Routing (`AutoAdapter`)

`AutoAdapter` registers adapters keyed by `(provider, endpoint, api_version)`.
Resolution order:

1. Exact key match
2. Fallback to `(provider, endpoint, None)`
3. Otherwise raise `ValueError` with available targets

This makes runtime dispatch deterministic while still allowing versioned adapters.

## Blob and Opaque Payload Strategy

Binary or opaque payloads are stored out-of-line and referenced by `BlobReference`.
Both built-in stores (`InMemoryBlobStore`, `FileBlobStore`) verify SHA-256 on every read.

Why this matters:
- Avoids corrupting opaque provider payloads during JSON round-trips
- Keeps `Context` snapshots lightweight
- Makes storage backend swappable

## Invariants

Implementation-level invariants enforced by dataclass design and helpers:

- Snapshot containers are normalized to immutable tuples
- Mapping payloads are frozen into read-only mappings in core types
- Adapter requests are normalized to plain dict/list structures in `RequestPlan`
- Usage details are preserved in `Usage.provider_usage` for provider-specific auditing

## Error Model

All library-level exceptions inherit from `LmctxError`.
Key error types:
- `BlobNotFoundError`
- `BlobIntegrityError`
- `ContextError`

Adapter conversion and routing failures typically raise `ValueError` or `TypeError` when inputs are structurally invalid.

## Extension Points

Common ways to extend `lmctx`:

1. Add a custom adapter implementing `LmctxAdapter` and register it in `AutoAdapter`.
2. Add a custom blob backend implementing `BlobStore` (`put_blob/get_blob/has_blob/delete_blob/list_blobs/prune_blobs`).
3. Use `RunSpec.extra_body`, `extra_headers`, `extra_query` for provider features that are not first-class fields.

This keeps the core stable while allowing provider-specific evolution.
