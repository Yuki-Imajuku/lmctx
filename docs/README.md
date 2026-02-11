# Documentation

Implementation-level documentation for `lmctx`.
If you are new to the project, start with [`../README.md`](../README.md), then use this directory for details.

## Read by Goal

### I want to integrate quickly

1. [`../README.md`](../README.md)
2. [`examples.md`](examples.md)
3. [`adapters.md`](adapters.md)
4. [`api-reference.md`](api-reference.md)

### I want to understand internals

1. [`architecture.md`](architecture.md)
2. [`data-model.md`](data-model.md)
3. [`adapters.md`](adapters.md)

### I want to debug payload issues

1. [`adapters.md`](adapters.md)
2. [`api-reference.md`](api-reference.md)
3. [`logs.md`](logs.md)

## File Map

- [`architecture.md`](architecture.md)
  Project boundaries, lifecycle (`plan -> execute -> ingest`), invariants, extension points.
- [`data-model.md`](data-model.md)
  Canonical type contracts for `Context`, `Part`, `RunSpec`, `RequestPlan`, blobs.
- [`api-reference.md`](api-reference.md)
  Public API reference: exported classes, helper functions, and key method signatures.
- [`adapters.md`](adapters.md)
  Built-in adapter IDs, capability matrix, and provider-specific conversion caveats.
- [`examples.md`](examples.md)
  Runnable script guide and provider prerequisites.
- [`logs.md`](logs.md)
  Recorded log locations and regeneration workflow.

## Fast Links

- English README: [`../README.md`](../README.md)
- Japanese README: [`../README.ja.md`](../README.ja.md)
- Contributing guide: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- Public exports: [`../src/lmctx/__init__.py`](../src/lmctx/__init__.py)
- Provider adapters: [`../src/lmctx/adapters/`](../src/lmctx/adapters/)
- Examples: [`../examples/`](../examples/)
- Local logs dir: [`../examples/logs/`](../examples/logs/)

## Scope

`lmctx` intentionally focuses on type-safe request planning and response normalization:

- builds provider payloads (`plan`)
- normalizes provider responses (`ingest`)
- preserves binary/opaque payloads through `BlobStore`

It does not execute network calls, tools, retries, or orchestration loops.
