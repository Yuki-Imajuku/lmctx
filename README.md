# lmctx

[![GitHub license](https://img.shields.io/github/license/Yuki-Imajuku/lmctx?logo=github)](https://github.com/Yuki-Imajuku/lmctx/blob/main/LICENSE)
[![GitHub Actions lint](https://img.shields.io/github/actions/workflow/status/Yuki-Imajuku/lmctx/lint.yml?branch=main&label=lint)](https://github.com/Yuki-Imajuku/lmctx/actions/workflows/lint.yml)
[![GitHub Actions test](https://img.shields.io/github/actions/workflow/status/Yuki-Imajuku/lmctx/test.yml?branch=main&label=test)](https://github.com/Yuki-Imajuku/lmctx/actions/workflows/test.yml)
[![Coverage >=90%](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen)](https://github.com/Yuki-Imajuku/lmctx/actions/workflows/test.yml)
[![GitHub stars](https://img.shields.io/github/stars/Yuki-Imajuku/lmctx?logo=github)](https://github.com/Yuki-Imajuku/lmctx/stargazers)
[![PyPI version](https://img.shields.io/pypi/v/lmctx)](https://pypi.org/project/lmctx/)
[![Python versions](https://img.shields.io/pypi/pyversions/lmctx)](https://pypi.org/project/lmctx/)

**Context Kernel for LLM APIs.**
Standardize what happens before and after every model call, while keeping execution in your own runtime.

- **Before call**: `adapter.plan(context, spec)` builds provider-ready payloads and diagnostics
- **After call**: `adapter.ingest(context, response, spec=...)` normalizes output back into `Context`
- **Boundary**: lmctx never sends HTTP requests, executes tools, or orchestrates loops

## Why lmctx

- **Append-only, snapshot-friendly context model** (`Context`) with immutable-by-default updates
- **Unified part model** (`Part`) for text, images, files, tool calls/results, thinking, compaction
- **Loss-resistant round-trips** for opaque provider payloads through `provider_raw` and blob references
- **Pluggable blob storage** (`InMemoryBlobStore`, `FileBlobStore`, or custom `BlobStore`)
- **Provider adapters + auto routing** via `AutoAdapter` on `(provider, endpoint, api_version)`
- **Explainable planning** through `RequestPlan` (`included`, `excluded`, `warnings`, `errors`)
- **Minimal dependencies** (core package has no runtime deps; provider SDKs are optional extras)

## Install

```bash
pip install lmctx

# provider extras (optional)
pip install 'lmctx[openai]'
pip install 'lmctx[anthropic]'
pip install 'lmctx[google]'
pip install 'lmctx[bedrock]'
pip install 'lmctx[all]'
```

## 5-Minute Integration

```python
from openai import OpenAI

from lmctx import AutoAdapter, Context, RunSpec
from lmctx.spec import Instructions

# 1) Build conversation state
ctx = Context().user("What is the capital of France?")

# 2) Describe runtime call settings
spec = RunSpec(
    provider="openai",
    endpoint="responses.create",
    model="gpt-4o-mini",
    instructions=Instructions(system="You are concise and accurate."),
)

# 3) Build request payload with lmctx
router = AutoAdapter()
plan = router.plan(ctx, spec)

# 4) Execute with provider SDK in your own code
client = OpenAI()
response = client.responses.create(**plan.request)

# 5) Normalize response back into Context
ctx = router.ingest(ctx, response, spec=spec)

assistant = ctx.last(role="assistant")
if assistant:
    print(assistant.parts[0].text)
```

## Core Types

| Type | Role |
|---|---|
| `Context` | Append-only conversation log (`messages`, `cursor`, `usage_log`, `blob_store`) |
| `Part` / `Message` | Canonical content model shared across adapters |
| `RunSpec` | Call configuration (provider, endpoint, model, tools, schema, extras) |
| `RequestPlan` | Planned payload + diagnostics for observability and debugging |
| `BlobReference` / `BlobStore` | Out-of-line binary/opaque payload storage with integrity verification |

## Built-in Adapters

| Adapter | `RunSpec` selector | Typical SDK call |
|---|---|---|
| `OpenAIResponsesAdapter` | `openai` / `responses.create` | `client.responses.create(**plan.request)` |
| `OpenAIResponsesCompactAdapter` | `openai` / `responses.compact` | `client.responses.compact(**plan.request)` |
| `OpenAIChatCompletionsAdapter` | `openai` / `chat.completions` | `client.chat.completions.create(**plan.request)` |
| `OpenAIImagesAdapter` | `openai` / `images.generate` | `client.images.generate(**plan.request)` |
| `AnthropicMessagesAdapter` | `anthropic` / `messages.create` | `client.messages.create(**plan.request)` |
| `GoogleGenAIAdapter` | `google` / `models.generate_content` | `client.models.generate_content(**plan.request)` |
| `BedrockConverseAdapter` | `bedrock` / `converse` | `client.converse(**plan.request)` |

## Documentation

- [`docs/README.md`](docs/README.md): doc map and recommended reading paths
- [`docs/architecture.md`](docs/architecture.md): boundaries, lifecycle, extension points
- [`docs/data-model.md`](docs/data-model.md): concrete type contracts and invariants
- [`docs/api-reference.md`](docs/api-reference.md): public API quick reference
- [`docs/adapters.md`](docs/adapters.md): adapter matrix and provider caveats
- [`docs/examples.md`](docs/examples.md): runnable examples and prerequisites
- [`docs/logs.md`](docs/logs.md): log files and regeneration workflow

## Examples

Scripts are in [`examples/`](examples/):

- Core (no API keys): `quickstart.py`, `multimodal.py`, `blob_stores.py`, `tool_calling.py`
- OpenAI: `api_openai_responses.py`, `api_openai_compact.py`, `api_openai_chat.py`, `api_openai_images.py`
- Anthropic: `api_anthropic.py`, `api_anthropic_compact.py`
- Google: `api_google_genai.py`, `api_google_image_generation.py`
- Bedrock: `api_bedrock.py`

Run one:

```bash
uv run python examples/quickstart.py
```

## Recorded Logs

Example outputs can be stored locally under [`examples/logs/`](examples/logs/) (git-ignored by default).
See [`docs/logs.md`](docs/logs.md) for mapping and regeneration commands.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines.

```bash
uv sync --all-extras --dev
make check
```

## Requirements

- Python `>=3.10,<3.15`

## License

Apache License 2.0. See [LICENSE](LICENSE).
