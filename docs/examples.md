# Examples Guide

Runnable scripts live under [`examples/`](../examples/).

Quick run:

```bash
uv run python examples/quickstart.py
```

## Script Categories

### Core examples (no API keys)

| Script | Purpose |
|---|---|
| [`examples/quickstart.py`](../examples/quickstart.py) | Immutable-by-default `Context` usage and querying |
| [`examples/multimodal.py`](../examples/multimodal.py) | Text + image parts with blob references |
| [`examples/blob_stores.py`](../examples/blob_stores.py) | `InMemoryBlobStore`, `FileBlobStore`, `put_file()` |
| [`examples/tool_calling.py`](../examples/tool_calling.py) | Tool call / tool result patterns |

### Provider examples

| Provider | Scripts |
|---|---|
| OpenAI | `api_openai_responses.py`, `api_openai_compact.py`, `api_openai_chat.py`, `api_openai_images.py` |
| Anthropic | `api_anthropic.py`, `api_anthropic_compact.py` |
| Google | `api_google_genai.py`, `api_google_image_generation.py` |
| Bedrock | `api_bedrock.py` |

## Prerequisites by Provider

### OpenAI (Responses, Compact, Images)

```bash
pip install 'lmctx[openai]'
export OPENAI_API_KEY=...
```

### OpenRouter via Chat Completions example

```bash
pip install 'lmctx[openai]'
export OPENROUTER_API_KEY=...
# optional ranking headers:
# export OPENROUTER_SITE_URL=https://example.com
# export OPENROUTER_APP_NAME=lmctx-demo
```

### Anthropic

```bash
pip install 'lmctx[anthropic]'
export ANTHROPIC_API_KEY=...
```

### Google GenAI

```bash
pip install 'lmctx[google]'
export GOOGLE_API_KEY=...
```

### AWS Bedrock

```bash
pip install 'lmctx[bedrock]'
# configure AWS credentials/region for bedrock-runtime
```

## Run Commands

```bash
# core
uv run python examples/quickstart.py
uv run python examples/multimodal.py
uv run python examples/blob_stores.py
uv run python examples/tool_calling.py

# openai
uv run python examples/api_openai_responses.py
uv run python examples/api_openai_compact.py
uv run python examples/api_openai_chat.py
uv run python examples/api_openai_images.py

# anthropic
uv run python examples/api_anthropic.py
uv run python examples/api_anthropic_compact.py

# google
uv run python examples/api_google_genai.py
uv run python examples/api_google_image_generation.py

# bedrock
uv run python examples/api_bedrock.py
```

## Local Assets and Outputs

Some examples use assets in [`examples/assets/`](../examples/assets/):

- [`examples/assets/image.jpg`](../examples/assets/image.jpg)
- [`examples/assets/notebook.pdf`](../examples/assets/notebook.pdf)

Image-generation scripts write output files under [`examples/outputs/`](../examples/outputs/).

## Common Flow in Provider Examples

Most provider scripts follow this pattern:

1. Build `Context` and `RunSpec`
2. `plan = adapter.plan(ctx, spec)`
3. Execute provider SDK call using `plan.request`
4. `ctx = adapter.ingest(ctx, response, spec=spec)`
5. Inspect assistant parts, usage log, and (when supported) cursor state

## Notes

- Model names in examples are snapshots and may need updates over time.
- Provider outputs are often nondeterministic.
- To preserve reproducible demonstrations, capture local logs using [`logs.md`](logs.md).
