# Example Logs

Runtime logs can be stored under [`examples/logs/`](../examples/logs/).
This directory is git-ignored by default via [`examples/logs/.gitignore`](../examples/logs/.gitignore).

## Script-to-Log Map

| Example script | Log file |
|---|---|
| [`examples/quickstart.py`](../examples/quickstart.py) | [`examples/logs/quickstart.log`](../examples/logs/quickstart.log) |
| [`examples/multimodal.py`](../examples/multimodal.py) | [`examples/logs/multimodal.log`](../examples/logs/multimodal.log) |
| [`examples/blob_stores.py`](../examples/blob_stores.py) | [`examples/logs/blob_stores.log`](../examples/logs/blob_stores.log) |
| [`examples/tool_calling.py`](../examples/tool_calling.py) | [`examples/logs/tool_calling.log`](../examples/logs/tool_calling.log) |
| [`examples/api_openai_responses.py`](../examples/api_openai_responses.py) | [`examples/logs/api_openai_responses.log`](../examples/logs/api_openai_responses.log) |
| [`examples/api_openai_compact.py`](../examples/api_openai_compact.py) | [`examples/logs/api_openai_compact.log`](../examples/logs/api_openai_compact.log) |
| [`examples/api_openai_chat.py`](../examples/api_openai_chat.py) | [`examples/logs/api_openai_chat.log`](../examples/logs/api_openai_chat.log) |
| [`examples/api_openai_images.py`](../examples/api_openai_images.py) | [`examples/logs/api_openai_images.log`](../examples/logs/api_openai_images.log) |
| [`examples/api_anthropic.py`](../examples/api_anthropic.py) | [`examples/logs/api_anthropic.log`](../examples/logs/api_anthropic.log) |
| [`examples/api_anthropic_compact.py`](../examples/api_anthropic_compact.py) | [`examples/logs/api_anthropic_compact.log`](../examples/logs/api_anthropic_compact.log) |
| [`examples/api_google_genai.py`](../examples/api_google_genai.py) | [`examples/logs/api_google_genai.log`](../examples/logs/api_google_genai.log) |
| [`examples/api_google_image_generation.py`](../examples/api_google_image_generation.py) | [`examples/logs/api_google_image_generation.log`](../examples/logs/api_google_image_generation.log) |
| [`examples/api_bedrock.py`](../examples/api_bedrock.py) | [`examples/logs/api_bedrock.log`](../examples/logs/api_bedrock.log) |

## Regenerate Logs

Create the directory once:

```bash
mkdir -p examples/logs
```

Core examples:

```bash
uv run python examples/quickstart.py > examples/logs/quickstart.log
uv run python examples/multimodal.py > examples/logs/multimodal.log
uv run python examples/blob_stores.py > examples/logs/blob_stores.log
uv run python examples/tool_calling.py > examples/logs/tool_calling.log
```

Provider examples:

```bash
uv run python examples/<script>.py > examples/logs/<script>.log
```

Use environment variables listed in [`examples.md`](examples.md) before running provider scripts.

## Practical Notes

- Logs are local artifacts and usually should not be committed.
- Provider logs may contain generated content and usage metadata; sanitize before sharing.
- Output can vary across runs due to model nondeterminism and provider-side changes.
