# Adapters

This page summarizes built-in adapters in `src/lmctx/adapters/` and how `RunSpec` fields map to provider payloads.

## Choose an Adapter

| Goal | Recommended adapter |
|---|---|
| OpenAI primary API with server-side threading | `OpenAIResponsesAdapter` (`responses.create`) |
| OpenAI transcript compaction only | `OpenAIResponsesCompactAdapter` (`responses.compact`) |
| OpenAI-compatible chat endpoints (OpenRouter, Azure, vLLM, SGLang) | `OpenAIChatCompletionsAdapter` (`chat.completions`) |
| OpenAI image generation (`images.generate`) | `OpenAIImagesAdapter` |
| Anthropic Messages + tool use + context-management edits | `AnthropicMessagesAdapter` |
| Gemini text/tool/image generation via google-genai | `GoogleGenAIAdapter` |
| AWS Bedrock Converse | `BedrockConverseAdapter` |

## Built-in IDs

| Adapter class | `AdapterId` (`provider` / `endpoint`) | Typical SDK call |
|---|---|---|
| `OpenAIResponsesAdapter` | `openai` / `responses.create` | `client.responses.create(**plan.request)` |
| `OpenAIResponsesCompactAdapter` | `openai` / `responses.compact` | `client.responses.compact(**plan.request)` |
| `OpenAIChatCompletionsAdapter` | `openai` / `chat.completions` | `client.chat.completions.create(**plan.request)` |
| `OpenAIImagesAdapter` | `openai` / `images.generate` | `client.images.generate(**plan.request)` |
| `AnthropicMessagesAdapter` | `anthropic` / `messages.create` | `client.messages.create(**plan.request)` |
| `GoogleGenAIAdapter` | `google` / `models.generate_content` | `client.models.generate_content(**plan.request)` |
| `BedrockConverseAdapter` | `bedrock` / `converse` | `client.converse(**plan.request)` |

`AutoAdapter` resolves by `(provider, endpoint, api_version)` with fallback to `(provider, endpoint, None)`.

## Capability API

Each adapter exposes runtime capability metadata:

```python
from lmctx.adapters import OpenAIChatCompletionsAdapter

adapter = OpenAIChatCompletionsAdapter()
caps = adapter.capabilities()
print(caps.level("seed"))  # "yes"
print(caps.is_supported("cursor_chaining"))  # False
```

`AutoAdapter` also provides capability lookup for a `RunSpec`:

```python
caps = router.capabilities(spec)
all_caps = router.available_capabilities()
```

## RunSpec Support Matrix

Legend:

- `yes`: directly mapped
- `partial`: mapped with constraints or warnings
- `no`: ignored/excluded

| Field | OpenAI Responses | OpenAI Compact | OpenAI Chat | OpenAI Images | Anthropic Messages | Google GenAI | Bedrock Converse |
|---|---|---|---|---|---|---|---|
| `instructions` | yes (`instructions`) | yes (`instructions`) | yes (`system`/`developer` messages) | partial (prepended to prompt) | yes (merged to `system`) | yes (`config.system_instruction`) | yes (`system` blocks) |
| `max_output_tokens` | yes | no | yes (`max_tokens`) | no | yes (defaults to 4096 if unset) | yes (`config.max_output_tokens`) | yes (`inferenceConfig.maxTokens`) |
| `temperature`, `top_p` | yes | no | yes | no | yes | yes | yes |
| `seed` | no | no | yes | no | no | yes | no |
| `tools` | yes | no | yes | no | yes | yes | yes |
| `tool_choice` | yes | no | yes | no | yes | yes (`config.tool_config`) | no |
| `response_schema` | yes (`text.format`) | no | yes (`response_format`) | no | yes (`output_config.format`) | yes (`response_schema` + JSON mime type) | yes (`outputConfig.textFormat.structure.jsonSchema`) |
| `response_modalities` | yes (`modalities`) | no | yes (`modalities`) | partial (DALL-E `response_format` only) | no | yes (`config.response_modalities`) | no |
| `extra_body` | yes (deep merge request) | yes (deep merge request) | yes (deep merge request) | yes (deep merge request) | yes (deep merge request) | yes (deep merge `config`) | yes (deep merge request) |
| `extra_headers` / `extra_query` | yes | yes | yes | yes | yes | no | no |
| `base_url` | surfaced in `RequestPlan.extra` | surfaced in `RequestPlan.extra` | surfaced in `RequestPlan.extra` | surfaced in `RequestPlan.extra` | surfaced in `RequestPlan.extra` | surfaced in `RequestPlan.extra` | surfaced in `RequestPlan.extra` |
| Cursor chaining (`Context.cursor`) | yes (`previous_response_id`) | yes (`previous_response_id`) | no | no | no | no | no |

## Provider Notes

### OpenAI Responses (`responses.create`)

- Uses mixed `input` items instead of `messages`
- Supports `Part(type="file")` as `input_file` (`file_id`, `file_data`, `file_url`, or blob-backed file payload)
- Serializes tool results to `function_call_output`
- Ingests reasoning as `Part(type="thinking")`
- Ingests compaction as blob-backed `Part(type="compaction")`

### OpenAI Responses Compact (`responses.compact`)

- Focused on compaction; generation, tools, schema, and modalities are excluded
- Ingest keeps only `compaction` output parts

### OpenAI Chat Completions

- Maps context to `messages` format
- Tool results become `role="tool"` messages
- Supports user content blocks: text, image, file
- Does not use lmctx cursor chaining

### OpenAI Images (`images.generate`)

- Uses latest user text as prompt source
- Prepends `instructions` text to prompt when provided
- `response_modalities` maps to `response_format` only for DALL-E models
- Ingest stores generated bytes as blob-backed `Part(type="image")` when available

### Anthropic Messages

- Context `system`/`developer` text and `RunSpec.instructions` are merged into request `system`
- Consecutive same-role messages are merged to satisfy alternation constraints
- Defaults `max_tokens` to `4096` if `max_output_tokens` is unset (with warning)
- Context-management compaction is supported through `RunSpec.extra_body`

### Google GenAI (`models.generate_content`)

- Assistant role maps to `model`, tool role maps to `user`
- Tool calls/responses map to `function_call` / `function_response`
- `extra_body` merges into `config` (not request root)
- Reads usage metadata from snake_case and camelCase payload variants

### Bedrock Converse

- Uses Bedrock camelCase structure (`modelId`, `inferenceConfig`, `toolConfig`)
- System/developer text goes into top-level `system` blocks
- Structured output maps to `outputConfig.textFormat`
- Transport overrides (`extra_headers`, `extra_query`) are intentionally not mapped

## AutoAdapter Example

```python
from lmctx import AutoAdapter, Context, RunSpec

router = AutoAdapter()
ctx = Context().user("Hello")
spec = RunSpec(provider="openai", endpoint="responses.create", model="gpt-4o-mini")

plan = router.plan(ctx, spec)
# call provider SDK with plan.request
# response = ...
# ctx = router.ingest(ctx, response, spec=spec)
```

If no adapter matches, `AutoAdapter.resolve()` raises `ValueError` with available targets.
