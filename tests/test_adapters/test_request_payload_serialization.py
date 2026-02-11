import json

import pytest

from lmctx import Context, RunSpec, ToolSpecification
from lmctx.adapters import (
    AnthropicMessagesAdapter,
    BedrockConverseAdapter,
    GoogleGenAIAdapter,
    OpenAIChatCompletionsAdapter,
    OpenAIImagesAdapter,
    OpenAIResponsesAdapter,
)

AdapterT = (
    OpenAIChatCompletionsAdapter
    | OpenAIResponsesAdapter
    | AnthropicMessagesAdapter
    | BedrockConverseAdapter
    | GoogleGenAIAdapter
    | OpenAIImagesAdapter
)


TOOL = ToolSpecification(
    name="get_weather",
    description="Get weather for a city.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name",
            }
        },
        "required": ["city"],
    },
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


@pytest.mark.parametrize(
    ("adapter", "spec_kwargs"),
    [
        (
            OpenAIChatCompletionsAdapter(),
            {
                "tools": (TOOL,),
                "response_schema": SCHEMA,
                "extra_body": {"extra_body": {"provider": {"order": ["x"], "allow_fallbacks": False}}},
            },
        ),
        (
            OpenAIResponsesAdapter(),
            {
                "tools": (TOOL,),
                "response_schema": SCHEMA,
                "extra_body": {"reasoning": {"effort": "none"}},
            },
        ),
        (
            AnthropicMessagesAdapter(),
            {
                "tools": (TOOL,),
                "response_schema": SCHEMA,
                "extra_body": {"metadata": {"run_id": "r1"}},
            },
        ),
        (
            BedrockConverseAdapter(),
            {
                "tools": (TOOL,),
                "response_schema": SCHEMA,
                "extra_body": {"inferenceConfig": {"topK": 50}},
            },
        ),
        (
            GoogleGenAIAdapter(),
            {
                "tools": (TOOL,),
                "response_schema": SCHEMA,
                "extra_body": {"tool_config": {"function_calling_config": {"mode": "ANY"}}},
            },
        ),
        (
            OpenAIImagesAdapter(),
            {
                "extra_body": {"size": "1024x1024", "quality": "high"},
            },
        ),
    ],
)
def test_plan_request_is_json_serializable(adapter: AdapterT, spec_kwargs: dict[str, object]) -> None:
    message = "Draw a cat." if isinstance(adapter, OpenAIImagesAdapter) else "Hello"
    ctx = Context().user(message)

    spec_data: dict[str, object] = {
        "provider": adapter.id.provider,
        "endpoint": adapter.id.endpoint,
        "model": "test-model",
    }
    spec_data.update(spec_kwargs)
    spec = RunSpec(**spec_data)  # type: ignore[arg-type]

    plan = adapter.plan(ctx, spec)
    json.dumps(plan.request)
