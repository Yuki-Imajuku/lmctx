from typing import Protocol

import pytest

from lmctx.adapters import (
    AnthropicMessagesAdapter,
    BedrockConverseAdapter,
    GoogleGenAIAdapter,
    OpenAIChatCompletionsAdapter,
    OpenAIImagesAdapter,
    OpenAIResponsesAdapter,
    OpenAIResponsesCompactAdapter,
)
from lmctx.plan import AdapterCapabilities, AdapterId

COMMON_FIELDS = {
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "seed",
    "tools",
    "tool_choice",
    "response_schema",
    "response_modalities",
    "extra_body",
    "extra_headers",
    "extra_query",
    "cursor_chaining",
}


class _CapabilityAdapter(Protocol):
    id: AdapterId

    def capabilities(self) -> AdapterCapabilities: ...


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        (
            OpenAIResponsesAdapter(),
            {"seed": "no", "cursor_chaining": "yes", "tools": "yes"},
        ),
        (
            OpenAIResponsesCompactAdapter(),
            {"max_output_tokens": "no", "tools": "no", "cursor_chaining": "yes"},
        ),
        (
            OpenAIChatCompletionsAdapter(),
            {"seed": "yes", "response_schema": "yes", "cursor_chaining": "no"},
        ),
        (
            OpenAIImagesAdapter(),
            {"instructions": "partial", "response_modalities": "partial", "tools": "no"},
        ),
        (
            AnthropicMessagesAdapter(),
            {"seed": "no", "response_schema": "yes", "response_modalities": "no"},
        ),
        (
            GoogleGenAIAdapter(),
            {"seed": "yes", "extra_headers": "no", "response_modalities": "yes"},
        ),
        (
            BedrockConverseAdapter(),
            {"tool_choice": "no", "extra_query": "no", "response_schema": "yes"},
        ),
    ],
)
def test_adapter_capabilities_expected_levels(
    adapter: _CapabilityAdapter,
    expected: dict[str, str],
) -> None:
    capabilities = adapter.capabilities()
    assert isinstance(capabilities, AdapterCapabilities)
    assert capabilities.id == adapter.id
    assert set(capabilities.fields) == COMMON_FIELDS

    for field_name, expected_level in expected.items():
        assert capabilities.level(field_name) == expected_level


@pytest.mark.parametrize(
    "adapter",
    [
        OpenAIResponsesAdapter(),
        OpenAIResponsesCompactAdapter(),
        OpenAIChatCompletionsAdapter(),
        OpenAIImagesAdapter(),
        AnthropicMessagesAdapter(),
        GoogleGenAIAdapter(),
        BedrockConverseAdapter(),
    ],
)
def test_adapter_capability_levels_are_valid(adapter: _CapabilityAdapter) -> None:
    capabilities = adapter.capabilities()
    assert all(level in {"yes", "partial", "no"} for level in capabilities.fields.values())
