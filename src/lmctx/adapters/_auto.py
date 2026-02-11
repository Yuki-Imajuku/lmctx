"""Automatic adapter routing based on RunSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lmctx.adapters._anthropic import AnthropicMessagesAdapter
from lmctx.adapters._bedrock import BedrockConverseAdapter
from lmctx.adapters._google import GoogleGenAIAdapter
from lmctx.adapters._openai_chat import OpenAIChatCompletionsAdapter
from lmctx.adapters._openai_images import OpenAIImagesAdapter
from lmctx.adapters._openai_responses import OpenAIResponsesAdapter, OpenAIResponsesCompactAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lmctx.context import Context
    from lmctx.plan import AdapterId, LmctxAdapter, RequestPlan
    from lmctx.spec import RunSpec


AdapterKey = tuple[str, str, str | None]


def _adapter_key(adapter_id: AdapterId) -> AdapterKey:
    """Build a stable dictionary key from AdapterId."""
    return adapter_id.provider, adapter_id.endpoint, adapter_id.api_version


def _spec_key(spec: RunSpec) -> AdapterKey:
    """Build a stable dictionary key from RunSpec."""
    return spec.provider, spec.endpoint, spec.api_version


def _adapter_sort_key(key: AdapterKey) -> tuple[str, str, str]:
    """Build a total-order sort key for adapter dictionary entries."""
    provider, endpoint, api_version = key
    return provider, endpoint, api_version or ""


class AutoAdapter:
    """Route plan/ingest calls to a concrete adapter selected from RunSpec.

    The default registry includes all built-in adapters. You can also register
    custom adapters with provider/endpoint/api_version combinations.
    """

    def __init__(self, adapters: Sequence[LmctxAdapter[object]] | None = None) -> None:
        """Initialize routing table with built-in adapters or a custom sequence."""
        self._adapters: dict[AdapterKey, LmctxAdapter[object]] = {}
        initial_adapters = (
            tuple(adapters)
            if adapters is not None
            else (
                OpenAIResponsesAdapter(),
                OpenAIResponsesCompactAdapter(),
                OpenAIChatCompletionsAdapter(),
                OpenAIImagesAdapter(),
                AnthropicMessagesAdapter(),
                GoogleGenAIAdapter(),
                BedrockConverseAdapter(),
            )
        )
        for adapter in initial_adapters:
            self.register(cast("LmctxAdapter[object]", adapter))

    def register(self, adapter: LmctxAdapter[object], *, replace: bool = False) -> None:
        """Register an adapter in the routing table."""
        key = _adapter_key(adapter.id)
        if key in self._adapters and not replace:
            provider, endpoint, api_version = key
            version_text = api_version if api_version is not None else "<none>"
            msg = (
                "Adapter already registered for "
                f"provider={provider!r}, endpoint={endpoint!r}, api_version={version_text!r}. "
                "Pass replace=True to overwrite."
            )
            raise ValueError(msg)
        self._adapters[key] = adapter

    def resolve(self, spec: RunSpec) -> LmctxAdapter[object]:
        """Resolve the concrete adapter for the given RunSpec."""
        exact_key = _spec_key(spec)
        exact_adapter = self._adapters.get(exact_key)
        if exact_adapter is not None:
            return exact_adapter

        fallback_key = (spec.provider, spec.endpoint, None)
        fallback_adapter = self._adapters.get(fallback_key)
        if fallback_adapter is not None:
            return fallback_adapter

        available_targets = ", ".join(
            f"{adapter.id.provider}/{adapter.id.endpoint}"
            + (f"@{adapter.id.api_version}" if adapter.id.api_version is not None else "")
            for _, adapter in sorted(self._adapters.items(), key=lambda item: _adapter_sort_key(item[0]))
        )
        msg = (
            "No adapter registered for "
            f"provider={spec.provider!r}, endpoint={spec.endpoint!r}, api_version={spec.api_version!r}. "
            f"Available: {available_targets or '<none>'}"
        )
        raise ValueError(msg)

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build provider request payload with the adapter selected from RunSpec."""
        adapter = self.resolve(spec)
        return adapter.plan(ctx, spec)

    def ingest(self, ctx: Context, response: object, *, spec: RunSpec) -> Context:
        """Normalize provider response with the adapter selected from RunSpec."""
        adapter = self.resolve(spec)
        return adapter.ingest(ctx, response, spec=spec)

    def available_ids(self) -> tuple[AdapterId, ...]:
        """Return all registered adapter IDs."""
        return tuple(
            adapter.id for _, adapter in sorted(self._adapters.items(), key=lambda item: _adapter_sort_key(item[0]))
        )
