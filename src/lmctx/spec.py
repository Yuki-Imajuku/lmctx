"""RunSpec: execution configuration for LLM API calls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmctx.types import ToolSpecification


@dataclass(frozen=True, slots=True)
class Instructions:
    """System and developer instructions for the LLM call."""

    system: str | None = None
    developer: str | None = None


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Execution configuration: provider, model, generation parameters, and escape hatches.

    RunSpec describes *how* to call the LLM, not *what* to say.
    The conversation content lives in Context; RunSpec carries everything else.
    """

    provider: str
    endpoint: str
    model: str
    api_version: str | None = None
    instructions: Instructions | None = None

    # Generation parameters
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    # Tools and structured output
    tools: tuple[ToolSpecification, ...] = ()
    tool_choice: object | None = None
    response_schema: Mapping[str, object] | None = None
    response_modalities: tuple[str, ...] = ()

    # Provider escape hatches (adapter-defined deep merge into provider payload)
    extra_body: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    extra_headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    extra_query: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    # Client hints (lmctx does not call the LLM, but plan can surface these)
    base_url: str | None = None

    def __post_init__(self) -> None:
        """Normalize containers and freeze mutable mapping payloads."""
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "response_modalities", tuple(self.response_modalities))
        object.__setattr__(self, "response_schema", _freeze_object_mapping(self.response_schema))
        object.__setattr__(self, "extra_body", _freeze_object_mapping(self.extra_body) or MappingProxyType({}))
        object.__setattr__(self, "extra_headers", _freeze_str_mapping(self.extra_headers) or MappingProxyType({}))
        object.__setattr__(self, "extra_query", _freeze_str_mapping(self.extra_query) or MappingProxyType({}))


def _freeze_value(value: object) -> object:
    """Recursively freeze mapping/sequence payload containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_object_mapping(value: Mapping[str, object] | None) -> Mapping[str, object] | None:
    """Freeze ``Mapping[str, object]`` into a mappingproxy."""
    if value is None:
        return None
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_str_mapping(value: Mapping[str, str] | None) -> Mapping[str, str] | None:
    """Freeze ``Mapping[str, str]`` into a mappingproxy."""
    if value is None:
        return None
    normalized: dict[str, str] = {}
    for key, item in value.items():
        frozen_item = _freeze_value(item)
        if not isinstance(frozen_item, str):
            msg = f"Expected string value for {key!r} in mapping."
            raise TypeError(msg)
        normalized[str(key)] = frozen_item
    return MappingProxyType(normalized)
