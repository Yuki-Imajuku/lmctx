"""RequestPlan and LmctxAdapter Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from lmctx.context import Context
    from lmctx.spec import RunSpec


ResponseT_contra = TypeVar("ResponseT_contra", contravariant=True)


def _to_plain_data(value: Any) -> Any:
    """Recursively normalize Mapping/tuple containers into plain dict/list values."""
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _to_plain_dict(value: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """Normalize a mapping to ``dict[str, Any]`` with plain nested containers."""
    normalized = _to_plain_data(value)
    if isinstance(normalized, dict):
        return normalized
    return {str(key): _to_plain_data(item) for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class AdapterId:
    """Unique identifier for an adapter: (provider, endpoint, api_version)."""

    provider: str
    endpoint: str
    api_version: str | None = None


@dataclass(frozen=True, slots=True)
class ExcludedItem:
    """An item excluded from the request, with a reason."""

    description: str
    reason: str


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """Explainable output of adapter.plan().

    Contains the provider-specific request payload along with diagnostics:
    what was included, what was excluded (with reasons), must-roundtrip items,
    warnings, and errors detected at plan time.
    """

    request: dict[str, Any]

    included: tuple[str, ...] = ()
    excluded: tuple[ExcludedItem, ...] = ()
    must_roundtrip: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    token_estimate: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure request/extra payloads are plain container types."""
        object.__setattr__(self, "request", _to_plain_dict(self.request))
        object.__setattr__(self, "extra", _to_plain_dict(self.extra))


@runtime_checkable
class LmctxAdapter(Protocol[ResponseT_contra]):
    """Protocol for provider adapters.

    Adapters convert between lmctx types and provider-specific payloads.
    They never call the LLM; they only build requests (plan) and
    normalize responses (ingest).
    """

    id: AdapterId

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build a provider-specific request from Context and RunSpec."""
        ...

    def ingest(self, ctx: Context, response: ResponseT_contra, *, spec: RunSpec) -> Context:
        """Normalize a provider response and append it to the Context."""
        ...
