"""RequestPlan and LmctxAdapter Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from lmctx.errors import PlanValidationError

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


def _excluded_to_warning(item: ExcludedItem) -> str:
    """Format one excluded item as an unused-parameter warning."""
    return f"unused parameter '{item.description}': {item.reason}"


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

    def unused_parameter_warnings(self) -> tuple[str, ...]:
        """Return warning messages derived from excluded (unused) items."""
        return tuple(_excluded_to_warning(item) for item in self.excluded)

    def warning_messages(self, *, include_unused_parameters: bool = True) -> tuple[str, ...]:
        """Return warnings, optionally including excluded unused-parameter warnings."""
        if not include_unused_parameters:
            return self.warnings
        return (*self.warnings, *self.unused_parameter_warnings())

    def assert_valid(
        self,
        *,
        fail_on_warnings: bool = False,
        fail_on_excluded: bool = False,
    ) -> None:
        """Raise PlanValidationError when the plan violates strict validation rules.

        Validation behavior:
        - `errors` always fail
        - `warnings` fail when `fail_on_warnings=True`
        - `excluded` (unused parameters) fail when `fail_on_excluded=True`
        """
        violations: list[str] = []

        violations.extend(f"error: {message}" for message in self.errors)
        if fail_on_warnings:
            violations.extend(f"warning: {message}" for message in self.warnings)
        if fail_on_excluded:
            violations.extend(self.unused_parameter_warnings())

        if violations:
            joined = "\n- ".join(violations)
            msg = f"RequestPlan validation failed:\n- {joined}"
            raise PlanValidationError(msg)


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
