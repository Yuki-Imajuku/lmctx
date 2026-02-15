"""RequestPlan and LmctxAdapter Protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, runtime_checkable

from lmctx.errors import PlanValidationError

if TYPE_CHECKING:
    from lmctx.context import Context
    from lmctx.spec import RunSpec


ResponseT_contra = TypeVar("ResponseT_contra", contravariant=True)
CapabilityLevel = Literal["yes", "partial", "no"]
_CAPABILITY_LEVELS = frozenset({"yes", "partial", "no"})


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


def _freeze_capability_fields(fields: Mapping[str, object]) -> Mapping[str, CapabilityLevel]:
    """Normalize capability levels into an immutable mapping."""
    normalized: dict[str, CapabilityLevel] = {}
    for field_name, level in fields.items():
        if not isinstance(level, str) or level not in _CAPABILITY_LEVELS:
            msg = f"Invalid capability level for {field_name!r}: {level!r}. Expected one of yes/partial/no."
            raise ValueError(msg)
        normalized[str(field_name)] = cast("CapabilityLevel", level)
    return MappingProxyType(normalized)


def _freeze_notes(notes: Mapping[str, object]) -> Mapping[str, str]:
    """Normalize capability notes into an immutable mapping."""
    normalized: dict[str, str] = {}
    for field_name, note in notes.items():
        if not isinstance(note, str):
            msg = f"Capability note for {field_name!r} must be a string."
            raise TypeError(msg)
        normalized[str(field_name)] = note
    return MappingProxyType(normalized)


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
class AdapterCapabilities:
    """Runtime capability metadata for one adapter."""

    id: AdapterId
    fields: Mapping[str, CapabilityLevel]
    notes: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Freeze and validate capability payloads."""
        frozen_fields = _freeze_capability_fields(self.fields)
        frozen_notes = _freeze_notes(self.notes)
        unknown_note_fields = tuple(field_name for field_name in frozen_notes if field_name not in frozen_fields)
        if unknown_note_fields:
            joined = ", ".join(repr(name) for name in unknown_note_fields)
            msg = f"Capability notes contain unknown field keys: {joined}"
            raise ValueError(msg)

        object.__setattr__(self, "fields", frozen_fields)
        object.__setattr__(self, "notes", frozen_notes)

    def level(self, field_name: str) -> CapabilityLevel | None:
        """Return support level for a capability field."""
        value = self.fields.get(field_name)
        return cast("CapabilityLevel | None", value)

    def is_supported(self, field_name: str, *, allow_partial: bool = True) -> bool:
        """Return whether a field is supported by this adapter."""
        level = self.level(field_name)
        if level is None:
            return False
        if level == "yes":
            return True
        return allow_partial and level == "partial"


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

    def capabilities(self) -> AdapterCapabilities:
        """Return structured adapter capability metadata."""
        ...

    def plan(self, ctx: Context, spec: RunSpec) -> RequestPlan:
        """Build a provider-specific request from Context and RunSpec."""
        ...

    def ingest(self, ctx: Context, response: ResponseT_contra, *, spec: RunSpec) -> Context:
        """Normalize a provider response and append it to the Context."""
        ...
