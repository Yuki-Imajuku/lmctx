"""RequestPlan and LmctxAdapter Protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

from lmctx.errors import PlanValidationError
from lmctx.serde import as_str_object_dict, optional_int, optional_string, string_tuple, to_plain_data

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lmctx.context import Context
    from lmctx.spec import RunSpec


ResponseT_contra = TypeVar("ResponseT_contra", contravariant=True)
CapabilityLevel = Literal["yes", "partial", "no"]


def _to_plain_dict(value: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """Normalize a mapping to ``dict[str, Any]`` with plain nested containers."""
    return {str(key): to_plain_data(item) for key, item in value.items()}


def _parse_capability_level(field_name: str, level: object) -> CapabilityLevel:
    """Validate and return a capability level literal."""
    if level == "yes":
        return "yes"
    if level == "partial":
        return "partial"
    if level == "no":
        return "no"
    msg = f"Invalid capability level for {field_name!r}: {level!r}. Expected one of yes/partial/no."
    raise ValueError(msg)


def _freeze_capability_fields(fields: Mapping[str, object]) -> Mapping[str, CapabilityLevel]:
    """Normalize capability levels into an immutable mapping."""
    normalized: dict[str, CapabilityLevel] = {}
    for field_name, level in fields.items():
        normalized[str(field_name)] = _parse_capability_level(field_name, level)
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

    def to_dict(self) -> dict[str, object]:
        """Serialize AdapterId to a plain dictionary."""
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AdapterId:
        """Deserialize AdapterId from a plain dictionary."""
        provider = value.get("provider")
        if not isinstance(provider, str) or not provider:
            msg = "AdapterId.provider must be a non-empty string."
            raise TypeError(msg)
        endpoint = value.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            msg = "AdapterId.endpoint must be a non-empty string."
            raise TypeError(msg)
        api_version = optional_string(value.get("api_version"), field_name="AdapterId.api_version")
        return cls(provider=provider, endpoint=endpoint, api_version=api_version)


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
        return self.fields.get(field_name)

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

    def to_dict(self) -> dict[str, object]:
        """Serialize ExcludedItem to a plain dictionary."""
        return {
            "description": self.description,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExcludedItem:
        """Deserialize ExcludedItem from a plain dictionary."""
        description = value.get("description")
        if not isinstance(description, str) or not description:
            msg = "ExcludedItem.description must be a non-empty string."
            raise TypeError(msg)
        reason = value.get("reason")
        if not isinstance(reason, str):
            msg = "ExcludedItem.reason must be a string."
            raise TypeError(msg)
        return cls(description=description, reason=reason)


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

    def to_dict(self) -> dict[str, object]:
        """Serialize RequestPlan to a plain dictionary."""
        return {
            "request": _to_plain_dict(self.request),
            "included": list(self.included),
            "excluded": [item.to_dict() for item in self.excluded],
            "must_roundtrip": list(self.must_roundtrip),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "token_estimate": self.token_estimate,
            "extra": _to_plain_dict(self.extra),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RequestPlan:
        """Deserialize RequestPlan from a plain dictionary."""
        request = as_str_object_dict(value.get("request"), field_name="RequestPlan.request")
        extra_raw = value.get("extra", {})
        extra = as_str_object_dict(extra_raw, field_name="RequestPlan.extra")

        excluded_raw = value.get("excluded", ())
        if not isinstance(excluded_raw, (list, tuple)):
            msg = "RequestPlan.excluded must be a sequence."
            raise TypeError(msg)
        excluded: list[ExcludedItem] = []
        for index, excluded_value in enumerate(excluded_raw):
            excluded_item_data = as_str_object_dict(
                excluded_value,
                field_name=f"RequestPlan.excluded[{index}]",
            )
            excluded.append(ExcludedItem.from_dict(excluded_item_data))

        return cls(
            request={str(key): to_plain_data(item) for key, item in request.items()},
            included=string_tuple(value.get("included"), field_name="RequestPlan.included"),
            excluded=tuple(excluded),
            must_roundtrip=string_tuple(
                value.get("must_roundtrip"),
                field_name="RequestPlan.must_roundtrip",
            ),
            warnings=string_tuple(value.get("warnings"), field_name="RequestPlan.warnings"),
            errors=string_tuple(value.get("errors"), field_name="RequestPlan.errors"),
            token_estimate=optional_int(value.get("token_estimate"), field_name="RequestPlan.token_estimate"),
            extra={str(key): to_plain_data(item) for key, item in extra.items()},
        )


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
