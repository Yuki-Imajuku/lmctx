"""RunSpec: execution configuration for LLM API calls."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from lmctx.types import ToolSpecification


@dataclass(frozen=True, slots=True)
class Instructions:
    """System and developer instructions for the LLM call."""

    system: str | None = None
    developer: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize Instructions to a plain dictionary."""
        return {
            "system": self.system,
            "developer": self.developer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Instructions":
        """Deserialize Instructions from a plain dictionary."""
        system = value.get("system")
        if system is not None and not isinstance(system, str):
            msg = "Instructions.system must be a string or None."
            raise TypeError(msg)
        developer = value.get("developer")
        if developer is not None and not isinstance(developer, str):
            msg = "Instructions.developer must be a string or None."
            raise TypeError(msg)
        return cls(
            system=system,
            developer=developer,
        )


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

    def to_dict(self) -> dict[str, object]:
        """Serialize RunSpec to a plain dictionary."""
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_version": self.api_version,
            "instructions": self.instructions.to_dict() if self.instructions is not None else None,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": _to_plain_data(tool.input_schema),
                }
                for tool in self.tools
            ],
            "tool_choice": _to_plain_data(self.tool_choice),
            "response_schema": _to_plain_data(self.response_schema),
            "response_modalities": list(self.response_modalities),
            "extra_body": _to_plain_data(self.extra_body),
            "extra_headers": _to_plain_data(self.extra_headers),
            "extra_query": _to_plain_data(self.extra_query),
            "base_url": self.base_url,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunSpec":
        """Deserialize RunSpec from a plain dictionary."""
        provider = value.get("provider")
        if not isinstance(provider, str) or not provider:
            msg = "RunSpec.provider must be a non-empty string."
            raise TypeError(msg)

        endpoint = value.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            msg = "RunSpec.endpoint must be a non-empty string."
            raise TypeError(msg)

        model = value.get("model")
        if not isinstance(model, str) or not model:
            msg = "RunSpec.model must be a non-empty string."
            raise TypeError(msg)

        api_version = value.get("api_version")
        if api_version is not None and not isinstance(api_version, str):
            msg = "RunSpec.api_version must be a string or None."
            raise TypeError(msg)

        instructions = _instructions_from_dict_value(value.get("instructions"))
        tools = _tools_from_dict_value(value.get("tools"))
        response_modalities = _string_tuple_from_dict_value(
            value.get("response_modalities"),
            field_name="response_modalities",
        )

        response_schema = value.get("response_schema")
        if response_schema is not None and not isinstance(response_schema, Mapping):
            msg = "RunSpec.response_schema must be a mapping or None."
            raise TypeError(msg)

        extra_body = _mapping_from_dict_value(value.get("extra_body"), field_name="extra_body")
        extra_headers = _string_mapping_from_dict_value(value.get("extra_headers"), field_name="extra_headers")
        extra_query = _string_mapping_from_dict_value(value.get("extra_query"), field_name="extra_query")

        return cls(
            provider=provider,
            endpoint=endpoint,
            model=model,
            api_version=api_version,
            instructions=instructions,
            max_output_tokens=_optional_int_from_dict_value(
                value.get("max_output_tokens"),
                field_name="max_output_tokens",
            ),
            temperature=_optional_float_from_dict_value(value.get("temperature"), field_name="temperature"),
            top_p=_optional_float_from_dict_value(value.get("top_p"), field_name="top_p"),
            seed=_optional_int_from_dict_value(value.get("seed"), field_name="seed"),
            tools=tools,
            tool_choice=_to_plain_data(value.get("tool_choice")),
            response_schema={str(key): _to_plain_data(item) for key, item in response_schema.items()}
            if isinstance(response_schema, Mapping)
            else None,
            response_modalities=response_modalities,
            extra_body=extra_body,
            extra_headers=extra_headers,
            extra_query=extra_query,
            base_url=_optional_string_from_dict_value(value.get("base_url"), field_name="base_url"),
        )


def _to_plain_data(value: object) -> object:
    """Recursively normalize Mapping/tuple containers into plain dict/list values."""
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    return value


def _instructions_from_dict_value(value: object) -> Instructions | None:
    """Deserialize an optional instructions payload."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = "RunSpec.instructions must be a mapping or None."
        raise TypeError(msg)
    return Instructions.from_dict({str(key): item for key, item in value.items()})


def _tools_from_dict_value(value: object) -> tuple[ToolSpecification, ...]:
    """Deserialize the tools list payload."""
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        msg = "RunSpec.tools must be a sequence."
        raise TypeError(msg)

    tools: list[ToolSpecification] = []
    for index, tool_value in enumerate(value):
        if not isinstance(tool_value, Mapping):
            msg = f"RunSpec.tools[{index}] must be a mapping."
            raise TypeError(msg)

        tool_data = {str(key): item for key, item in tool_value.items()}
        name = tool_data.get("name")
        description = tool_data.get("description")
        input_schema = tool_data.get("input_schema")

        if not isinstance(name, str) or not name:
            msg = f"RunSpec.tools[{index}].name must be a non-empty string."
            raise TypeError(msg)
        if not isinstance(description, str):
            msg = f"RunSpec.tools[{index}].description must be a string."
            raise TypeError(msg)
        if not isinstance(input_schema, Mapping):
            msg = f"RunSpec.tools[{index}].input_schema must be a mapping."
            raise TypeError(msg)

        tools.append(
            ToolSpecification(
                name=name,
                description=description,
                input_schema={str(key): _to_plain_data(item) for key, item in input_schema.items()},
            )
        )

    return tuple(tools)


def _string_tuple_from_dict_value(value: object, *, field_name: str) -> tuple[str, ...]:
    """Deserialize an optional sequence of strings into tuple[str, ...]."""
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        msg = f"RunSpec.{field_name} must be a sequence of strings."
        raise TypeError(msg)

    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            msg = f"RunSpec.{field_name}[{index}] must be a string."
            raise TypeError(msg)
        result.append(item)
    return tuple(result)


def _mapping_from_dict_value(value: object, *, field_name: str) -> dict[str, object]:
    """Deserialize an optional mapping payload into dict[str, object]."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"RunSpec.{field_name} must be a mapping."
        raise TypeError(msg)
    return {str(key): _to_plain_data(item) for key, item in value.items()}


def _string_mapping_from_dict_value(value: object, *, field_name: str) -> dict[str, str]:
    """Deserialize an optional string mapping payload into dict[str, str]."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"RunSpec.{field_name} must be a mapping."
        raise TypeError(msg)

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            msg = f"RunSpec.{field_name}[{key!r}] must be a string."
            raise TypeError(msg)
        result[str(key)] = item
    return result


def _optional_string_from_dict_value(value: object, *, field_name: str) -> str | None:
    """Deserialize an optional string field."""
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"RunSpec.{field_name} must be a string or None."
        raise TypeError(msg)
    return value


def _optional_int_from_dict_value(value: object, *, field_name: str) -> int | None:
    """Deserialize an optional integer field."""
    if value is None:
        return None
    if not isinstance(value, int):
        msg = f"RunSpec.{field_name} must be an int or None."
        raise TypeError(msg)
    return value


def _optional_float_from_dict_value(value: object, *, field_name: str) -> float | None:
    """Deserialize an optional float field."""
    if value is None:
        return None
    if not isinstance(value, int | float):
        msg = f"RunSpec.{field_name} must be a float or None."
        raise TypeError(msg)
    return float(value)


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
