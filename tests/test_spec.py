import pytest

from lmctx.spec import Instructions, RunSpec
from lmctx.types import ToolSpecification


def test_runspec_defaults() -> None:
    spec = RunSpec(provider="openai", endpoint="chat.completions", model="gpt-4o")
    assert spec.provider == "openai"
    assert spec.endpoint == "chat.completions"
    assert spec.model == "gpt-4o"
    assert spec.temperature is None
    assert spec.max_output_tokens is None
    assert spec.tools == ()
    assert spec.extra_body == {}
    assert spec.base_url is None


def test_runspec_with_instructions() -> None:
    instr = Instructions(system="You are a helpful assistant.", developer="Be concise.")
    spec = RunSpec(
        provider="anthropic",
        endpoint="messages.create",
        model="claude-sonnet-4-5-20250929",
        instructions=instr,
        max_output_tokens=1024,
        temperature=0.7,
    )
    assert spec.instructions is not None
    assert spec.instructions.system == "You are a helpful assistant."
    assert spec.instructions.developer == "Be concise."
    assert spec.max_output_tokens == 1024
    assert spec.temperature == 0.7


def test_runspec_with_tools() -> None:
    tool = ToolSpecification(
        name="get_weather",
        description="Get the current weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = RunSpec(
        provider="openai",
        endpoint="chat.completions",
        model="gpt-4o",
        tools=(tool,),
        tool_choice="auto",
    )
    assert len(spec.tools) == 1
    assert spec.tools[0].name == "get_weather"
    assert spec.tool_choice == "auto"


def test_runspec_is_frozen() -> None:
    spec = RunSpec(provider="openai", endpoint="chat.completions", model="gpt-4o")
    with pytest.raises(AttributeError):
        spec.model = "gpt-3.5-turbo"  # type: ignore[misc]


def test_instructions_defaults() -> None:
    instr = Instructions()
    assert instr.system is None
    assert instr.developer is None


def test_runspec_extra_body() -> None:
    spec = RunSpec(
        provider="openai",
        endpoint="chat.completions",
        model="gpt-4o",
        extra_body={"stream": True, "logprobs": True},
        extra_headers={"X-Custom": "value"},
    )
    assert spec.extra_body == {"stream": True, "logprobs": True}
    assert spec.extra_headers == {"X-Custom": "value"}


def test_runspec_mappings_are_immutable() -> None:
    spec = RunSpec(
        provider="openai",
        endpoint="responses.create",
        model="gpt-4o",
        extra_body={"reasoning": {"effort": "high"}},
        extra_headers={"X-Test": "1"},
        extra_query={"api-version": "2025-01-01"},
    )
    with pytest.raises(TypeError):
        spec.extra_body["reasoning"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.extra_headers["X-Test"] = "2"  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.extra_query["api-version"] = "2025-01-02"  # type: ignore[index]


def test_runspec_nested_sequences_are_immutable_and_detached() -> None:
    allowed_function_names = ["weather"]
    spec = RunSpec(
        provider="google",
        endpoint="models.generate_content",
        model="gemini-2.0-flash",
        extra_body={
            "tool_config": {
                "function_calling_config": {"allowed_function_names": allowed_function_names},
            }
        },
    )

    assert spec.extra_body == {
        "tool_config": {
            "function_calling_config": {"allowed_function_names": ("weather",)},
        }
    }

    allowed_function_names.append("search")
    assert spec.extra_body == {
        "tool_config": {
            "function_calling_config": {"allowed_function_names": ("weather",)},
        }
    }


def test_runspec_response_modalities_normalized_to_tuple() -> None:
    spec = RunSpec(
        provider="google",
        endpoint="models.generate_content",
        model="gemini-2.0-flash",
        response_modalities=["TEXT", "IMAGE"],  # type: ignore[arg-type]
    )
    assert spec.response_modalities == ("TEXT", "IMAGE")


def test_runspec_accepts_none_for_header_and_query_mappings() -> None:
    spec = RunSpec(
        provider="openai",
        endpoint="responses.create",
        model="gpt-4o",
        extra_headers=None,  # type: ignore[arg-type]
        extra_query=None,  # type: ignore[arg-type]
    )
    assert spec.extra_headers == {}
    assert spec.extra_query == {}


def test_runspec_rejects_non_string_value_in_extra_headers() -> None:
    with pytest.raises(TypeError, match="Expected string value for 'X-Num' in mapping"):
        RunSpec(
            provider="openai",
            endpoint="responses.create",
            model="gpt-4o",
            extra_headers={"X-Num": 1},  # type: ignore[dict-item]
        )


def test_runspec_freezes_tuple_values_inside_extra_body() -> None:
    spec = RunSpec(
        provider="openai",
        endpoint="responses.create",
        model="gpt-4o",
        extra_body={"chain": ("first", {"step": "second"})},
    )
    assert spec.extra_body["chain"] == ("first", {"step": "second"})


def test_instructions_to_from_dict_round_trip() -> None:
    original = Instructions(system="sys", developer="dev")
    serialized = original.to_dict()
    restored = Instructions.from_dict(serialized)
    assert restored == original


def test_runspec_to_from_dict_round_trip() -> None:
    spec = RunSpec(
        provider="openai",
        endpoint="responses.create",
        model="gpt-4o",
        api_version="2025-01-01",
        instructions=Instructions(system="sys", developer="dev"),
        max_output_tokens=512,
        temperature=0.2,
        top_p=0.95,
        seed=7,
        tools=(
            ToolSpecification(
                name="weather",
                description="Fetch weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
        tool_choice={"type": "function", "name": "weather"},
        response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        response_modalities=("TEXT", "IMAGE"),
        extra_body={"reasoning": {"effort": "medium"}},
        extra_headers={"X-Test": "1"},
        extra_query={"api-version": "2025-01-01"},
        base_url="https://api.example.com",
    )
    serialized = spec.to_dict()

    assert serialized["response_modalities"] == ["TEXT", "IMAGE"]

    restored = RunSpec.from_dict(serialized)
    assert restored == spec


def test_runspec_from_dict_rejects_invalid_provider() -> None:
    with pytest.raises(TypeError, match=r"RunSpec\.provider must be a non-empty string"):
        RunSpec.from_dict(
            {
                "provider": None,
                "endpoint": "responses.create",
                "model": "gpt-4o",
            }
        )


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        (
            {"provider": "openai", "endpoint": None, "model": "gpt-4o"},
            r"RunSpec\.endpoint must be a non-empty string",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": None},
            r"RunSpec\.model must be a non-empty string",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "api_version": 1},
            r"RunSpec\.api_version must be a string or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "instructions": "bad"},
            r"RunSpec\.instructions must be a mapping or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "response_schema": 1},
            r"RunSpec\.response_schema must be a mapping or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "extra_body": []},
            r"RunSpec\.extra_body must be a mapping",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "extra_headers": []},
            r"RunSpec\.extra_headers must be a mapping",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "extra_query": []},
            r"RunSpec\.extra_query must be a mapping",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "max_output_tokens": "1"},
            r"RunSpec\.max_output_tokens must be an int or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "temperature": "0.1"},
            r"RunSpec\.temperature must be a float or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "top_p": "0.9"},
            r"RunSpec\.top_p must be a float or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "seed": "1"},
            r"RunSpec\.seed must be an int or None",
        ),
        (
            {"provider": "openai", "endpoint": "responses.create", "model": "gpt-4o", "base_url": 123},
            r"RunSpec\.base_url must be a string or None",
        ),
    ],
)
def test_runspec_from_dict_rejects_invalid_fields(payload: dict[str, object], error_pattern: str) -> None:
    with pytest.raises(TypeError, match=error_pattern):
        RunSpec.from_dict(payload)


@pytest.mark.parametrize(
    ("tools_value", "error_pattern"),
    [
        ("not-a-sequence", r"RunSpec\.tools must be a sequence"),
        ([1], r"RunSpec\.tools\[0\] must be a mapping"),
        (
            [{"name": "", "description": "d", "input_schema": {}}],
            r"RunSpec\.tools\[0\]\.name must be a non-empty string",
        ),
        (
            [{"name": "weather", "description": 1, "input_schema": {}}],
            r"RunSpec\.tools\[0\]\.description must be a string",
        ),
        (
            [{"name": "weather", "description": "d", "input_schema": "bad"}],
            r"RunSpec\.tools\[0\]\.input_schema must be a mapping",
        ),
    ],
)
def test_runspec_from_dict_rejects_invalid_tools(tools_value: object, error_pattern: str) -> None:
    with pytest.raises(TypeError, match=error_pattern):
        RunSpec.from_dict(
            {
                "provider": "openai",
                "endpoint": "responses.create",
                "model": "gpt-4o",
                "tools": tools_value,
            }
        )


def test_runspec_from_dict_rejects_invalid_response_modalities() -> None:
    with pytest.raises(TypeError, match=r"RunSpec\.response_modalities\[0\] must be a string"):
        RunSpec.from_dict(
            {
                "provider": "openai",
                "endpoint": "responses.create",
                "model": "gpt-4o",
                "response_modalities": [1],
            }
        )


def test_runspec_from_dict_rejects_non_string_extra_header_value() -> None:
    with pytest.raises(TypeError, match=r"RunSpec\.extra_headers\['X-Num'\] must be a string"):
        RunSpec.from_dict(
            {
                "provider": "openai",
                "endpoint": "responses.create",
                "model": "gpt-4o",
                "extra_headers": {"X-Num": 1},
            }
        )


def test_instructions_from_dict_rejects_invalid_fields() -> None:
    with pytest.raises(TypeError, match=r"Instructions\.system must be a string or None"):
        Instructions.from_dict({"system": 1})
    with pytest.raises(TypeError, match=r"Instructions\.developer must be a string or None"):
        Instructions.from_dict({"developer": 1})
