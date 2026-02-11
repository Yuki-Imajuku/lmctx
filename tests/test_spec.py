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
