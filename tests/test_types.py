"""Tests for lmctx.types."""

from datetime import timezone

import pytest

from lmctx.blobs import BlobReference
from lmctx.types import Cursor, Message, Part, ToolSpecification, Usage


def test_text_part() -> None:
    part = Part(type="text", text="hello")
    assert part.type == "text"
    assert part.text == "hello"
    assert part.blob is None


def test_tool_call_part() -> None:
    part = Part(
        type="tool_call",
        tool_call_id="call_1",
        tool_name="search",
        tool_args={"query": "test"},
    )
    assert part.tool_call_id == "call_1"
    assert part.tool_name == "search"
    assert part.tool_args == {"query": "test"}


def test_tool_result_part() -> None:
    part = Part(
        type="tool_result",
        tool_call_id="call_1",
        tool_output={"results": [1, 2, 3]},
    )
    assert part.tool_call_id == "call_1"
    assert part.tool_output == {"results": (1, 2, 3)}


def test_part_with_blob() -> None:
    ref = BlobReference(id="b1", sha256="abc", media_type="image/png", kind="image", size=100)
    part = Part(type="image", blob=ref)
    assert part.blob is ref


def test_part_is_frozen() -> None:
    part = Part(type="text", text="x")
    with pytest.raises(AttributeError):
        part.text = "y"  # type: ignore[misc]


def test_part_provider_raw() -> None:
    part = Part(type="text", text="hi", provider_raw={"raw_key": "raw_val"})
    assert part.provider_raw == {"raw_key": "raw_val"}


def test_part_tool_args_are_immutable_mapping() -> None:
    part = Part(type="tool_call", tool_args={"city": "Tokyo"})
    assert part.tool_args == {"city": "Tokyo"}
    assert part.tool_args is not None
    with pytest.raises(TypeError):
        part.tool_args["city"] = "London"  # type: ignore[index]


def test_part_nested_sequences_are_immutable_and_detached() -> None:
    cities = ["Tokyo", "London"]
    scores = [1, 2]
    part = Part(
        type="tool_call",
        tool_args={"cities": cities},
        tool_output={"scores": scores},
    )

    assert part.tool_args == {"cities": ("Tokyo", "London")}
    assert part.tool_output == {"scores": (1, 2)}

    cities.append("Paris")
    scores.append(3)

    assert part.tool_args == {"cities": ("Tokyo", "London")}
    assert part.tool_output == {"scores": (1, 2)}


def test_part_tool_output_tuple_is_frozen_and_detached() -> None:
    steps = {"scores": [1, 2]}
    part = Part(type="tool_result", tool_output=("ok", steps))

    assert part.tool_output == ("ok", {"scores": (1, 2)})
    steps["scores"].append(3)
    assert part.tool_output == ("ok", {"scores": (1, 2)})


def test_message_construction() -> None:
    msg = Message(role="user", parts=(Part(type="text", text="hello"),))
    assert msg.role == "user"
    assert len(msg.parts) == 1
    assert msg.parts[0].text == "hello"


def test_message_parts_normalized_to_tuple() -> None:
    msg = Message(role="assistant", parts=[Part(type="text", text="hello")])  # type: ignore[arg-type]
    assert isinstance(msg.parts, tuple)
    assert msg.parts[0].text == "hello"


def test_message_timestamp_is_utc() -> None:
    msg = Message(role="user", parts=())
    assert msg.timestamp.tzinfo == timezone.utc


def test_message_optional_fields_default_none() -> None:
    msg = Message(role="assistant", parts=())
    assert msg.id is None
    assert msg.provider is None
    assert msg.turn_id is None


def test_message_is_frozen() -> None:
    msg = Message(role="user", parts=())
    with pytest.raises(AttributeError):
        msg.role = "assistant"  # type: ignore[misc]


def test_tool_specification_construction() -> None:
    spec = ToolSpecification(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert spec.name == "get_weather"
    assert spec.description == "Get current weather"
    assert "properties" in spec.input_schema


def test_tool_specification_rejects_none_input_schema() -> None:
    with pytest.raises(TypeError, match="input_schema cannot be None"):
        ToolSpecification(
            name="get_weather",
            description="Get current weather",
            input_schema=None,  # type: ignore[arg-type]
        )


def test_cursor_defaults_all_none() -> None:
    cursor = Cursor()
    assert cursor.last_response_id is None
    assert cursor.conversation_id is None
    assert cursor.session_id is None


def test_usage_defaults() -> None:
    usage = Usage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.total_tokens is None
    assert usage.provider_usage == {}


def test_usage_with_values() -> None:
    usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_usage_provider_usage_is_immutable() -> None:
    usage = Usage(provider_usage={"cached_input_tokens": 3})
    assert usage.provider_usage == {"cached_input_tokens": 3}
    with pytest.raises(TypeError):
        usage.provider_usage["cached_input_tokens"] = 4  # type: ignore[index]


def test_usage_provider_usage_values() -> None:
    usage = Usage(provider_usage={"input_tokens": 10, "custom_metric": 5})
    assert usage.provider_usage["input_tokens"] == 10
    assert usage.provider_usage["custom_metric"] == 5


def test_usage_rejects_none_provider_usage() -> None:
    with pytest.raises(TypeError, match="provider_usage cannot be None"):
        Usage(provider_usage=None)  # type: ignore[arg-type]
