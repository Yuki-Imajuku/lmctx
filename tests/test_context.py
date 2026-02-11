"""Tests for lmctx.context."""

import pytest

from lmctx.blobs import InMemoryBlobStore
from lmctx.context import Context
from lmctx.types import Cursor, Message, Part, Usage


def test_empty_context_length() -> None:
    ctx = Context()
    assert len(ctx) == 0


def test_empty_context_iter() -> None:
    ctx = Context()
    assert list(ctx) == []


def test_empty_context_last_returns_none() -> None:
    ctx = Context()
    assert ctx.last() is None


def test_user_convenience() -> None:
    ctx = Context().user("hi")
    assert len(ctx) == 1
    assert ctx.messages[0].role == "user"
    assert ctx.messages[0].parts[0].text == "hi"


def test_assistant_convenience() -> None:
    ctx = Context().assistant("hello")
    assert len(ctx) == 1
    assert ctx.messages[0].role == "assistant"
    assert ctx.messages[0].parts[0].text == "hello"


def test_user_with_single_part() -> None:
    part = Part(type="image", blob=None)
    ctx = Context().user(part)
    assert len(ctx) == 1
    assert ctx.messages[0].role == "user"
    assert ctx.messages[0].parts == (part,)


def test_user_with_multiple_parts() -> None:
    parts = [Part(type="text", text="See this:"), Part(type="image", blob=None)]
    ctx = Context().user(parts)
    assert len(ctx) == 1
    assert len(ctx.messages[0].parts) == 2
    assert ctx.messages[0].parts[0].text == "See this:"
    assert ctx.messages[0].parts[1].type == "image"


def test_user_rejects_non_part_sequence_item() -> None:
    with pytest.raises(TypeError, match="content sequence items must be Part instances"):
        Context().user(["not-a-part"])  # type: ignore[list-item]


def test_assistant_rejects_non_part_sequence_item() -> None:
    with pytest.raises(TypeError, match="content sequence items must be Part instances"):
        Context().assistant(["not-a-part"])  # type: ignore[list-item]


def test_assistant_with_single_part() -> None:
    part = Part(type="text", text="response")
    ctx = Context().assistant(part)
    assert ctx.messages[0].parts == (part,)


def test_assistant_with_multiple_parts() -> None:
    parts = [
        Part(type="text", text="Here:"),
        Part(type="tool_call", tool_call_id="c1", tool_name="search", tool_args={"q": "x"}),
    ]
    ctx = Context().assistant(parts)
    assert len(ctx.messages[0].parts) == 2
    assert ctx.messages[0].parts[0].text == "Here:"
    assert ctx.messages[0].parts[1].tool_name == "search"


def test_append_returns_new_context() -> None:
    original = Context()
    updated = original.user("hi")
    assert len(original) == 0
    assert len(updated) == 1


def test_chaining() -> None:
    ctx = Context().user("a").assistant("b").user("c")
    assert len(ctx) == 3
    assert ctx.messages[0].parts[0].text == "a"
    assert ctx.messages[1].parts[0].text == "b"
    assert ctx.messages[2].parts[0].text == "c"


def test_append_custom_message() -> None:
    msg = Message(
        role="assistant",
        parts=(
            Part(type="text", text="thinking..."),
            Part(type="tool_call", tool_call_id="c1", tool_name="search", tool_args={"q": "x"}),
        ),
    )
    ctx = Context().append(msg)
    assert len(ctx) == 1
    assert len(ctx.messages[0].parts) == 2


def test_last_no_filter() -> None:
    ctx = Context().user("a").assistant("b")
    last = ctx.last()
    assert last is not None
    assert last.role == "assistant"


def test_last_with_role_filter() -> None:
    ctx = Context().user("a").assistant("b").user("c")
    last_assistant = ctx.last(role="assistant")
    assert last_assistant is not None
    assert last_assistant.parts[0].text == "b"


def test_last_role_not_found() -> None:
    ctx = Context().user("a").user("b")
    assert ctx.last(role="assistant") is None


def test_iteration_order() -> None:
    ctx = Context().user("1").assistant("2").user("3")
    texts = [msg.parts[0].text for msg in ctx]
    assert texts == ["1", "2", "3"]


def test_with_cursor() -> None:
    original = Context()
    cursor = Cursor(last_response_id="resp_123")
    updated = original.with_cursor(cursor)
    assert updated.cursor.last_response_id == "resp_123"
    assert original.cursor.last_response_id is None


def test_with_usage() -> None:
    original = Context()
    usage = Usage(input_tokens=100, output_tokens=50)
    updated = original.with_usage(usage)
    assert len(updated.usage_log) == 1
    assert updated.usage_log[0].input_tokens == 100
    assert len(original.usage_log) == 0


def test_blob_store_shared_across_snapshots() -> None:
    ctx1 = Context()
    ctx2 = ctx1.user("hello")
    assert ctx1.blob_store is ctx2.blob_store


def test_append_message_with_blob_part() -> None:
    store = InMemoryBlobStore()
    ctx = Context(blob_store=store)
    ref = store.put(b"image data", media_type="image/png", kind="image")
    msg = Message(role="user", parts=(Part(type="image", blob=ref),))
    ctx = ctx.append(msg)
    blob = ctx.messages[0].parts[0].blob
    assert blob is not None
    retrieved = store.get(blob)
    assert retrieved == b"image data"


def test_append_inplace_mutates_and_returns_none() -> None:
    ctx = Context()
    msg = Message(role="user", parts=(Part(type="text", text="hello"),))

    returned = ctx.append(msg, inplace=True)

    assert returned is None
    assert len(ctx) == 1
    assert ctx.messages[0].parts[0].text == "hello"


def test_user_and_assistant_inplace() -> None:
    ctx = Context()
    first = ctx.user("u1", inplace=True)
    second = ctx.assistant("a1", inplace=True)

    assert first is None
    assert second is None
    assert len(ctx) == 2
    assert ctx.messages[0].role == "user"
    assert ctx.messages[1].role == "assistant"


def test_with_cursor_and_usage_inplace() -> None:
    ctx = Context()
    cursor = Cursor(last_response_id="resp_1")
    usage = Usage(input_tokens=10, output_tokens=3)

    first = ctx.with_cursor(cursor, inplace=True)
    second = ctx.with_usage(usage, inplace=True)

    assert first is None
    assert second is None
    assert ctx.cursor.last_response_id == "resp_1"
    assert len(ctx.usage_log) == 1
    assert ctx.usage_log[0].input_tokens == 10


def test_extend_appends_multiple_messages() -> None:
    ctx = Context()
    messages = (
        Message(role="user", parts=(Part(type="text", text="u1"),)),
        Message(role="assistant", parts=(Part(type="text", text="a1"),)),
    )

    updated = ctx.extend(messages)

    assert len(updated) == 2
    assert len(ctx) == 0
    assert updated.messages[0].parts[0].text == "u1"
    assert updated.messages[1].parts[0].text == "a1"


def test_extend_inplace_mutates() -> None:
    ctx = Context()
    messages = (Message(role="user", parts=(Part(type="text", text="u1"),)),)

    returned = ctx.extend(messages, inplace=True)

    assert returned is None
    assert len(ctx) == 1
    assert ctx.messages[0].parts[0].text == "u1"


def test_extend_inplace_noop_returns_none() -> None:
    ctx = Context()
    returned = ctx.extend((), inplace=True)
    assert returned is None
    assert len(ctx) == 0


def test_extend_empty_returns_new_context() -> None:
    ctx = Context().user("u1")
    cloned = ctx.extend(())

    assert cloned is not ctx
    assert cloned.messages == ctx.messages
    assert cloned.cursor == ctx.cursor
    assert cloned.usage_log == ctx.usage_log
    assert cloned.blob_store is ctx.blob_store


def test_clear_returns_new_context_by_default() -> None:
    ctx = Context().user("u1").assistant("a1").with_usage(Usage(input_tokens=10))
    cleared = ctx.clear()

    assert len(ctx) == 2
    assert len(ctx.usage_log) == 1
    assert len(cleared) == 0
    assert len(cleared.usage_log) == 0
    assert cleared.cursor.last_response_id is None


def test_clear_inplace_mutates_context() -> None:
    ctx = Context().user("u1").with_cursor(Cursor(last_response_id="resp_1")).with_usage(Usage(input_tokens=10))
    returned = ctx.clear(inplace=True)

    assert returned is None
    assert len(ctx) == 0
    assert len(ctx.usage_log) == 0
    assert ctx.cursor.last_response_id is None


def test_clone_creates_new_snapshot_with_same_data() -> None:
    ctx = Context().user("u1")
    cloned = ctx.clone()

    assert cloned is not ctx
    assert cloned.messages == ctx.messages
    assert cloned.cursor == ctx.cursor
    assert cloned.usage_log == ctx.usage_log
    assert cloned.blob_store is ctx.blob_store


def test_pipe_applies_callable() -> None:
    ctx = Context().user("u1")

    def count_messages(c: Context, role: str) -> int:
        return len([m for m in c if m.role == role])

    assert ctx.pipe(count_messages, "user") == 1


def test_context_is_unhashable() -> None:
    ctx = Context()
    with pytest.raises(TypeError):
        hash(ctx)


def test_nested_payload_is_immutable_across_snapshots() -> None:
    ctx1 = Context().append(
        Message(
            role="assistant",
            parts=(Part(type="tool_call", tool_call_id="call_1", tool_name="weather", tool_args={"city": "Tokyo"}),),
        )
    )
    ctx2 = ctx1.user("next turn")
    tool_args = ctx1.messages[0].parts[0].tool_args
    assert tool_args == {"city": "Tokyo"}
    assert tool_args is not None

    with pytest.raises(TypeError):
        tool_args["city"] = "London"  # type: ignore[index]

    assert ctx1.messages[0].parts[0].tool_args == {"city": "Tokyo"}
    assert ctx2.messages[0].parts[0].tool_args == {"city": "Tokyo"}
