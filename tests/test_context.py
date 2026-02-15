"""Tests for lmctx.context."""

import base64

import pytest

from lmctx.blobs import BlobReference, InMemoryBlobStore
from lmctx.context import Context
from lmctx.errors import BlobIntegrityError, BlobNotFoundError, ContextError
from lmctx.types import Cursor, Message, Part, Usage


def _minimal_context_payload() -> dict[str, object]:
    return {
        "messages": [
            {
                "role": "user",
                "parts": [{"type": "text", "text": "hello"}],
            }
        ]
    }


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


def test_context_to_from_dict_round_trip_with_explicit_blob_store() -> None:
    store = InMemoryBlobStore()
    blob_ref = store.put(b"image-data", media_type="image/png", kind="image")

    message = Message(
        role="assistant",
        parts=(
            Part(type="text", text="result"),
            Part(type="image", blob=blob_ref),
        ),
        id="m1",
        provider="openai",
        turn_id="t1",
    )
    usage = Usage(input_tokens=10, output_tokens=4, total_tokens=14, provider_usage={"cached_tokens": 2})
    original = Context(
        messages=(message,),
        cursor=Cursor(last_response_id="resp_1"),
        usage_log=(usage,),
        blob_store=store,
    )

    serialized = original.to_dict()
    restored = Context.from_dict(serialized, blob_store=store)

    assert restored == original
    assert restored.blob_store is store


def test_context_to_from_dict_round_trip_with_blob_payloads() -> None:
    store = InMemoryBlobStore()
    blob_ref = store.put(b"binary-data", media_type="application/octet-stream", kind="file")
    original = Context(blob_store=store).append(Message(role="user", parts=(Part(type="file", blob=blob_ref),)))

    serialized = original.to_dict(include_blob_payloads=True)
    assert "blob_payloads" in serialized

    restored = Context.from_dict(serialized)
    restored_ref = restored.messages[0].parts[0].blob
    assert restored_ref is not None
    assert restored.blob_store.get(restored_ref) == b"binary-data"


def test_context_to_dict_raises_when_blob_payload_missing_from_store() -> None:
    missing_ref = BlobReference(
        id="missing",
        sha256="0" * 64,
        media_type="application/octet-stream",
        kind="file",
        size=10,
    )
    ctx = Context().append(Message(role="user", parts=(Part(type="file", blob=missing_ref),)))

    with pytest.raises(ContextError, match="Cannot serialize blob payload"):
        ctx.to_dict(include_blob_payloads=True)


def test_context_from_dict_rejects_blob_store_and_blob_payloads_together() -> None:
    payload = {"messages": [], "blob_payloads": []}
    with pytest.raises(ValueError, match="cannot accept both blob_store and blob_payloads"):
        Context.from_dict(payload, blob_store=InMemoryBlobStore())


def test_context_from_dict_rejects_blob_payload_sha_mismatch() -> None:
    payload = {
        "messages": [],
        "blob_payloads": [
            {
                "ref": {
                    "id": "blob-1",
                    "sha256": "0" * 64,
                    "media_type": "application/octet-stream",
                    "kind": "file",
                    "size": 3,
                },
                "data_b64": base64.b64encode(b"abc").decode("ascii"),
            }
        ],
    }

    with pytest.raises(ValueError, match="sha256 mismatch"):
        Context.from_dict(payload)


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        ({"messages": 1}, r"Context\.messages must be a sequence"),
        (
            {"messages": [1]},
            r"Context\.messages\[0\] must be a mapping",
        ),
        (
            {"messages": [{"role": 1, "parts": []}]},
            r"Context\.messages\[0\]\.role must be one of",
        ),
        (
            {"messages": [{"role": "user", "parts": 1}]},
            r"Context\.messages\[0\]\.parts must be a sequence",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": 1}]}]},
            r"Context\.messages\[0\]\.parts\[0\]\.type must be a non-empty string",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "json", "json": 1}]}]},
            r"Context\.messages\[0\]\.parts\[0\]\.json must be a mapping or None",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "tool_call", "tool_args": 1}]}]},
            r"Context\.messages\[0\]\.parts\[0\]\.tool_args must be a mapping or None",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "text", "provider_raw": 1}]}]},
            r"Context\.messages\[0\]\.parts\[0\]\.provider_raw must be a mapping or None",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "text"}], "id": 1}]},
            r"Context\.messages\[0\]\.id must be a string or None",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "text"}], "timestamp": "not-a-date"}]},
            r"Context\.messages\[0\]\.timestamp must be an ISO-8601 datetime string",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "text"}], "timestamp": 1}]},
            r"Context\.messages\[0\]\.timestamp must be a string or None",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "text"}]}], "usage_log": 1},
            r"Context\.usage_log must be a sequence",
        ),
        (
            {"messages": [{"role": "user", "parts": [{"type": "text"}]}], "usage_log": [{"provider_usage": 1}]},
            r"Context\.usage_log\[0\]\.provider_usage must be a mapping or None",
        ),
        (
            {"messages": [], "blob_payloads": "bad"},
            r"Context\.blob_payloads must be a sequence",
        ),
    ],
)
def test_context_from_dict_rejects_invalid_fields(payload: dict[str, object], error_pattern: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error_pattern):
        Context.from_dict(payload)


def test_context_from_dict_normalizes_naive_timestamp_to_utc() -> None:
    payload = _minimal_context_payload()
    payload["messages"][0]["timestamp"] = "2026-02-01T12:34:56"  # type: ignore[index]
    restored = Context.from_dict(payload)
    assert restored.messages[0].timestamp.tzinfo is not None


def test_context_from_dict_accepts_none_timestamp() -> None:
    payload = _minimal_context_payload()
    payload["messages"][0]["timestamp"] = None  # type: ignore[index]
    restored = Context.from_dict(payload)
    assert restored.messages[0].timestamp.tzinfo is not None


def test_context_to_dict_serializes_all_part_optional_fields() -> None:
    ctx = Context().append(
        Message(
            role="assistant",
            parts=(
                Part(
                    type="tool_result",
                    json={"ok": True},
                    tool_call_id="c1",
                    tool_name="weather",
                    tool_args={"city": "Tokyo"},
                    tool_output=("done", ["nested"]),
                    provider_raw={"raw": {"x": 1}},
                ),
            ),
        )
    )
    payload = ctx.to_dict()
    part_payload = payload["messages"][0]["parts"][0]  # type: ignore[index]
    assert part_payload["json"] == {"ok": True}
    assert part_payload["tool_call_id"] == "c1"
    assert part_payload["tool_name"] == "weather"
    assert part_payload["tool_args"] == {"city": "Tokyo"}
    assert part_payload["tool_output"] == ["done", ["nested"]]
    assert part_payload["provider_raw"] == {"raw": {"x": 1}}


def test_context_to_dict_deduplicates_blob_payloads_by_id() -> None:
    store = InMemoryBlobStore()
    blob_ref = store.put(b"same")
    ctx = Context(blob_store=store).append(
        Message(role="user", parts=(Part(type="file", blob=blob_ref), Part(type="file", blob=blob_ref)))
    )

    payload = ctx.to_dict(include_blob_payloads=True)
    blob_payloads = payload["blob_payloads"]
    assert len(blob_payloads) == 1  # type: ignore[arg-type]


def test_context_from_dict_rejects_invalid_blob_payload_data_b64() -> None:
    base_ref = {
        "id": "blob-1",
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "media_type": "application/octet-stream",
        "kind": "file",
        "size": 3,
    }

    with pytest.raises(TypeError, match=r"Context\.blob_payloads\[0\]\.data_b64 must be a base64 string"):
        Context.from_dict({"messages": [], "blob_payloads": [{"ref": base_ref, "data_b64": 1}]})

    with pytest.raises(ValueError, match=r"Context\.blob_payloads\[0\]\.data_b64 must be valid base64"):
        Context.from_dict({"messages": [], "blob_payloads": [{"ref": base_ref, "data_b64": "%%%"}]})


@pytest.mark.parametrize(
    ("ref_payload", "error_pattern"),
    [
        (
            {"id": 1, "sha256": "x", "media_type": None, "kind": "file", "size": 1},
            r"Context\.blob_payloads\[0\]\.ref\.id must be a non-empty string",
        ),
        (
            {"id": "blob-1", "sha256": "", "media_type": None, "kind": "file", "size": 1},
            r"Context\.blob_payloads\[0\]\.ref\.sha256 must be a non-empty string",
        ),
        (
            {"id": "blob-1", "sha256": "x", "media_type": None, "kind": 1, "size": 1},
            r"Context\.blob_payloads\[0\]\.ref\.kind must be a non-empty string",
        ),
        (
            {"id": "blob-1", "sha256": "x", "media_type": None, "kind": "file", "size": "1"},
            r"Context\.blob_payloads\[0\]\.ref\.size must be an int",
        ),
    ],
)
def test_context_from_dict_rejects_invalid_blob_ref_fields(ref_payload: dict[str, object], error_pattern: str) -> None:
    with pytest.raises(TypeError, match=error_pattern):
        Context.from_dict(
            {
                "messages": [],
                "blob_payloads": [{"ref": ref_payload, "data_b64": base64.b64encode(b"a").decode("ascii")}],
            }
        )


def test_serialized_blob_store_methods_and_integrity_checks() -> None:
    payload = {
        "messages": [],
        "blob_payloads": [
            {
                "ref": {
                    "id": "blob-1",
                    "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    "media_type": "application/octet-stream",
                    "kind": "file",
                    "size": 3,
                },
                "data_b64": base64.b64encode(b"abc").decode("ascii"),
            }
        ],
    }
    ctx = Context.from_dict(payload)
    store = ctx.blob_store

    new_ref = store.put(b"xyz")
    assert store.contains(new_ref) is True

    missing_ref = BlobReference(
        id="missing",
        sha256="0" * 64,
        media_type=None,
        kind="file",
        size=1,
    )
    with pytest.raises(BlobNotFoundError):
        store.get(missing_ref)

    ref = BlobReference(
        id="blob-1",
        sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        media_type="application/octet-stream",
        kind="file",
        size=3,
    )
    store._blobs["blob-1"] = b"tampered"  # type: ignore[attr-defined]
    with pytest.raises(BlobIntegrityError):
        store.get(ref)
