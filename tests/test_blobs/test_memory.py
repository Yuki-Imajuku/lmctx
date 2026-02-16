"""Tests for InMemoryBlobStore."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from lmctx.blobs import BlobReference, BlobStore, InMemoryBlobStore
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


def test_put_and_get() -> None:
    store = InMemoryBlobStore()
    data = b"hello world"
    ref = store.put_blob(data, media_type="text/plain", kind="file")
    assert store.get_blob(ref) == data


def test_put_returns_correct_metadata() -> None:
    store = InMemoryBlobStore()
    data = b"\x89PNG"
    ref = store.put_blob(data, media_type="image/png", kind="image")
    assert ref.media_type == "image/png"
    assert ref.kind == "image"
    assert ref.size == len(data)


def test_sha256_is_correct() -> None:
    store = InMemoryBlobStore()
    data = b"test data"
    ref = store.put_blob(data)
    expected = hashlib.sha256(data).hexdigest()
    assert ref.sha256 == expected


def test_get_missing_blob_raises() -> None:
    store = InMemoryBlobStore()
    ref = BlobReference(id="nonexistent", sha256="x", media_type=None, kind="file", size=0)
    with pytest.raises(BlobNotFoundError) as exc_info:
        store.get_blob(ref)
    assert exc_info.value.blob_id == "nonexistent"


def test_integrity_check() -> None:
    store = InMemoryBlobStore()
    ref = store.put_blob(b"original")
    # Tamper with internal storage to simulate corruption
    store._blobs[ref.id] = b"corrupted"
    with pytest.raises(BlobIntegrityError):
        store.get_blob(ref)


def test_contains_positive() -> None:
    store = InMemoryBlobStore()
    ref = store.put_blob(b"data")
    assert store.has_blob(ref) is True


def test_contains_negative() -> None:
    store = InMemoryBlobStore()
    ref = BlobReference(id="missing", sha256="x", media_type=None, kind="file", size=0)
    assert store.has_blob(ref) is False


def test_satisfies_protocol() -> None:
    store = InMemoryBlobStore()
    assert isinstance(store, BlobStore)


def test_put_default_kind() -> None:
    store = InMemoryBlobStore()
    ref = store.put_blob(b"data")
    assert ref.kind == "file"


def test_put_default_media_type() -> None:
    store = InMemoryBlobStore()
    ref = store.put_blob(b"data")
    assert ref.media_type is None


def test_delete_existing_blob_returns_true_and_removes_data() -> None:
    store = InMemoryBlobStore()
    ref = store.put_blob(b"hello")

    assert store.delete_blob(ref) is True
    assert store.has_blob(ref) is False
    with pytest.raises(BlobNotFoundError):
        store.get_blob(ref)


def test_delete_missing_blob_returns_false() -> None:
    store = InMemoryBlobStore()
    assert store.delete_blob("missing") is False


def test_list_returns_entries_sorted_and_filterable() -> None:
    store = InMemoryBlobStore()
    text_ref = store.put_blob(b"text", media_type="text/plain", kind="file")
    image_ref = store.put_blob(b"img", media_type="image/png", kind="image")

    entries = store.list_blobs()
    assert [entry.ref.id for entry in entries] == [text_ref.id, image_ref.id]

    image_entries = store.list_blobs(kind="image")
    assert len(image_entries) == 1
    assert image_entries[0].ref.id == image_ref.id

    text_entries = store.list_blobs(media_type="text/plain")
    assert len(text_entries) == 1
    assert text_entries[0].ref.id == text_ref.id


def test_prune_max_bytes_removes_oldest_entries() -> None:
    store = InMemoryBlobStore()
    first = store.put_blob(b"1" * 4, kind="file")
    second = store.put_blob(b"2" * 4, kind="file")
    third = store.put_blob(b"3" * 2, kind="file")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ordered_refs = (first, second, third)
    for index, ref in enumerate(ordered_refs):
        original = store._entries[ref.id]
        store._entries[ref.id] = original.__class__(
            ref=original.ref,
            created_at=base + timedelta(seconds=index),
            last_accessed_at=original.last_accessed_at,
        )

    report = store.prune_blobs(max_bytes=5)

    assert report.examined == 3
    assert report.bytes_freed == 8
    assert {entry.ref.id for entry in report.deleted} == {first.id, second.id}
    assert store.has_blob(first) is False
    assert store.has_blob(second) is False
    assert store.has_blob(third) is True


def test_prune_older_than_supports_dry_run() -> None:
    store = InMemoryBlobStore()
    old_ref = store.put_blob(b"old")
    new_ref = store.put_blob(b"new")

    old_entry = next(entry for entry in store.list_blobs() if entry.ref.id == old_ref.id)
    store._entries[old_ref.id] = old_entry.__class__(
        ref=old_entry.ref,
        created_at=old_entry.created_at - timedelta(days=2),
        last_accessed_at=old_entry.last_accessed_at,
    )

    cutoff = old_entry.created_at - timedelta(days=1)
    dry_run = store.prune_blobs(older_than=cutoff, dry_run=True)
    assert {entry.ref.id for entry in dry_run.deleted} == {old_ref.id}
    assert store.has_blob(old_ref) is True

    report = store.prune_blobs(older_than=cutoff)
    assert {entry.ref.id for entry in report.deleted} == {old_ref.id}
    assert store.has_blob(old_ref) is False
    assert store.has_blob(new_ref) is True


def test_prune_rejects_negative_max_bytes() -> None:
    store = InMemoryBlobStore()
    store.put_blob(b"data")
    with pytest.raises(ValueError, match="max_bytes must be >= 0"):
        store.prune_blobs(max_bytes=-1)
