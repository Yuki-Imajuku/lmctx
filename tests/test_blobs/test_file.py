"""Tests for FileBlobStore."""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lmctx.blobs import BlobReference, BlobStore, FileBlobStore
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


def test_put_and_get(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    data = b"hello world"
    ref = store.put_blob(data, media_type="text/plain", kind="file")
    assert store.get_blob(ref) == data


def test_creates_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "dir"
    store = FileBlobStore(root)
    assert root.is_dir()
    assert store.root == root


def test_sha256_is_correct(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    data = b"test data"
    ref = store.put_blob(data)
    expected = hashlib.sha256(data).hexdigest()
    assert ref.sha256 == expected


def test_get_missing_blob_raises(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = BlobReference(id="nonexistent", sha256="x", media_type=None, kind="file", size=0)
    with pytest.raises(BlobNotFoundError):
        store.get_blob(ref)


def test_integrity_check(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = store.put_blob(b"original")
    # Tamper with the file on disk
    (store.root / f"{ref.id}.blob").write_bytes(b"corrupted")
    with pytest.raises(BlobIntegrityError):
        store.get_blob(ref)


def test_contains_positive(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = store.put_blob(b"data")
    assert store.has_blob(ref) is True


def test_contains_negative(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = BlobReference(id="missing", sha256="x", media_type=None, kind="file", size=0)
    assert store.has_blob(ref) is False


def test_get_rejects_path_traversal(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside data")

    ref = BlobReference(
        id="../outside.txt",
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        media_type="text/plain",
        kind="file",
        size=outside.stat().st_size,
    )
    with pytest.raises(BlobNotFoundError):
        store.get_blob(ref)


def test_contains_rejects_path_traversal(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside data")

    ref = BlobReference(
        id="../outside.txt",
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        media_type="text/plain",
        kind="file",
        size=outside.stat().st_size,
    )
    assert store.has_blob(ref) is False


def test_satisfies_protocol(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    assert isinstance(store, BlobStore)


def test_accepts_str_root(tmp_path: Path) -> None:
    store = FileBlobStore(str(tmp_path / "blobs"))
    ref = store.put_blob(b"data")
    assert store.get_blob(ref) == b"data"


def test_delete_existing_blob_returns_true_and_removes_data(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = store.put_blob(b"hello")

    assert store.delete_blob(ref) is True
    assert store.has_blob(ref) is False
    with pytest.raises(BlobNotFoundError):
        store.get_blob(ref)


def test_delete_missing_blob_returns_false(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    assert store.delete_blob("missing") is False


def test_list_returns_entries_sorted_and_filterable(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
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


def test_prune_max_bytes_removes_oldest_entries(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
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
        store._write_entry(store._entries[ref.id])

    report = store.prune_blobs(max_bytes=5)

    assert report.examined == 3
    assert report.bytes_freed == 8
    assert {entry.ref.id for entry in report.deleted} == {first.id, second.id}
    assert store.has_blob(first) is False
    assert store.has_blob(second) is False
    assert store.has_blob(third) is True


def test_prune_older_than_supports_dry_run(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    old_ref = store.put_blob(b"old")
    new_ref = store.put_blob(b"new")

    old_entry = next(entry for entry in store.list_blobs() if entry.ref.id == old_ref.id)
    store._entries[old_ref.id] = old_entry.__class__(
        ref=old_entry.ref,
        created_at=old_entry.created_at - timedelta(days=2),
        last_accessed_at=old_entry.last_accessed_at,
    )
    store._write_entry(store._entries[old_ref.id])

    cutoff = old_entry.created_at - timedelta(days=1)
    dry_run = store.prune_blobs(older_than=cutoff, dry_run=True)
    assert {entry.ref.id for entry in dry_run.deleted} == {old_ref.id}
    assert store.has_blob(old_ref) is True

    report = store.prune_blobs(older_than=cutoff)
    assert {entry.ref.id for entry in report.deleted} == {old_ref.id}
    assert store.has_blob(old_ref) is False
    assert store.has_blob(new_ref) is True


def test_prune_rejects_negative_max_bytes(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    store.put_blob(b"data")
    with pytest.raises(ValueError, match="max_bytes must be >= 0"):
        store.prune_blobs(max_bytes=-1)


def test_list_restores_entries_from_sidecar_metadata(tmp_path: Path) -> None:
    root = tmp_path / "blobs"
    store = FileBlobStore(root)
    ref = store.put_blob(b"persisted", media_type="text/plain", kind="file")

    reopened = FileBlobStore(root)
    entries = reopened.list_blobs()

    assert len(entries) == 1
    assert entries[0].ref.id == ref.id
    assert entries[0].ref.sha256 == ref.sha256
