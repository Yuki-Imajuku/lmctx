"""Tests for InMemoryBlobStore."""

import hashlib

import pytest

from lmctx.blobs import BlobReference, BlobStore, InMemoryBlobStore
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


def test_put_and_get() -> None:
    store = InMemoryBlobStore()
    data = b"hello world"
    ref = store.put(data, media_type="text/plain", kind="file")
    assert store.get(ref) == data


def test_put_returns_correct_metadata() -> None:
    store = InMemoryBlobStore()
    data = b"\x89PNG"
    ref = store.put(data, media_type="image/png", kind="image")
    assert ref.media_type == "image/png"
    assert ref.kind == "image"
    assert ref.size == len(data)


def test_sha256_is_correct() -> None:
    store = InMemoryBlobStore()
    data = b"test data"
    ref = store.put(data)
    expected = hashlib.sha256(data).hexdigest()
    assert ref.sha256 == expected


def test_get_missing_blob_raises() -> None:
    store = InMemoryBlobStore()
    ref = BlobReference(id="nonexistent", sha256="x", media_type=None, kind="file", size=0)
    with pytest.raises(BlobNotFoundError) as exc_info:
        store.get(ref)
    assert exc_info.value.blob_id == "nonexistent"


def test_integrity_check() -> None:
    store = InMemoryBlobStore()
    ref = store.put(b"original")
    # Tamper with internal storage to simulate corruption
    store._blobs[ref.id] = b"corrupted"
    with pytest.raises(BlobIntegrityError):
        store.get(ref)


def test_contains_positive() -> None:
    store = InMemoryBlobStore()
    ref = store.put(b"data")
    assert store.contains(ref) is True


def test_contains_negative() -> None:
    store = InMemoryBlobStore()
    ref = BlobReference(id="missing", sha256="x", media_type=None, kind="file", size=0)
    assert store.contains(ref) is False


def test_satisfies_protocol() -> None:
    store = InMemoryBlobStore()
    assert isinstance(store, BlobStore)


def test_put_default_kind() -> None:
    store = InMemoryBlobStore()
    ref = store.put(b"data")
    assert ref.kind == "file"


def test_put_default_media_type() -> None:
    store = InMemoryBlobStore()
    ref = store.put(b"data")
    assert ref.media_type is None
