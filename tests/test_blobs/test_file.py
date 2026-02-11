"""Tests for FileBlobStore."""

import hashlib
from pathlib import Path

import pytest

from lmctx.blobs import BlobReference, BlobStore, FileBlobStore
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


def test_put_and_get(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    data = b"hello world"
    ref = store.put(data, media_type="text/plain", kind="file")
    assert store.get(ref) == data


def test_creates_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "dir"
    store = FileBlobStore(root)
    assert root.is_dir()
    assert store.root == root


def test_sha256_is_correct(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    data = b"test data"
    ref = store.put(data)
    expected = hashlib.sha256(data).hexdigest()
    assert ref.sha256 == expected


def test_get_missing_blob_raises(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = BlobReference(id="nonexistent", sha256="x", media_type=None, kind="file", size=0)
    with pytest.raises(BlobNotFoundError):
        store.get(ref)


def test_integrity_check(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = store.put(b"original")
    # Tamper with the file on disk
    (store.root / ref.id).write_bytes(b"corrupted")
    with pytest.raises(BlobIntegrityError):
        store.get(ref)


def test_contains_positive(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = store.put(b"data")
    assert store.contains(ref) is True


def test_contains_negative(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    ref = BlobReference(id="missing", sha256="x", media_type=None, kind="file", size=0)
    assert store.contains(ref) is False


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
        store.get(ref)


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
    assert store.contains(ref) is False


def test_satisfies_protocol(tmp_path: Path) -> None:
    store = FileBlobStore(tmp_path / "blobs")
    assert isinstance(store, BlobStore)


def test_accepts_str_root(tmp_path: Path) -> None:
    store = FileBlobStore(str(tmp_path / "blobs"))
    ref = store.put(b"data")
    assert store.get(ref) == b"data"
