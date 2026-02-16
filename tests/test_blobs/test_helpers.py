"""Tests for put_file and helper functions."""

from pathlib import Path

from lmctx.blobs import FileBlobStore, InMemoryBlobStore, put_file


def test_put_file_image(tmp_path: Path) -> None:
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG fake image data")
    store = InMemoryBlobStore()
    ref = put_file(store, img)
    assert ref.media_type == "image/png"
    assert ref.kind == "image"
    assert store.get_blob(ref) == b"\x89PNG fake image data"


def test_put_file_audio(tmp_path: Path) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake mp3")
    store = InMemoryBlobStore()
    ref = put_file(store, audio)
    assert ref.media_type == "audio/mpeg"
    assert ref.kind == "audio"


def test_put_file_unknown_extension(tmp_path: Path) -> None:
    unknown = tmp_path / "data.qzx"
    unknown.write_bytes(b"mystery")
    store = InMemoryBlobStore()
    ref = put_file(store, unknown)
    assert ref.media_type is None
    assert ref.kind == "file"


def test_put_file_explicit_kind(tmp_path: Path) -> None:
    img = tmp_path / "photo.png"
    img.write_bytes(b"data")
    store = InMemoryBlobStore()
    ref = put_file(store, img, kind="thumbnail")
    assert ref.kind == "thumbnail"


def test_put_file_accepts_str_path(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello")
    store = InMemoryBlobStore()
    ref = put_file(store, str(f))
    assert ref.media_type == "text/plain"
    assert store.get_blob(ref) == b"hello"


def test_put_file_with_file_blob_store(tmp_path: Path) -> None:
    src = tmp_path / "input.jpg"
    src.write_bytes(b"\xff\xd8\xff fake jpeg")
    store = FileBlobStore(tmp_path / "blobs")
    ref = put_file(store, src)
    assert ref.media_type == "image/jpeg"
    assert ref.kind == "image"
    assert store.get_blob(ref) == b"\xff\xd8\xff fake jpeg"
