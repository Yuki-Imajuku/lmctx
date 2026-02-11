"""Tests for BlobReference."""

import pytest

from lmctx.blobs import BlobReference


def test_is_frozen() -> None:
    ref = BlobReference(id="x", sha256="h", media_type=None, kind="file", size=0)
    with pytest.raises(AttributeError):
        ref.id = "y"  # type: ignore[misc]


def test_fields() -> None:
    ref = BlobReference(
        id="abc",
        sha256="deadbeef",
        media_type="image/png",
        kind="image",
        size=1024,
    )
    assert ref.id == "abc"
    assert ref.sha256 == "deadbeef"
    assert ref.media_type == "image/png"
    assert ref.kind == "image"
    assert ref.size == 1024
