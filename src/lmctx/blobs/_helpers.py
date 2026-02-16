"""Helper functions for blob operations."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmctx.blobs._reference import BlobReference
    from lmctx.blobs._store import BlobStore


def _guess_kind(media_type: str | None) -> str:
    if media_type is None:
        return "file"
    major = media_type.split("/", 1)[0]
    if major in ("image", "audio", "video"):
        return major
    return "file"


def put_file(store: BlobStore, path: str | Path, *, kind: str | None = None) -> BlobReference:
    """Store a file in a BlobStore, guessing media_type from the extension.

    Read the file at the given path, detect its media type from the
    file extension, and store the contents in the given BlobStore.
    """
    path = Path(path)
    media_type, _ = mimetypes.guess_type(str(path))
    if kind is None:
        kind = _guess_kind(media_type)
    data = path.read_bytes()
    return store.put_blob(data, media_type=media_type, kind=kind)
