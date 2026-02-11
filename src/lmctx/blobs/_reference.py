"""BlobReference: immutable handle to stored binary data."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlobReference:
    """Immutable reference to a blob stored in a BlobStore."""

    id: str
    sha256: str
    media_type: str | None
    kind: str
    size: int
