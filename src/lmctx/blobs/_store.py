"""BlobStore: protocol for blob storage backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lmctx.blobs._reference import BlobReference


@runtime_checkable
class BlobStore(Protocol):
    """Minimal protocol for blob storage."""

    def put(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        kind: str = "file",
    ) -> BlobReference:
        """Store bytes and return a BlobReference."""
        ...

    def get(self, ref: BlobReference) -> bytes:
        """Retrieve bytes by BlobReference."""
        ...

    def contains(self, ref: BlobReference) -> bool:
        """Check whether a blob exists."""
        ...
