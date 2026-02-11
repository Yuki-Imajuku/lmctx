"""InMemoryBlobStore: dict-based blob storage for development and testing."""

import hashlib
import uuid

from lmctx.blobs._reference import BlobReference
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


class InMemoryBlobStore:
    """In-memory blob store for development and testing."""

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._blobs: dict[str, bytes] = {}

    def put(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        kind: str = "file",
    ) -> BlobReference:
        """Store bytes and return a BlobReference."""
        digest = hashlib.sha256(data).hexdigest()
        blob_id = uuid.uuid4().hex
        ref = BlobReference(
            id=blob_id,
            sha256=digest,
            media_type=media_type,
            kind=kind,
            size=len(data),
        )
        self._blobs[blob_id] = data
        return ref

    def get(self, ref: BlobReference) -> bytes:
        """Retrieve bytes and verify SHA-256 integrity."""
        data = self._blobs.get(ref.id)
        if data is None:
            raise BlobNotFoundError(ref.id)
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            raise BlobIntegrityError(ref.id, ref.sha256, actual)
        return data

    def contains(self, ref: BlobReference) -> bool:
        """Check whether a blob exists."""
        return ref.id in self._blobs
