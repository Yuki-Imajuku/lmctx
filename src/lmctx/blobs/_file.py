"""FileBlobStore: file-system-based blob storage."""

import hashlib
import uuid
from pathlib import Path

from lmctx.blobs._reference import BlobReference
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


class FileBlobStore:
    """File-system-based blob store.

    Store each blob as an individual file under a root directory,
    named by its unique ID.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize with a root directory, creating it if needed."""
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the root directory path."""
        return self._root

    def _resolve_path(self, blob_id: str) -> Path | None:
        """Resolve blob path and ensure it stays under the store root."""
        root = self._root.resolve()
        candidate = (self._root / blob_id).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def put(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        kind: str = "file",
    ) -> BlobReference:
        """Store bytes as a file and return a BlobReference."""
        digest = hashlib.sha256(data).hexdigest()
        blob_id = uuid.uuid4().hex
        (self._root / blob_id).write_bytes(data)
        return BlobReference(
            id=blob_id,
            sha256=digest,
            media_type=media_type,
            kind=kind,
            size=len(data),
        )

    def get(self, ref: BlobReference) -> bytes:
        """Read a blob file and verify SHA-256 integrity."""
        path = self._resolve_path(ref.id)
        if path is None or not path.exists():
            raise BlobNotFoundError(ref.id)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            raise BlobIntegrityError(ref.id, ref.sha256, actual)
        return data

    def contains(self, ref: BlobReference) -> bool:
        """Check whether a blob file exists."""
        path = self._resolve_path(ref.id)
        return path is not None and path.exists()
