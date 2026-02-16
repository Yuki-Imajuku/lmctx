"""InMemoryBlobStore: dict-based blob storage for development and testing."""

import hashlib
import uuid
from datetime import datetime

from lmctx.blobs._reference import BlobReference
from lmctx.blobs._store import (
    BlobEntry,
    PruneReport,
    entry_matches_filters,
    normalize_blob_id,
    select_prune_candidates,
    utc_now,
)
from lmctx.errors import BlobIntegrityError, BlobNotFoundError


class InMemoryBlobStore:
    """In-memory blob store for development and testing."""

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._blobs: dict[str, bytes] = {}
        self._entries: dict[str, BlobEntry] = {}

    def put_blob(
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
        self._entries[blob_id] = BlobEntry(
            ref=ref,
            created_at=utc_now(),
            last_accessed_at=None,
        )
        return ref

    def get_blob(self, ref: BlobReference) -> bytes:
        """Retrieve bytes and verify SHA-256 integrity."""
        data = self._blobs.get(ref.id)
        if data is None:
            raise BlobNotFoundError(ref.id)
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            raise BlobIntegrityError(ref.id, ref.sha256, actual)
        entry = self._entries.get(ref.id)
        if entry is not None:
            self._entries[ref.id] = BlobEntry(
                ref=entry.ref,
                created_at=entry.created_at,
                last_accessed_at=utc_now(),
            )
        return data

    def has_blob(self, ref: BlobReference) -> bool:
        """Check whether a blob exists."""
        return ref.id in self._blobs

    def delete_blob(self, ref_or_id: BlobReference | str) -> bool:
        """Delete a blob by reference or ID."""
        blob_id = normalize_blob_id(ref_or_id)
        removed_blob = self._blobs.pop(blob_id, None)
        self._entries.pop(blob_id, None)
        return removed_blob is not None

    def list_blobs(
        self,
        *,
        kind: str | None = None,
        media_type: str | None = None,
    ) -> tuple[BlobEntry, ...]:
        """List stored blobs, optionally filtered by kind/media type."""
        entries = tuple(
            entry for entry in self._entries.values() if entry_matches_filters(entry, kind=kind, media_type=media_type)
        )
        return tuple(sorted(entries, key=lambda entry: (entry.created_at, entry.ref.id)))

    def prune_blobs(
        self,
        *,
        older_than: datetime | None = None,
        max_bytes: int | None = None,
        kind: str | None = None,
        media_type: str | None = None,
        dry_run: bool = False,
    ) -> PruneReport:
        """Prune blobs by age/size thresholds, optionally in dry-run mode."""
        entries = self.list_blobs(kind=kind, media_type=media_type)
        deleted = select_prune_candidates(entries, older_than=older_than, max_bytes=max_bytes)
        bytes_freed = sum(entry.ref.size for entry in deleted)

        if not dry_run:
            for entry in deleted:
                self.delete_blob(entry.ref.id)
            remaining = len(self._entries)
        else:
            remaining = len(self._entries) - len(deleted)

        return PruneReport(
            deleted=deleted,
            bytes_freed=bytes_freed,
            examined=len(entries),
            remaining=remaining,
            dry_run=dry_run,
        )
