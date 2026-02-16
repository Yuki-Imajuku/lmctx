"""BlobStore: protocol for blob storage backends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lmctx.blobs._reference import BlobReference


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def normalize_blob_id(ref_or_id: BlobReference | str) -> str:
    """Normalize a blob selector into a blob ID string."""
    if isinstance(ref_or_id, str):
        return ref_or_id
    return ref_or_id.id


def normalize_cutoff(older_than: datetime | None) -> datetime | None:
    """Normalize prune cutoff into timezone-aware UTC."""
    if older_than is None:
        return None
    if older_than.tzinfo is None:
        return older_than.replace(tzinfo=timezone.utc)
    return older_than.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class BlobEntry:
    """One stored blob and its store-side metadata."""

    ref: BlobReference
    created_at: datetime
    last_accessed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PruneReport:
    """Result of a BlobStore prune operation."""

    deleted: tuple[BlobEntry, ...]
    bytes_freed: int
    examined: int
    remaining: int
    dry_run: bool


def entry_matches_filters(
    entry: BlobEntry,
    *,
    kind: str | None = None,
    media_type: str | None = None,
) -> bool:
    """Return whether an entry matches list_blobs/prune_blobs filters."""
    if kind is not None and entry.ref.kind != kind:
        return False
    return media_type is None or entry.ref.media_type == media_type


def select_prune_candidates(
    entries: Iterable[BlobEntry],
    *,
    older_than: datetime | None = None,
    max_bytes: int | None = None,
) -> tuple[BlobEntry, ...]:
    """Select prune candidates from already-filtered entries.

    Selection order is oldest first (`created_at`, then `ref.id`) to keep
    behavior deterministic and stable across stores.
    """
    if max_bytes is not None and max_bytes < 0:
        msg = "max_bytes must be >= 0."
        raise ValueError(msg)

    cutoff = normalize_cutoff(older_than)
    ordered = tuple(sorted(entries, key=lambda entry: (entry.created_at, entry.ref.id)))
    selected_ids: set[str] = set()

    if cutoff is not None:
        for entry in ordered:
            if entry.created_at < cutoff:
                selected_ids.add(entry.ref.id)

    if max_bytes is not None:
        remaining = [entry for entry in ordered if entry.ref.id not in selected_ids]
        remaining_bytes = sum(entry.ref.size for entry in remaining)
        for entry in remaining:
            if remaining_bytes <= max_bytes:
                break
            selected_ids.add(entry.ref.id)
            remaining_bytes -= entry.ref.size

    return tuple(entry for entry in ordered if entry.ref.id in selected_ids)


@runtime_checkable
class BlobStore(Protocol):
    """Blob storage protocol.

    Implementations must store/retrieve bytes and provide basic lifecycle
    operations for long-running blob management.
    """

    def put_blob(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        kind: str = "file",
    ) -> BlobReference:
        """Store bytes and return a BlobReference."""
        ...

    def get_blob(self, ref: BlobReference) -> bytes:
        """Retrieve bytes by BlobReference."""
        ...

    def has_blob(self, ref: BlobReference) -> bool:
        """Check whether a blob exists."""
        ...

    def delete_blob(self, ref_or_id: BlobReference | str) -> bool:
        """Delete a blob by reference or ID. Return ``True`` when deleted."""
        ...

    def list_blobs(
        self,
        *,
        kind: str | None = None,
        media_type: str | None = None,
    ) -> tuple[BlobEntry, ...]:
        """List stored blobs, optionally filtered by kind/media type."""
        ...

    def prune_blobs(
        self,
        *,
        older_than: datetime | None = None,
        max_bytes: int | None = None,
        kind: str | None = None,
        media_type: str | None = None,
        dry_run: bool = False,
    ) -> PruneReport:
        """Prune blobs and return a report.

        - `older_than`: remove entries created before this timestamp
        - `max_bytes`: keep filtered total bytes at or below this threshold
        - `kind` / `media_type`: apply pruning only to matching entries
        - `dry_run`: compute and report deletions without removing blobs
        """
        ...
