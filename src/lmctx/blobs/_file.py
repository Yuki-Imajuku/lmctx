"""FileBlobStore: file-system-based blob storage."""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path

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

_PAYLOAD_SUFFIX = ".blob"
_META_SUFFIX = ".meta.json"


class FileBlobStore:
    """File-system-based blob store.

    Store each blob payload as ``<id>.blob`` and metadata as ``<id>.meta.json`` under a root directory.
    """

    def __init__(self, root: str | Path) -> None:
        """Initialize with a root directory, creating it if needed."""
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, BlobEntry] = {}
        self._load_entries()

    @property
    def root(self) -> Path:
        """Return the root directory path."""
        return self._root

    def _resolve_path(self, blob_id: str, *, suffix: str = "") -> Path | None:
        """Resolve blob path and ensure it stays under the store root."""
        root = self._root.resolve()
        candidate = (self._root / f"{blob_id}{suffix}").resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def _payload_path(self, blob_id: str) -> Path | None:
        """Resolve payload path for a blob ID."""
        return self._resolve_path(blob_id, suffix=_PAYLOAD_SUFFIX)

    def _meta_path(self, blob_id: str) -> Path | None:
        """Resolve metadata path for a blob ID."""
        return self._resolve_path(blob_id, suffix=_META_SUFFIX)

    def _entry_to_payload(self, entry: BlobEntry) -> dict[str, object]:
        """Serialize a BlobEntry for sidecar metadata."""
        return {
            "id": entry.ref.id,
            "sha256": entry.ref.sha256,
            "media_type": entry.ref.media_type,
            "kind": entry.ref.kind,
            "size": entry.ref.size,
            "created_at": entry.created_at.isoformat(),
            "last_accessed_at": entry.last_accessed_at.isoformat() if entry.last_accessed_at is not None else None,
        }

    def _as_str_object_dict(self, payload: object) -> dict[str, object] | None:
        """Convert dict-like payload into ``dict[str, object]``."""
        if not isinstance(payload, dict):
            return None
        return {str(key): value for key, value in payload.items()}

    def _parse_iso_timestamp(self, value: object) -> datetime | None:
        """Parse one ISO-8601 timestamp string."""
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _entry_from_payload(self, payload: object, *, blob_id: str) -> BlobEntry | None:
        """Deserialize one metadata sidecar payload."""
        data = self._as_str_object_dict(payload)
        if data is None:
            return None

        payload_id = data.get("id")
        sha256 = data.get("sha256")
        media_type = data.get("media_type")
        kind = data.get("kind")
        size = data.get("size")
        created_at_raw = data.get("created_at")
        last_accessed_raw = data.get("last_accessed_at")

        if (
            not isinstance(payload_id, str)
            or payload_id != blob_id
            or not isinstance(sha256, str)
            or not sha256
            or (media_type is not None and not isinstance(media_type, str))
            or not isinstance(kind, str)
            or not kind
            or not isinstance(size, int)
        ):
            return None

        created_at = self._parse_iso_timestamp(created_at_raw)
        if created_at is None:
            return None
        if last_accessed_raw is None:
            last_accessed_at = None
        else:
            last_accessed_at = self._parse_iso_timestamp(last_accessed_raw)
            if last_accessed_at is None:
                return None

        return BlobEntry(
            ref=BlobReference(
                id=payload_id,
                sha256=sha256,
                media_type=media_type,
                kind=kind,
                size=size,
            ),
            created_at=created_at,
            last_accessed_at=last_accessed_at,
        )

    def _write_entry(self, entry: BlobEntry) -> None:
        """Write metadata sidecar for a blob entry."""
        meta_path = self._meta_path(entry.ref.id)
        if meta_path is None:
            msg = f"Blob ID {entry.ref.id!r} resolves outside store root."
            raise ValueError(msg)
        meta_path.write_text(json.dumps(self._entry_to_payload(entry), ensure_ascii=False), encoding="utf-8")

    def _load_entries(self) -> None:
        """Load metadata sidecars into the in-memory index."""
        for meta_path in self._root.glob(f"*{_META_SUFFIX}"):
            blob_id = meta_path.name[: -len(_META_SUFFIX)]
            payload_path = self._payload_path(blob_id)
            if payload_path is None or not payload_path.exists():
                continue

            try:
                raw = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            entry = self._entry_from_payload(raw, blob_id=blob_id)
            if entry is None:
                continue
            self._entries[blob_id] = entry

    def _touch_entry(self, ref: BlobReference) -> None:
        """Update access time for one entry and persist metadata."""
        current = self._entries.get(ref.id)
        if current is None:
            return
        updated = BlobEntry(
            ref=current.ref,
            created_at=current.created_at,
            last_accessed_at=utc_now(),
        )
        self._entries[ref.id] = updated
        self._write_entry(updated)

    def put_blob(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
        kind: str = "file",
    ) -> BlobReference:
        """Store bytes as a file and return a BlobReference."""
        digest = hashlib.sha256(data).hexdigest()
        blob_id = uuid.uuid4().hex
        payload_path = self._payload_path(blob_id)
        if payload_path is None:
            msg = f"Generated blob ID {blob_id!r} resolves outside store root."
            raise ValueError(msg)
        payload_path.write_bytes(data)

        ref = BlobReference(
            id=blob_id,
            sha256=digest,
            media_type=media_type,
            kind=kind,
            size=len(data),
        )
        entry = BlobEntry(ref=ref, created_at=utc_now(), last_accessed_at=None)
        self._entries[blob_id] = entry
        self._write_entry(entry)
        return ref

    def get_blob(self, ref: BlobReference) -> bytes:
        """Read a blob file and verify SHA-256 integrity."""
        path = self._payload_path(ref.id)
        if path is None or not path.exists():
            raise BlobNotFoundError(ref.id)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != ref.sha256:
            raise BlobIntegrityError(ref.id, ref.sha256, actual)
        self._touch_entry(ref)
        return data

    def has_blob(self, ref: BlobReference) -> bool:
        """Check whether a blob file exists."""
        path = self._payload_path(ref.id)
        return path is not None and path.exists()

    def delete_blob(self, ref_or_id: BlobReference | str) -> bool:
        """Delete a blob payload and metadata sidecar by ref or ID."""
        blob_id = normalize_blob_id(ref_or_id)
        deleted = False

        payload_path = self._payload_path(blob_id)
        if payload_path is not None and payload_path.exists():
            payload_path.unlink()
            deleted = True

        meta_path = self._meta_path(blob_id)
        if meta_path is not None and meta_path.exists():
            meta_path.unlink()
            deleted = True

        self._entries.pop(blob_id, None)
        return deleted

    def list_blobs(
        self,
        *,
        kind: str | None = None,
        media_type: str | None = None,
    ) -> tuple[BlobEntry, ...]:
        """List stored blobs, optionally filtered by kind/media type."""
        stale_ids = [blob_id for blob_id, entry in self._entries.items() if not self.has_blob(entry.ref)]
        for blob_id in stale_ids:
            self._entries.pop(blob_id, None)

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
