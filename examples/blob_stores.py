"""BlobStore backends and the put_file helper."""

import tempfile
from pathlib import Path

from lmctx import Context, FileBlobStore, InMemoryBlobStore, put_file

# ---- InMemoryBlobStore ----
# Best for development, testing, and short-lived processes.

memory_store = InMemoryBlobStore()
ref = memory_store.put_blob(b"hello world", media_type="text/plain")
print(f"[InMemory] id={ref.id[:8]}..., sha256={ref.sha256[:16]}...")
print(f"  get_blob() = {memory_store.get_blob(ref)!r}")
print(f"  list_blobs() count = {len(memory_store.list_blobs())}")
print(f"  delete_blob() = {memory_store.delete_blob(ref)}")
print(f"  has_blob() after delete = {memory_store.has_blob(ref)}")

# Refill and prune by max_bytes.
memory_store.put_blob(b"a" * 4, kind="file")
memory_store.put_blob(b"b" * 4, kind="file")
memory_store.put_blob(b"c" * 2, kind="file")
dry_run = memory_store.prune_blobs(max_bytes=5, dry_run=True)
print(f"  prune_blobs(dry_run=True) -> delete={len(dry_run.deleted)}, bytes_freed={dry_run.bytes_freed}")
report = memory_store.prune_blobs(max_bytes=5)
print(f"  prune_blobs() -> delete={len(report.deleted)}, remaining={report.remaining}")

# ---- FileBlobStore ----
# Persists blobs as files under a root directory.

with tempfile.TemporaryDirectory() as tmpdir:
    file_store = FileBlobStore(Path(tmpdir) / "blobs")
    print(f"\n[File] root = {file_store.root}")

    ref = file_store.put_blob(b"persistent data", media_type="application/octet-stream")
    print(f"  id={ref.id[:8]}..., sha256={ref.sha256[:16]}...")
    print(f"  get_blob() = {file_store.get_blob(ref)!r}")
    print(f"  has_blob() = {file_store.has_blob(ref)}")
    print(f"  list_blobs() count = {len(file_store.list_blobs())}")

    # Prune by target capacity.
    file_store.put_blob(b"more data", media_type="application/octet-stream")
    prune_report = file_store.prune_blobs(max_bytes=0)
    print(f"  prune_blobs(max_bytes=0) -> delete={len(prune_report.deleted)}, remaining={prune_report.remaining}")

# ---- put_file helper ----
# Store a file from disk, auto-detecting media_type and kind.

with tempfile.TemporaryDirectory() as tmpdir:
    store = InMemoryBlobStore()

    sample = Path(tmpdir) / "photo.png"
    sample.write_bytes(b"\x89PNG fake image")

    ref = put_file(store, sample)
    print(f"\n[put_file] media_type={ref.media_type}, kind={ref.kind}, size={ref.size}")

    # Override kind when needed
    ref2 = put_file(store, sample, kind="thumbnail")
    print(f"  custom kind={ref2.kind}")

# ---- BlobStore sharing ----
# All Context snapshots from the same chain share one BlobStore.

store = InMemoryBlobStore()
ctx1 = Context(blob_store=store)
ctx2 = ctx1.user("hello")
ctx3 = ctx2.assistant("world")

print(f"\nAll snapshots share the same store: {ctx1.blob_store is ctx2.blob_store is ctx3.blob_store}")

# A blob stored after a snapshot was created is still accessible
ref = store.put_blob(b"late addition")
print(f"Blob added later is visible to all snapshots: {store.has_blob(ref)}")
