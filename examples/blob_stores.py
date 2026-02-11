"""BlobStore backends and the put_file helper."""

import tempfile
from pathlib import Path

from lmctx import Context, FileBlobStore, InMemoryBlobStore, put_file

# ---- InMemoryBlobStore ----
# Best for development, testing, and short-lived processes.

memory_store = InMemoryBlobStore()
ref = memory_store.put(b"hello world", media_type="text/plain")
print(f"[InMemory] id={ref.id[:8]}..., sha256={ref.sha256[:16]}...")
print(f"  get() = {memory_store.get(ref)!r}")

# ---- FileBlobStore ----
# Persists blobs as files under a root directory.

with tempfile.TemporaryDirectory() as tmpdir:
    file_store = FileBlobStore(Path(tmpdir) / "blobs")
    print(f"\n[File] root = {file_store.root}")

    ref = file_store.put(b"persistent data", media_type="application/octet-stream")
    print(f"  id={ref.id[:8]}..., sha256={ref.sha256[:16]}...")
    print(f"  get() = {file_store.get(ref)!r}")
    print(f"  contains() = {file_store.contains(ref)}")

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
ref = store.put(b"late addition")
print(f"Blob added later is visible to all snapshots: {store.contains(ref)}")
