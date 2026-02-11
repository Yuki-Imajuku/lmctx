"""lmctx: Context Kernel for LLM APIs."""

import importlib.metadata as importlib_metadata

from lmctx.adapters import AutoAdapter
from lmctx.blobs import BlobReference, BlobStore, FileBlobStore, InMemoryBlobStore, put_file
from lmctx.context import Context
from lmctx.errors import (
    BlobIntegrityError,
    BlobNotFoundError,
    ContextError,
    LmctxError,
)
from lmctx.plan import AdapterId, ExcludedItem, LmctxAdapter, RequestPlan
from lmctx.spec import Instructions, RunSpec
from lmctx.types import Cursor, Message, Part, Role, ToolSpecification, Usage


def _detect_version() -> str:
    """Return installed package version or a local fallback when metadata is unavailable."""
    try:
        return importlib_metadata.version("lmctx")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _detect_version()

__all__ = [
    "AdapterId",
    "AutoAdapter",
    "BlobIntegrityError",
    "BlobNotFoundError",
    "BlobReference",
    "BlobStore",
    "Context",
    "ContextError",
    "Cursor",
    "ExcludedItem",
    "FileBlobStore",
    "InMemoryBlobStore",
    "Instructions",
    "LmctxAdapter",
    "LmctxError",
    "Message",
    "Part",
    "RequestPlan",
    "Role",
    "RunSpec",
    "ToolSpecification",
    "Usage",
    "put_file",
]
