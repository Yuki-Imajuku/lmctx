"""lmctx: Context Kernel for LLM APIs."""

import importlib.metadata as importlib_metadata

from lmctx.adapters import AutoAdapter
from lmctx.blobs import BlobEntry, BlobReference, BlobStore, FileBlobStore, InMemoryBlobStore, PruneReport, put_file
from lmctx.context import Context
from lmctx.errors import (
    BlobIntegrityError,
    BlobNotFoundError,
    ContextError,
    LmctxError,
    PlanValidationError,
)
from lmctx.plan import AdapterCapabilities, AdapterId, CapabilityLevel, ExcludedItem, LmctxAdapter, RequestPlan
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
    "AdapterCapabilities",
    "AdapterId",
    "AutoAdapter",
    "BlobEntry",
    "BlobIntegrityError",
    "BlobNotFoundError",
    "BlobReference",
    "BlobStore",
    "CapabilityLevel",
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
    "PlanValidationError",
    "PruneReport",
    "RequestPlan",
    "Role",
    "RunSpec",
    "ToolSpecification",
    "Usage",
    "put_file",
]
