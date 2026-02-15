"""Typed errors for lmctx."""


class LmctxError(Exception):
    """Base exception for all lmctx errors."""


class BlobNotFoundError(LmctxError):
    """Raised when a BlobReference cannot be resolved in a BlobStore."""

    def __init__(self, blob_id: str) -> None:
        """Initialize with the missing blob's ID."""
        self.blob_id = blob_id
        super().__init__(f"Blob not found: {blob_id}")


class BlobIntegrityError(LmctxError):
    """Raised when blob data does not match its expected SHA-256 digest."""

    def __init__(self, blob_id: str, expected: str, actual: str) -> None:
        """Initialize with the blob ID and mismatched digests."""
        self.blob_id = blob_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"Blob integrity check failed for {blob_id}: expected sha256={expected}, got {actual}")


class ContextError(LmctxError):
    """Raised for invalid operations on Context."""


class PlanValidationError(LmctxError):
    """Raised when RequestPlan validation fails in strict mode."""
