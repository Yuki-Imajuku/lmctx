"""Tests for lmctx.errors."""

from lmctx.errors import (
    BlobIntegrityError,
    BlobNotFoundError,
    ContextError,
    LmctxError,
    PlanValidationError,
)


def test_lmctx_error_is_exception() -> None:
    assert issubclass(LmctxError, Exception)


def test_blob_not_found_is_lmctx_error() -> None:
    assert issubclass(BlobNotFoundError, LmctxError)


def test_blob_integrity_is_lmctx_error() -> None:
    assert issubclass(BlobIntegrityError, LmctxError)


def test_context_error_is_lmctx_error() -> None:
    assert issubclass(ContextError, LmctxError)


def test_plan_validation_error_is_lmctx_error() -> None:
    assert issubclass(PlanValidationError, LmctxError)


def test_blob_not_found_carries_blob_id() -> None:
    err = BlobNotFoundError("abc123")
    assert err.blob_id == "abc123"


def test_blob_not_found_message() -> None:
    err = BlobNotFoundError("abc123")
    assert "abc123" in str(err)


def test_blob_integrity_carries_attributes() -> None:
    err = BlobIntegrityError("id1", "expected_hash", "actual_hash")
    assert err.blob_id == "id1"
    assert err.expected == "expected_hash"
    assert err.actual == "actual_hash"


def test_blob_integrity_message() -> None:
    err = BlobIntegrityError("id1", "aaa", "bbb")
    msg = str(err)
    assert "id1" in msg
    assert "aaa" in msg
    assert "bbb" in msg
