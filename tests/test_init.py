"""Tests for package-level initialization helpers."""

import importlib.metadata

import pytest

import lmctx


def test_detect_version_uses_importlib_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lmctx.importlib_metadata, "version", lambda name: "1.2.3")
    assert lmctx._detect_version() == "1.2.3"


def test_detect_version_falls_back_without_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_missing(_: str) -> str:
        msg = "metadata missing"
        raise importlib.metadata.PackageNotFoundError(msg)

    monkeypatch.setattr(lmctx.importlib_metadata, "version", _raise_missing)
    assert lmctx._detect_version() == "0.0.0+unknown"
