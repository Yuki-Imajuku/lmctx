"""Validate per-module coverage thresholds from a coverage.py JSON report."""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

THRESHOLDS: Final[dict[str, float]] = {
    "src/lmctx/adapters/_google.py": 77.0,
    "src/lmctx/adapters/_anthropic.py": 87.5,
    "src/lmctx/adapters/_openai_responses.py": 88.0,
    "src/lmctx/adapters/_bedrock.py": 92.0,
}


def _load_report(path: Path) -> dict[str, object]:
    """Load coverage JSON report into a plain dict."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = "Coverage report root must be a JSON object."
        raise TypeError(msg)
    return raw


def _files_payload(report: dict[str, object]) -> dict[str, object]:
    """Return the ``files`` payload from a coverage JSON report."""
    files = report.get("files")
    if not isinstance(files, dict):
        msg = "Coverage report is missing a valid 'files' object."
        raise TypeError(msg)
    return files


def _percent_covered(file_entry: object) -> float | None:
    """Extract ``summary.percent_covered`` from one file entry."""
    if not isinstance(file_entry, dict):
        return None
    summary = file_entry.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get("percent_covered")
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalize_coverage_key(path: str) -> str:
    """Normalize coverage file keys across OS path styles."""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    marker = "/src/"
    marker_index = normalized.rfind(marker)
    if marker_index >= 0:
        return normalized[marker_index + 1 :]
    return normalized


def _file_entry_for_module(files: dict[str, object], module_path: str) -> object:
    """Resolve a module entry from coverage ``files`` with path normalization."""
    direct_match = files.get(module_path)
    if direct_match is not None:
        return direct_match

    module_key = _normalize_coverage_key(module_path)
    for file_key, entry in files.items():
        if not isinstance(file_key, str):
            continue
        if _normalize_coverage_key(file_key) == module_key:
            return entry
    return None


def _check_thresholds(report_path: Path) -> list[str]:
    """Return validation failures for configured per-module thresholds."""
    report = _load_report(report_path)
    files = _files_payload(report)
    failures: list[str] = []

    for module_path, min_percent in THRESHOLDS.items():
        percent = _percent_covered(_file_entry_for_module(files, module_path))
        if percent is None:
            failures.append(f"{module_path}: coverage entry not found")
            continue
        if percent < min_percent:
            failures.append(f"{module_path}: {percent:.2f}% < required {min_percent:.2f}%")

    return failures


def main() -> int:
    """Run the module-level coverage threshold check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        default="coverage.json",
        help="Path to coverage.py JSON report (default: coverage.json).",
    )
    args = parser.parse_args()
    report_path = Path(args.report)
    failures = _check_thresholds(report_path)
    if not failures:
        return 0

    for failure in failures:
        sys.stderr.write(f"{failure}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
