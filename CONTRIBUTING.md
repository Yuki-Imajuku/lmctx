# Contributing to lmctx

Thank you for your interest in contributing to lmctx! This document covers the development setup, coding standards, and workflow.

## Prerequisites

- **Python**: 3.10 or later
- **[uv](https://docs.astral.sh/uv/)**: Package and project manager

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Yuki-Imajuku/lmctx.git
cd lmctx

# Install all dependencies (including dev tools)
uv sync --all-extras --dev

# Verify everything works
make check
```

## Development Commands

All commands are available via `make`:

| Command | Description |
|---|---|
| `make check` | Run all checks: lint + format + typecheck + test |
| `make check-fix` | Run all checks with auto-fix applied first |
| `make lint` | Run ruff linter |
| `make lint-fix` | Auto-fix lint issues and reformat |
| `make format` | Run ruff formatter |
| `make typecheck` | Run ty type checker |
| `make test` | Run pytest with global + adapter module coverage gates |

You can also run tools directly:

```bash
uv run ruff check          # lint
uv run ruff format         # format
uv run ty check            # type check
uv run pytest              # test
uv run pytest -k test_name # run specific test
uv run pytest --cov=lmctx --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=90
.venv/bin/python scripts/check_coverage_thresholds.py coverage.json
```

## Versioning and PyPI Release

### How versioning works in this repo

- `pyproject.toml` uses `dynamic = ["version"]`.
- The actual version is derived from Git metadata via `uv-dynamic-versioning` (`[tool.hatch.version] source = "uv-dynamic-versioning"`).
- If no usable Git version metadata is found, the fallback is `0.0.0+unknown`.

In practice, **release version is set by Git tag**, not by editing a hardcoded version string in `pyproject.toml`.

### Setting the next version

Use SemVer-like tags with a `v` prefix (for example: `v0.4.0`).

```bash
# 1) Make sure release commit is on main and checks are green
make check

# 2) Create an annotated release tag
git tag -a v0.4.0 -m "Release v0.4.0"

# 3) Push commit + tag
git push origin main
git push origin v0.4.0
```

To release another iteration, create a new tag (for example `v0.4.1`).
Do not reuse or move published release tags.

### Building and publishing to PyPI

Prerequisites:

- PyPI account with permission for `lmctx`
- PyPI API token (recommended: project-scoped token)

```bash
# 1) Build sdist + wheel
uv build --out-dir dist --clear

# 2) Confirm artifacts
ls -lh dist/

# 3) Publish (token-based auth)
export UV_PUBLISH_TOKEN=pypi-...
uv publish dist/*
```

Optional: publish to TestPyPI first.

```bash
export UV_PUBLISH_TOKEN=pypi-...
uv publish \
  --publish-url https://test.pypi.org/legacy/ \
  --check-url https://test.pypi.org/simple/ \
  dist/*
```

If `uv publish` reports an existing file/version, bump the tag/version and rebuild.

### Release procedure (end-to-end)

Use this checklist for a production release:

1. Ensure the target commit is merged into `main`.
2. Run the full local checks (`make check`).
3. Create and push the release tag (`vX.Y.Z`).
4. Build and publish to PyPI (`uv build` + `uv publish`).
5. Create a GitHub Release for the same tag with release notes.
6. Verify the published version on PyPI (`pip install lmctx==X.Y.Z`).

### Creating a GitHub Release

After pushing the tag and publishing to PyPI, create a GitHub Release for discoverability and changelog tracking.

GitHub UI:

1. Open the repository `Releases` page.
2. Click `Draft a new release`.
3. Select the pushed tag (for example `v0.4.0`).
4. Set the release title (for example `v0.4.0`) and add notes.
5. Publish the release.

GitHub CLI (optional):

```bash
gh release create v0.4.0 \
  --title "v0.4.0" \
  --notes-file RELEASE_NOTES.md
```

## Code Standards

### Type Annotations

- **All public APIs must have type annotations.** This is enforced by ruff (`ANN` rules) and ty.
- Prefer built-in generics over `typing` equivalents (e.g., `list[int]` instead of `List[int]`).
- Minimize `from __future__ import annotations`. Since the project targets Python 3.10+, most modern syntax (e.g., `X | None`) works natively. Only use `__future__` when necessary for forward references or `TYPE_CHECKING` imports.
- Place imports used only for type checking inside `if TYPE_CHECKING:` blocks (enforced by ruff `TC` rules).

### Docstrings

- **All public modules, classes, and functions must have docstrings** (enforced by ruff `D` rules / pydocstyle).
- Use imperative mood for the first line: "Return the last message." not "Returns the last message."
- Test files are exempt from docstring requirements.

### Immutability

- Core data types (`Part`, `Message`, `BlobReference`, `Context`, etc.) are `frozen=True, slots=True` dataclasses.
- `Context` mutations return a new instance by default. `inplace=True` is supported when explicit mutation is desired.
- `BlobStore` is intentionally shared across `Context` snapshots. This is by design.

### Error Handling

- All library exceptions inherit from `LmctxError`.
- Use specific exception types (`BlobNotFoundError`, `BlobIntegrityError`, `ContextError`).

### Naming Conventions

- Prefer full names over abbreviations: `Context` (not `Ctx`), `BlobReference` (not `BlobRef`), `ToolSpecification` (not `ToolSpec`).
- Well-established technical terms are fine as-is: `Part`, `Message`, `Cursor`, `BlobStore`.

### Linting & Formatting

- **Linter**: [ruff](https://docs.astral.sh/ruff/) with an extensive rule set (see `pyproject.toml` for the full list).
- **Formatter**: ruff format (double quotes, 120 char line length).
- **Type checker**: [ty](https://docs.astral.sh/ty/) with `all = "error"`.
- Run `make check` before submitting. CI will reject PRs that fail any check.

## Testing

- Tests live in `tests/` and use [pytest](https://docs.pytest.org/).
- Test file names mirror source files: `src/lmctx/blobs.py` -> `tests/test_blobs.py`.
- All test functions must have `-> None` return type annotations.
- Tests are exempt from: `S101` (assert), `SLF001` (private access), `ARG` (unused args), `D` (docstrings), `PLR2004` (magic values).

### Writing Tests

- Write tests as flat functions by default. Only use classes when shared setup via `self` is necessary.
- Use descriptive names for test functions.
- Use `pytest.mark.parametrize` for testing multiple input/output cases.
- Include edge cases and error conditions.

```python
def test_user_message_appends_to_context() -> None:
    # Arrange
    ctx = Context().user("hello")

    # Act
    result = ctx.last(role="user")

    # Assert
    assert result is not None
    assert result.parts[0].text == "hello"


class TestContextWithMockLLM:
    """Use a class only when shared setup is needed."""

    def __init__(self) -> None:
        self.mock_llm = MockLLM()

    def test_append_message(self) -> None:
        ctx = Context()
        ctx = ctx.user("hello")
        response = self.mock_llm.generate(ctx)
        ctx = ctx.assistant(response)
        assert ctx.last(role="assistant") is not None
```

## Project Layout

```
src/lmctx/               # Library source code
    __init__.py           # Public API re-exports
    context.py            # Context (append-only conversation log)
    types.py              # Part, Message, ToolSpecification, Cursor, Usage, Role
    spec.py               # RunSpec, Instructions
    plan.py               # RequestPlan, ExcludedItem, AdapterId, LmctxAdapter
    blobs/                # Binary blob storage
        _reference.py     # BlobReference
        _store.py         # BlobStore (Protocol)
        _memory.py        # InMemoryBlobStore
        _file.py          # FileBlobStore
        _helpers.py       # put_file()
    adapters/             # Provider adapters
        _auto.py              # AutoAdapter routing
        _util.py              # _to_dict(), _deep_merge()
        _openai_responses.py  # OpenAIResponsesAdapter
        _openai_chat.py       # OpenAIChatCompletionsAdapter
        _openai_images.py     # OpenAIImagesAdapter
        _anthropic.py         # AnthropicMessagesAdapter
        _google.py            # GoogleGenAIAdapter
        _bedrock.py           # BedrockConverseAdapter
    errors.py             # Typed exception hierarchy
    py.typed              # PEP 561 marker
tests/                    # Test suite
    test_blobs/           # Mirrors src/lmctx/blobs/
        test_reference.py
        test_memory.py
        test_file.py
        test_helpers.py
    test_adapters/        # Mirrors src/lmctx/adapters/
        test_openai_responses.py
        test_auto.py
        test_openai_chat.py
        test_openai_images.py
        test_anthropic.py
        test_google.py
        test_bedrock.py
        test_request_payload_serialization.py
    test_context.py
    test_types.py
    test_spec.py
    test_plan.py
    test_errors.py
examples/                 # Runnable usage examples
docs/                     # Documentation
    README.md             # Doc map / reading order
    architecture.md       # High-level design and boundaries
    data-model.md         # Core type contracts
    api-reference.md      # Public API quick reference
    adapters.md           # Adapter capability matrix
    examples.md           # Example execution guide
    logs.md               # Local log regeneration guide
.github/workflows/        # CI configuration
```

## CI

GitHub Actions runs on every push to `main` and on all pull requests:

- **Lint workflow** (`lint.yml`): ruff check, ruff format, ty type check (ubuntu-latest)
- **Test workflow** (`test.yml`): pytest + coverage gates (global and key adapter modules) across Python 3.10 to 3.14 on ubuntu/macos/windows (15 matrix jobs)

All checks must pass before a PR can be merged.

## Pull Request Guidelines

1. **Branch from `main`** and keep your branch up to date.
2. **Run `make check`** locally before pushing.
3. **Write tests** for new functionality. Maintain or improve coverage.
4. **Keep PRs focused.** One feature or fix per PR.
5. **Follow existing patterns.** Look at neighboring code for style guidance.

## Branch Naming Convention

Use descriptive branch names in lowercase kebab-case.

- Format: `<type>/<topic>`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Topic: short, specific, and implementation-oriented
- For Codex created branches, prepend `codex/` (for example `codex/feat/new-feature`).
- For Claude Code created branches, prepend `claude/` (for example `claude/feat/new-feature`).
- For other AI agent created branches, prepend `<agent-name>/` (for example `opencode/feat/new-feature`).

Examples:
- `feat/add-xxx-yyy`
- `fix/handle-edge-case-in-zzz`
- `docs/update-aaa-examples`
- `refactor/clean-up-bbb-module`
- `test/add-tests-for-ccc`
- `chore/update-dependencies`
- `codex/feat/implement-meow-bow-wow`
- `claude/fix/resolve-fizz-buzz-issue`

## AI-Assisted Contributions

AI-assisted contributions are welcome, but responsibility remains with the human contributor who commits the change.

- You must review and understand the proposed code before submitting.
- You are responsible for correctness, tests, security, and licensing/compliance.
- PRs that are effectively unreviewed auto-generated dumps may be closed.
- Submissions should respect maintainer time: keep changes scoped, explain intent, and include validation evidence.

## Community and OSS Workflow Files

This repository includes standard GitHub community workflow files:

- Issue templates: `.github/ISSUE_TEMPLATE/`
- Pull request template: `.github/pull_request_template.md`
- Code of Conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Support guide: `SUPPORT.md`
