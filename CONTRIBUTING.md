# Contributing

Thanks for your interest in contributing to `boltzmann-generators`.

## Development setup

```bash
uv sync --extra dev --extra notebook
pre-commit install
```

## Running checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python -m build
uv run twine check dist/*
```

## Pull request guidelines

- Open PRs against `main`; CI runs automatically on each PR.
- To release: bump version in `pyproject.toml` + `__init__.py`, update `CHANGELOG.md`, merge — CD publishes on merge.
- Keep changes focused and scoped to a clear problem.
- Add or update tests for behavior changes.
- Update docs when public APIs change.
- Add a `CHANGELOG.md` entry for user-visible changes.
- Keep notebooks reproducible and deterministic where possible.

## Code style

- Python 3.11+.
- Ruff handles linting and import sorting.
- Prefer explicit, typed APIs for public modules.
- Use OOP service classes (`Trainer`, `SamplingEngine`, etc.) for new features.
