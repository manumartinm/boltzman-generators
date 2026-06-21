# Release Checklist

Follow this checklist for every PyPI release.

## Pre-release

1. Ensure `main` is green in CI (lint, mypy, tests, coverage >= 75%, build).
2. Update version in:
   - `pyproject.toml`
   - `src/boltzmann_generators/__init__.py`
   - `CITATION.cff` (optional but recommended)
3. Add a new section to `CHANGELOG.md` under `[X.Y.Z]` with categories:
   - Added
   - Changed
   - Fixed
   - Deprecated
   - Removed
   - Security
4. Update README badges/examples if public API changed.

## Release

```bash
git checkout main
git pull
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python -m build
uv run twine check dist/*
git tag vX.Y.Z
git push origin main --tags
```

The `release.yml` workflow validates the changelog entry and publishes to PyPI via trusted publishing (OIDC).

## Post-release

1. Verify install from PyPI:

```bash
pip install boltzmann-generators==X.Y.Z
python -c "import boltzmann_generators; print(boltzmann_generators.__version__)"
```

2. Create a GitHub release from the tag and paste the changelog section.
3. Smoke-run `examples/notebooks/01_quickstart_realnvp.ipynb`.
