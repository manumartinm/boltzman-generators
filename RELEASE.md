# Release Checklist

Releases are automated on merge to `main` when the version is bumped.

## Workflow

| Event | Workflow | Action |
|---|---|---|
| Pull request → `main` | `CI` | lint, mypy, tests, coverage, build |
| Merge to `main` | `CD` | detect version bump → tag + PyPI publish (if bumped) |

## Pre-merge (in your PR)

1. Ensure CI is green on the PR.
2. Bump version in:
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

## What happens on merge

If `pyproject.toml` version differs from the previous commit on `main`:

1. CD validates the changelog entry exists for that version.
2. CD creates git tag `vX.Y.Z`.
3. CD builds and publishes to PyPI.
4. CD creates a GitHub Release.

If the version was **not** bumped, CD exits without publishing.

## Required secrets / configuration

- GitHub secret: `PYPI_TOKEN` (PyPI API token with upload scope)
- GitHub environment: `pypi` (optional protection rules)

Alternatively, replace the publish step with PyPI trusted publishing (OIDC) and remove `PYPI_TOKEN`.

## Post-release

1. Verify install from PyPI:

```bash
pip install boltzmann-generators==X.Y.Z
python -c "import boltzmann_generators; print(boltzmann_generators.__version__)"
```

2. Smoke-run `examples/notebooks/01_quickstart_realnvp.ipynb`.
