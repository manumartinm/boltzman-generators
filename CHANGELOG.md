# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-20

### Added

- OOP core abstractions: `EnergyModel`, `BaseDensityModel`, and `DensityModel`.
- Class-based training API via `boltzmann_generators.training.Trainer`.
- Loss strategy classes: `ForwardKLLoss`, `ReverseKLLoss`, `MixedLossStrategy`.
- Service classes: `SamplingEngine`, `AnalysisSuite`, `CheckpointManager`.
- PEP 561 typing marker (`py.typed`) and mypy configuration.
- Coverage gate (`>=75%`) in pytest/CI.
- PyPI release workflow with changelog validation and trusted publishing.
- Example notebooks under `examples/notebooks/`.
- Release checklist in `RELEASE.md` and migration guide in `MIGRATION.md`.

### Changed

- `FlowModel` and `CNFFlowModel` now inherit from `BaseDensityModel`.
- Energy classes now inherit from `EnergyModel`.
- `train_bg()` is deprecated in favor of `Trainer.fit()`.
- Package version bumped to `0.2.0`.

### Fixed

- Removed stale `src/bg` wheel package reference that could break builds.
- README quickstart no longer calls `.to(device)` on non-module energy objects.

## [0.1.0] - 2026-06-16

### Added

- Initial open-source packaging of the Boltzmann Generators PyTorch implementation.
- Split package layout with `boltzmann_generators` as the primary import path.
- OSS project metadata, CI, contribution docs, and testing scaffold.
