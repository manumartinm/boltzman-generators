# boltzmann-generators

[![CI](https://github.com/manumartinm/boltzmann-generators/actions/workflows/ci.yml/badge.svg)](https://github.com/manumartinm/boltzmann-generators/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/boltzmann-generators)](https://pypi.org/project/boltzmann-generators/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyTorch tools for Boltzmann Generators: invertible flows, benchmark energy functions, BG losses, and reweighting diagnostics (ESS and free-energy differences). The project is designed for reproducible experiments on toy molecular systems and as a base for transferable BG research.

## References

1. Noé, Olsson, Köhler, Wu, *Boltzmann Generators: Sampling Equilibrium States of Many-Body Systems with Deep Learning* (Science, 2019): <https://arxiv.org/abs/1812.01729>
2. Klein, Noé, *Transferable Boltzmann Generators* (NeurIPS, 2024): <https://openreview.net/forum?id=AYq6GxxrrY>
3. Original reference implementation: <https://github.com/noegroup/paper_boltzmann_generators>

## Installation

```bash
pip install boltzmann-generators
```

For notebooks and plots:

```bash
pip install "boltzmann-generators[notebook]"
```

For development:

```bash
pip install "boltzmann-generators[dev]"
```

For higher-accuracy CNF ODE integration:

```bash
pip install "boltzmann-generators[ode]"
```

## Quickstart

```python
import torch
from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel, GaussianPrior, RealNVP
from boltzmann_generators.training import TrainConfig, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

energy = DoubleWell2D()
prior = GaussianPrior(dim=2).to(device)
flow = RealNVP(dim=2, num_layers=8, hidden_dim=64, mask="halves").to(device)
model = FlowModel(prior, flow).to(device)

config = TrainConfig(n_epochs=200, batch_size=256, lr=1e-3, w_ml=0.0, w_kl=1.0)
trainer = Trainer(model, energy, config, device=device)
history = trainer.fit(x_data=None)
print("Final loss:", history[-1]["loss"])
```

## Project layout

```text
src/boltzmann_generators/
  base/                # EnergyModel and BaseDensityModel ABCs
  training/            # Trainer and loss strategy classes
  services/            # SamplingEngine, AnalysisSuite, CheckpointManager
  energies/            # double-well, Muller-Brown, Ramachandran-like energy
  flows/               # base flow API, RealNVP, CNF
  losses.py            # KL_x, KL_z, mixed loss (function wrappers)
  sampling.py          # weighted samples, ESS, free-energy difference
  mcmc.py              # Metropolis baseline sampler
  train.py             # deprecated train_bg wrapper
  analysis.py          # basin and population utilities

examples/notebooks/    # curated runnable examples
notebooks/             # end-to-end reproductions (02-07)
```

## Reproduced results

| Notebook | System | Metric | Value |
|---|---|---|---|
| 02 | 4-mode Gaussian mixture | Final NLL | 2.23 nats (target entropy: 2.13) |
| 03 | Double-well 2D (4 kT barrier) | ESS / DeltaF | 98% / +0.004 kT (target 0) |
| 04 | Muller-Brown (3 basins) | ESS / populations | 96% / target [80.37, 7.74, 11.89]%, BG [80.23, 7.71, 12.05]% |
| 05 | Double-well paper comparison | ESS | 92% |
| 06 | Gaussian mixture (CFM) | Visual mode coverage | All 4 modes recovered |
| 07 | Synthetic Ramachandran | Visual + basin populations | Reweighted BG close to ground truth |

## Documentation

- Example notebooks: [`examples/notebooks/`](examples/notebooks/)
- Migration guide (0.1 → 0.2): [`MIGRATION.md`](MIGRATION.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Release process: [`RELEASE.md`](RELEASE.md)
- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Research notebooks: `notebooks/`

If you need the original Noé group reference repository locally, clone it into:

```bash
git clone --depth 1 https://github.com/noegroup/paper_boltzmann_generators.git references/paper_boltzmann_generators
```
