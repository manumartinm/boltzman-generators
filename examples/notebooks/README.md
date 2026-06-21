# Example Notebooks

Curated, runnable examples for the `boltzmann-generators` library.

## Setup

```bash
uv sync --extra notebook
uv run jupyter lab
```

Open notebooks from this directory.

## Notebooks

| Notebook | Description | Approx. runtime (CPU) |
|---|---|---|
| `01_quickstart_realnvp.ipynb` | End-to-end RealNVP training with `Trainer` | ~30 s |
| `02_reweighting_and_ess.ipynb` | Importance weights, ESS, free-energy difference | ~20 s |
| `03_cnf_flow_matching.ipynb` | CNF flow-matching training and sampling | ~45 s |
| `04_checkpoint_and_resume.ipynb` | Save/load checkpoints with `CheckpointManager` | ~15 s |

All notebooks use reduced epochs and small models for fast execution.
