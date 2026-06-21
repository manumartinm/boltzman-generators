# Migration Guide (0.1.x → 0.2.x)

Version 0.2.0 introduces a class-based OOP API while keeping backward-compatible function wrappers.

## Training

**Before (0.1.x):**

```python
from boltzmann_generators.train import TrainConfig, train_bg

history = train_bg(model, energy, x_data=x_data, config=config, device=device)
```

**After (0.2.x, recommended):**

```python
from boltzmann_generators.training import TrainConfig, Trainer

trainer = Trainer(model, energy, config, device=device)
history = trainer.fit(x_data)
```

`train_bg()` still works but emits a `DeprecationWarning`.

## Sampling and analysis

**Before:**

```python
from boltzmann_generators.sampling import sample_with_weights, effective_sample_size
from boltzmann_generators.analysis import basin_populations
```

**After (recommended service API):**

```python
from boltzmann_generators.services import SamplingEngine, AnalysisSuite

engine = SamplingEngine(model, energy)
x, log_w, log_q = engine.sample_with_weights(1024, device="cpu")
ess = engine.effective_sample_size(log_w)

suite = AnalysisSuite(engine)
pops = suite.basin_populations(x, {"left": left_fn, "right": right_fn}, log_w=log_w)
```

Function imports in `boltzmann_generators.sampling` and `boltzmann_generators.analysis` remain available.

## Checkpoints

**Before:**

```python
from boltzmann_generators.io import save_checkpoint, load_checkpoint
```

**After (recommended):**

```python
from boltzmann_generators.services import CheckpointManager

mgr = CheckpointManager()
mgr.save(model, "model.pt", config={"lr": 1e-3})
model, meta = mgr.load("model.pt", model)
```

## Type interfaces

- Energies should subclass or conform to `boltzmann_generators.base.EnergyModel`.
- Density models should subclass `boltzmann_generators.base.BaseDensityModel`.
- Loss functions accept any object matching the `DensityModel` protocol.

## Example notebooks

See runnable examples in [`examples/notebooks/`](examples/notebooks/).
