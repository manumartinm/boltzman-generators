from __future__ import annotations

import torch

from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel, GaussianPrior, RealNVP
from boltzmann_generators.io import CheckpointManager, load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: object) -> None:
    torch.manual_seed(0)
    model = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))
    x = torch.randn(8, 2)
    before = model.log_prob(x).detach().clone()

    path = tmp_path / "ckpt.pt"  # type: ignore[operator]
    save_checkpoint(model, path, config={"lr": 1e-3}, metrics={"loss": 1.0})
    model2 = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))
    loaded, meta = load_checkpoint(path, model2)
    after = loaded.log_prob(x).detach()
    assert torch.allclose(before, after)
    assert meta["config"]["lr"] == 1e-3
    assert meta["metrics"]["loss"] == 1.0


def test_checkpoint_manager_class(tmp_path: object) -> None:
    model = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))
    mgr = CheckpointManager()
    path = tmp_path / "m.pt"  # type: ignore[operator]
    mgr.save(model, path, metrics={"epoch": 10})
    model2 = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))
    _, meta = mgr.load(path, model2)
    assert meta["metrics"]["epoch"] == 10


def test_checkpoint_manager_with_energy_model(tmp_path: object) -> None:
    model = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))
    energy = DoubleWell2D()
    _ = energy(torch.zeros(1, 2))
    path = tmp_path / "e.pt"  # type: ignore[operator]
    CheckpointManager().save(model, path)
    assert path.exists()  # type: ignore[union-attr]
