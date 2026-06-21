from __future__ import annotations

import torch

from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel, GaussianPrior, RealNVP
from boltzmann_generators.losses import kl_by_energy, mixed_loss


def _model() -> FlowModel:
    return FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))


def test_kl_by_energy_is_finite() -> None:
    torch.manual_seed(0)
    model = _model()
    energy = DoubleWell2D()
    loss = kl_by_energy(model, energy, n_samples=64, device="cpu", energy_max=20.0)
    assert torch.isfinite(loss)


def test_mixed_loss_respects_weights() -> None:
    torch.manual_seed(0)
    model = _model()
    energy = DoubleWell2D()
    x_data = torch.randn(64, 2)
    loss, parts = mixed_loss(
        model,
        energy,
        x_data,
        n_samples=64,
        device="cpu",
        w_ml=1.0,
        w_kl=0.0,
    )
    assert "kl_x" in parts
    assert "kl_z" not in parts
    assert torch.isfinite(loss)
