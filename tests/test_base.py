from __future__ import annotations

import torch

from boltzmann_generators.base import BaseDensityModel, EnergyModel
from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import (
    CNFFlowModel,
    FlowModel,
    GaussianPrior,
    RealNVP,
    VelocityField,
)


def test_flow_model_is_base_density_model() -> None:
    model = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))
    assert isinstance(model, BaseDensityModel)


def test_cnf_flow_model_is_base_density_model() -> None:
    model = CNFFlowModel(VelocityField(dim=2, hidden_dim=16, num_hidden=2, num_freqs=4))
    assert isinstance(model, BaseDensityModel)


def test_energy_model_grid() -> None:
    energy: EnergyModel = DoubleWell2D()
    gx, gy, u = energy.grid(n=16)
    assert gx.shape == (16, 16)
    assert gy.shape == (16, 16)
    assert u.shape == (16, 16)


def test_cnf_sample_returns_log_prob() -> None:
    torch.manual_seed(0)
    model = CNFFlowModel(VelocityField(dim=2, hidden_dim=16, num_hidden=2, num_freqs=4))
    x, log_q = model.sample(16, device="cpu", n_steps=8, n_hutchinson=1)
    assert x.shape == (16, 2)
    assert log_q.shape == (16,)
    assert torch.isfinite(log_q).all()
