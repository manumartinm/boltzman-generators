from __future__ import annotations

import pytest

from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel, GaussianPrior, RealNVP
from boltzmann_generators.training import TrainConfig


@pytest.fixture
def tiny_flow_model() -> FlowModel:
    return FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))


@pytest.fixture
def double_well() -> DoubleWell2D:
    return DoubleWell2D()


@pytest.fixture
def tiny_train_config() -> TrainConfig:
    return TrainConfig(n_epochs=5, batch_size=32, lr=1e-3, w_ml=1.0, w_kl=0.2, log_every=1)
