from __future__ import annotations

import torch

from boltzmann_generators.analysis import basin_populations, rectangular_region
from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel, GaussianPrior, RealNVP
from boltzmann_generators.sampling import effective_sample_size, sample_with_weights
from boltzmann_generators.training import TrainConfig, Trainer


def test_end_to_end_train_sample_analyze() -> None:
    torch.manual_seed(42)
    energy = DoubleWell2D(a=4.0)
    model = FlowModel(
        GaussianPrior(2),
        RealNVP(dim=2, num_layers=4, hidden_dim=32, mask="halves"),
    )
    cfg = TrainConfig(n_epochs=20, batch_size=64, lr=1e-3, w_ml=0.0, w_kl=1.0, log_every=5)
    Trainer(model, energy, cfg, device="cpu").fit(x_data=None)

    x, log_w, _ = sample_with_weights(model, energy, n=512, device="cpu", chunk=128)
    ess = effective_sample_size(log_w)
    assert ess > 10.0

    left = rectangular_region(x_min=-2.0, x_max=0.0, y_min=-2.0, y_max=2.0)
    right = rectangular_region(x_min=0.0, x_max=2.0, y_min=-2.0, y_max=2.0)
    pops = basin_populations(x, {"left": left, "right": right}, log_w=log_w)
    assert 0.1 < pops["left"] < 0.9
    assert 0.1 < pops["right"] < 0.9


def test_flow_model_sample_log_prob_consistency() -> None:
    torch.manual_seed(0)
    model = FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=4, hidden_dim=32, mask="halves"))
    x, log_q_sample = model.sample(128, device="cpu")
    log_q_direct = model.log_prob(x)
    assert torch.allclose(log_q_sample, log_q_direct, atol=1e-4, rtol=1e-4)
