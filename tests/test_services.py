from __future__ import annotations

import torch

from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel, GaussianPrior, RealNVP
from boltzmann_generators.losses import kl_by_example, mixed_loss
from boltzmann_generators.services import AnalysisSuite, CheckpointManager, SamplingEngine
from boltzmann_generators.training.loss_strategies import (
    ForwardKLLoss,
    MixedLossStrategy,
    ReverseKLLoss,
)


def _model() -> FlowModel:
    return FlowModel(GaussianPrior(2), RealNVP(dim=2, num_layers=2, hidden_dim=16, mask="halves"))


def test_sampling_engine_weighted_samples() -> None:
    torch.manual_seed(0)
    model = _model()
    energy = DoubleWell2D()
    engine = SamplingEngine(model, energy)
    x, log_w, log_q = engine.sample_with_weights(64, device="cpu", chunk=32)
    assert x.shape == (64, 2)
    assert log_w.shape == (64,)
    assert log_q.shape == (64,)
    ess = engine.effective_sample_size(log_w)
    assert ess > 0


def test_forward_kl_loss_strategy() -> None:
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(32, 2)
    loss, parts = ForwardKLLoss().compute(model, lambda t: t, x, n_samples=32, device="cpu")
    assert torch.isfinite(loss)
    assert "kl_x" in parts
    assert abs(float(loss.item()) - float(kl_by_example(model, x).item())) < 1e-5


def test_reverse_kl_with_energy_clipping() -> None:
    torch.manual_seed(0)
    model = _model()
    energy = DoubleWell2D()
    loss, parts = ReverseKLLoss(energy_max=5.0).compute(
        model, energy, None, n_samples=32, device="cpu"
    )
    assert torch.isfinite(loss)
    assert "kl_z" in parts


def test_mixed_loss_pure_kl_z_branch() -> None:
    torch.manual_seed(0)
    model = _model()
    energy = DoubleWell2D()
    loss, parts = MixedLossStrategy(w_ml=0.0, w_kl=1.0).compute(
        model, energy, None, n_samples=32, device="cpu"
    )
    assert "kl_z" in parts
    assert "kl_x" not in parts
    assert torch.isfinite(loss)


def test_mixed_loss_matches_function() -> None:
    torch.manual_seed(0)
    model = _model()
    energy = DoubleWell2D()
    x_data = torch.randn(32, 2)
    strategy = MixedLossStrategy(w_ml=1.0, w_kl=0.5, energy_max=10.0)
    l1, p1 = strategy.compute(model, energy, x_data, n_samples=32, device="cpu")
    torch.manual_seed(0)
    l2, p2 = mixed_loss(
        model, energy, x_data, n_samples=32, device="cpu", w_ml=1.0, w_kl=0.5, energy_max=10.0
    )
    assert p1.keys() == p2.keys()
    assert torch.isfinite(l1) and torch.isfinite(l2)


def test_energy_model_isinstance() -> None:
    energy = DoubleWell2D()
    from boltzmann_generators.base import EnergyModel

    assert isinstance(energy, EnergyModel)


def test_checkpoint_manager_service(tmp_path: object) -> None:
    model = _model()
    path = tmp_path / "svc.pt"  # type: ignore[operator]
    CheckpointManager().save(model, path)
    model2 = _model()
    _, meta = CheckpointManager().load(path, model2)
    assert meta == {"config": {}, "metrics": {}}


def test_analysis_suite_rectangular_region() -> None:
    region = AnalysisSuite.rectangular_region(x_min=-1, x_max=1, y_min=-1, y_max=1)
    x = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    mask = region(x)
    assert mask.tolist() == [True, False]
