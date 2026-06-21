from __future__ import annotations

import torch

from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel
from boltzmann_generators.training import TrainConfig, Trainer


def test_trainer_fit_returns_history(
    tiny_flow_model: FlowModel,
    double_well: DoubleWell2D,
    tiny_train_config: TrainConfig,
) -> None:
    torch.manual_seed(0)
    x_data = torch.randn(128, 2)
    trainer = Trainer(tiny_flow_model, double_well, tiny_train_config, device="cpu")
    hist = trainer.fit(x_data)
    assert len(hist) == 5
    assert "loss" in hist[-1]
    assert trainer.history is hist


def test_trainer_pure_kl_z_mode(tiny_flow_model: FlowModel, double_well: DoubleWell2D) -> None:
    torch.manual_seed(0)
    cfg = TrainConfig(n_epochs=3, batch_size=16, w_ml=0.0, w_kl=1.0)
    trainer = Trainer(tiny_flow_model, double_well, cfg, device="cpu")
    hist = trainer.fit(x_data=None)
    assert len(hist) == 3
    assert "kl_z" in hist[-1]


def test_train_bg_deprecation_wrapper(
    tiny_flow_model: FlowModel,
    double_well: DoubleWell2D,
    tiny_train_config: TrainConfig,
) -> None:
    import warnings

    from boltzmann_generators.train import train_bg

    torch.manual_seed(0)
    x_data = torch.randn(128, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        hist = train_bg(tiny_flow_model, double_well, x_data=x_data, config=tiny_train_config)
    assert len(hist) == 5
