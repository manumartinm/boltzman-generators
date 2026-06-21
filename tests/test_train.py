from __future__ import annotations

import torch

from boltzmann_generators.energies import DoubleWell2D
from boltzmann_generators.flows import FlowModel
from boltzmann_generators.training import TrainConfig, Trainer


def test_train_bg_smoke_runs(
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
