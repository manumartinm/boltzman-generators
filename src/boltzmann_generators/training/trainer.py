"""Trainer orchestration for Boltzmann Generator models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..base.density import BaseDensityModel
from ..base.energy import EnergyModel
from .loss_strategies import EnergyFn, LossStrategy, MixedLossStrategy

TrainCallback = Callable[[int, dict[str, float]], None]


@dataclass(slots=True)
class TrainConfig:
    n_epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-3
    w_ml: float = 1.0
    w_kl: float = 1.0
    energy_max: float | None = None
    grad_clip: float | None = None
    log_every: int = 50


class Trainer:
    """Class-based training loop for flow and CNF density models."""

    def __init__(
        self,
        model: BaseDensityModel,
        energy: EnergyModel | EnergyFn,
        config: TrainConfig,
        *,
        device: torch.device | str = "cpu",
        loss_strategy: LossStrategy | None = None,
    ) -> None:
        self.model = model
        self.energy = energy
        self.config = config
        self.device = device
        self.loss_strategy = loss_strategy or MixedLossStrategy(
            w_ml=config.w_ml,
            w_kl=config.w_kl,
            energy_max=config.energy_max,
        )
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        x_data: Tensor | None = None,
        *,
        callback: TrainCallback | None = None,
    ) -> list[dict[str, float]]:
        """Run training for ``config.n_epochs`` and return per-epoch metrics."""
        self.model = self.model.to(self.device)
        if x_data is not None:
            x_data = x_data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.history = []

        for epoch in range(self.config.n_epochs):
            batch = None
            if x_data is not None and self.config.w_ml > 0:
                idx = torch.randint(
                    0, x_data.shape[0], (self.config.batch_size,), device=self.device
                )
                batch = x_data[idx]

            loss, parts = self.loss_strategy.compute(
                self.model,
                self.energy,
                batch,
                n_samples=self.config.batch_size,
                device=self.device,
            )
            optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            optimizer.step()

            record = {"epoch": float(epoch), "loss": float(loss.item()), **parts}
            self.history.append(record)
            if callback is not None and (
                epoch % self.config.log_every == 0 or epoch == self.config.n_epochs - 1
            ):
                callback(epoch, record)

        return self.history
