"""Checkpoint persistence for trained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class CheckpointManager:
    """Save and load model checkpoints with optional metadata."""

    def save(
        self,
        model: nn.Module,
        path: str | Path,
        *,
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "state_dict": model.state_dict(),
            "config": config or {},
            "metrics": metrics or {},
        }
        torch.save(payload, str(path))

    def load(
        self,
        path: str | Path,
        model: nn.Module,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple[nn.Module, dict[str, Any]]:
        payload = torch.load(str(path), map_location=map_location, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        meta = {
            "config": payload.get("config", {}),
            "metrics": payload.get("metrics", {}),
        }
        return model, meta
