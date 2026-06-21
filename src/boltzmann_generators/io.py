"""Checkpoint I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .services.checkpoint import CheckpointManager

__all__ = ["CheckpointManager", "load_checkpoint", "save_checkpoint"]

_manager = CheckpointManager()


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    _manager.save(model, path, config=config, metrics=metrics)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    return _manager.load(path, model, map_location=map_location)
