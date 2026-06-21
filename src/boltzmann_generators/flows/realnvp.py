"""RealNVP: stack of affine coupling layers with alternating masks."""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Flow
from .coupling import AffineCoupling


def alternating_mask(dim: int, k: int, device: torch.device | str = "cpu") -> Tensor:
    """Even/odd binary mask. k even → 1s at even idx; k odd → flipped."""
    idx = torch.arange(dim, device=device)
    base = (idx % 2 == 0).float()
    return base if k % 2 == 0 else 1 - base


def halves_mask(dim: int, k: int, device: torch.device | str = "cpu") -> Tensor:
    """First/second half mask. Better for very low dim (e.g. 2D)."""
    half = dim // 2
    mask = torch.zeros(dim, device=device)
    mask[:half] = 1.0
    return mask if k % 2 == 0 else 1 - mask


class RealNVP(Flow):
    """Stack of AffineCoupling layers with alternating masks.

    For dim=2 use mask='halves' (alternating leaves only 1 dim conditioning,
    which is fine but halves is conceptually clearer). For larger dim use
    'alternating' or pass an explicit list of masks.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int = 8,
        hidden_dim: int = 64,
        num_hidden: int = 2,
        mask: str = "alternating",
        scale_clip: float = 3.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        if mask == "alternating":
            mask_fn = alternating_mask
        elif mask == "halves":
            mask_fn = halves_mask
        else:
            raise ValueError(f"unknown mask scheme: {mask}")
        self.layers = torch.nn.ModuleList(
            [
                AffineCoupling(
                    dim=dim,
                    mask=mask_fn(dim, k),
                    hidden_dim=hidden_dim,
                    num_hidden=num_hidden,
                    scale_clip=scale_clip,
                )
                for k in range(num_layers)
            ]
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        log_det_total = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        h = z
        for layer in self.layers:
            h, log_det = layer.forward(h)
            log_det_total = log_det_total + log_det
        return h, log_det_total

    def inverse(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        h = x
        for layer in reversed(self.layers):
            h, log_det = layer.inverse(h)
            log_det_total = log_det_total + log_det
        return h, log_det_total
