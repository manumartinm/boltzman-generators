"""RealNVP affine coupling layer.

Reference: Dinh, Sohl-Dickstein, Bengio (ICLR 2017),
"Density estimation using Real NVP". https://arxiv.org/abs/1605.08803

Mask convention: `mask` is a 0/1 tensor of shape (dim,). Where mask=1 the
component passes through unchanged ("conditioning" dims). Where mask=0 the
component is transformed by an affine map whose scale and shift are functions
of the conditioning dims.

The same network `st` is used in both forward and inverse — only the coupling
is invertible, the network need not be.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import Flow


class _STNet(nn.Module):
    """MLP that outputs concatenated (s, t) of size 2*dim. Last layer init to 0."""

    def __init__(self, dim: int, hidden_dim: int, num_hidden: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        final = nn.Linear(hidden_dim, 2 * dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        out = self.net(h)
        s, t = out.chunk(2, dim=-1)
        return s, t


class AffineCoupling(Flow):
    """Affine coupling layer with stability-clipped log-scale.

    log_scale is `scale_clip * tanh(s_raw)` to keep `exp(log_scale)` bounded
    during early training (paper trick). Final-layer zero init makes the layer
    start as identity (log_scale=0, shift=0).
    """

    def __init__(
        self,
        dim: int,
        mask: Tensor,
        hidden_dim: int = 64,
        num_hidden: int = 2,
        scale_clip: float = 3.0,
    ) -> None:
        super().__init__()
        assert mask.shape == (dim,)
        assert mask.dtype in (torch.float32, torch.float64)
        self.register_buffer("mask", mask)
        self.st = _STNet(dim, hidden_dim, num_hidden)
        self.scale_clip = scale_clip

    def _st(self, h_cond: Tensor) -> tuple[Tensor, Tensor]:
        s_raw, t = self.st(h_cond)
        log_scale = self.scale_clip * torch.tanh(s_raw)
        return log_scale, t

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        h_cond = z * self.mask
        log_scale, t = self._st(h_cond)
        # only transform the (1-mask) components
        log_scale = log_scale * (1 - self.mask)
        t = t * (1 - self.mask)
        x = h_cond + (1 - self.mask) * (z * torch.exp(log_scale) + t)
        log_det = log_scale.sum(dim=-1)
        return x, log_det

    def inverse(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h_cond = x * self.mask
        log_scale, t = self._st(h_cond)
        log_scale = log_scale * (1 - self.mask)
        t = t * (1 - self.mask)
        z = h_cond + (1 - self.mask) * ((x - t) * torch.exp(-log_scale))
        log_det = -log_scale.sum(dim=-1)
        return z, log_det
