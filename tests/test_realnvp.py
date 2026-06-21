from __future__ import annotations

import torch

from boltzmann_generators.flows import RealNVP


def test_realnvp_inverse_is_consistent() -> None:
    torch.manual_seed(0)
    flow = RealNVP(dim=2, num_layers=4, hidden_dim=32, mask="halves")
    z = torch.randn(128, 2)
    x, log_det_fwd = flow.forward(z)
    z_rec, log_det_inv = flow.inverse(x)

    assert torch.allclose(z, z_rec, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        log_det_fwd + log_det_inv,
        torch.zeros_like(log_det_fwd),
        atol=1e-5,
        rtol=1e-5,
    )
