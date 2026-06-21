from __future__ import annotations

import math

import torch

from boltzmann_generators.sampling import effective_sample_size, free_energy_diff


def test_effective_sample_size_uniform_weights() -> None:
    log_w = torch.zeros(100)
    ess = effective_sample_size(log_w)
    assert abs(ess - 100.0) < 1e-4


def test_free_energy_diff_symmetric_regions_is_zero() -> None:
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-0.8, 0.1], [0.9, -0.2]])
    log_w = torch.zeros(x.shape[0])

    def left(t: torch.Tensor) -> torch.Tensor:
        return t[:, 0] < 0

    def right(t: torch.Tensor) -> torch.Tensor:
        return t[:, 0] >= 0

    d_f = free_energy_diff(x, log_w, left, right)
    assert math.isclose(d_f, 0.0, abs_tol=1e-8)
