from __future__ import annotations

import torch

from boltzmann_generators.analysis import AnalysisSuite, basin_populations, rectangular_region
from boltzmann_generators.sampling import normalized_weights


def test_basin_populations_unweighted() -> None:
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-0.5, 0.1], [0.8, -0.2]])
    left = rectangular_region(x_min=-2.0, x_max=0.0, y_min=-2.0, y_max=2.0)
    right = rectangular_region(x_min=0.0, x_max=2.0, y_min=-2.0, y_max=2.0)
    pops = basin_populations(x, {"left": left, "right": right})
    assert abs(pops["left"] - 0.5) < 1e-6
    assert abs(pops["right"] - 0.5) < 1e-6


def test_basin_populations_weighted() -> None:
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    log_w = torch.tensor([0.0, 10.0])
    left = rectangular_region(x_min=-2.0, x_max=0.0, y_min=-2.0, y_max=2.0)
    right = rectangular_region(x_min=0.0, x_max=2.0, y_min=-2.0, y_max=2.0)
    pops = AnalysisSuite().basin_populations(x, {"left": left, "right": right}, log_w=log_w)
    w = normalized_weights(log_w)
    assert abs(pops["left"] - float(w[0])) < 1e-6
    assert abs(pops["right"] - float(w[1])) < 1e-6


def test_analysis_suite_free_energy_diff() -> None:
    x = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    log_w = torch.zeros(2)

    def left(t: torch.Tensor) -> torch.Tensor:
        return t[:, 0] < 0

    def right(t: torch.Tensor) -> torch.Tensor:
        return t[:, 0] >= 0

    suite = AnalysisSuite()
    d_f = suite.free_energy_diff(x, log_w, left, right)
    assert abs(d_f) < 1e-8
