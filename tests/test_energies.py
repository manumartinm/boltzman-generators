from __future__ import annotations

import torch

from boltzmann_generators.energies import (
    DoubleWell1D,
    DoubleWell2D,
    MullerBrown,
    RamachandranDipeptide,
)


def test_double_well_minima_are_lower_than_barrier() -> None:
    dw = DoubleWell2D(a=4.0, sigma_y=0.5)
    minima = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    barrier = torch.tensor([[0.0, 0.0]])
    assert torch.all(dw(minima) < dw(barrier))


def test_double_well_1d_has_two_minima() -> None:
    dw = DoubleWell1D(a=4.0)
    mins = dw(torch.tensor([[-1.0], [1.0]])).mean()
    barrier = dw(torch.tensor([[0.0]])).item()
    assert mins < barrier


def test_muller_and_dipeptide_grid_shapes() -> None:
    m = MullerBrown()
    r = RamachandranDipeptide()
    _, _, um = m.grid(n=32)
    _, _, ur = r.grid(n=32)
    assert um.shape == (32, 32)
    assert ur.shape == (32, 32)
