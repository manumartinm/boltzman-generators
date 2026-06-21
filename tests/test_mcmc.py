from __future__ import annotations

import torch

from boltzmann_generators.energies import DoubleWell1D
from boltzmann_generators.mcmc import mcmc


def test_mcmc_explores_both_wells() -> None:
    torch.manual_seed(0)
    e = DoubleWell1D(a=4.0)
    x0 = torch.zeros(8, 1)
    samples = mcmc(e, n_steps=200, x0=x0, sigma=0.8, n_chains=8)
    frac_negative = (samples[:, 0] < 0).float().mean().item()
    assert 0.2 < frac_negative < 0.8
