from __future__ import annotations

import torch

from boltzmann_generators.flows import PeriodicEmbedding, periodic_inverse


def test_periodic_embedding_roundtrip() -> None:
    angles = torch.tensor([[0.0, 1.0], [3.14, -1.57]])
    embed = PeriodicEmbedding()
    recovered = periodic_inverse(embed(angles))
    assert torch.allclose(angles, recovered, atol=1e-5)


def test_periodic_inverse_raises_on_odd_dim() -> None:
    try:
        periodic_inverse(torch.randn(4, 3))
        raised = False
    except ValueError:
        raised = True
    assert raised
