from __future__ import annotations

import torch

from boltzmann_generators.flows import CNFFlowModel, VelocityField


def test_cnf_log_prob_returns_finite_values() -> None:
    torch.manual_seed(0)
    model = CNFFlowModel(VelocityField(dim=2, hidden_dim=16, num_hidden=2, num_freqs=4))
    x = torch.randn(32, 2)
    lp = model.log_prob(x, n_steps=8, n_hutchinson=1)
    assert lp.shape == (32,)
    assert torch.isfinite(lp).all()


def test_cfm_loss_backpropagates() -> None:
    torch.manual_seed(0)
    model = CNFFlowModel(VelocityField(dim=2, hidden_dim=16, num_hidden=2, num_freqs=4))
    x = torch.randn(32, 2)
    loss = model.cfm_loss(x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
