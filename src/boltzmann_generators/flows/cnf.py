"""Continuous Normalizing Flow trained by Conditional Flow Matching.

Refs:
- Lipman, Chen, Ben-Hamu, Nickel, Le 2023, "Flow Matching for Generative Modeling", ICLR.
- Chen et al. 2018, "Neural ODEs" — the underlying generative model.

CFM with linear (optimal-transport) paths:
    x_t = (1 - t) z_0 + t x_1,  u_t(x_t | z_0, x_1) = x_1 - z_0

Training is plain MSE regression of the model's velocity onto (x_1 - z_0).
Sampling integrates the learned ODE with Euler (kept dependency-free; for
high-precision use torchdiffeq.odeint).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..base.density import BaseDensityModel
from .base import GaussianPrior

try:
    from torchdiffeq import odeint  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    odeint = None


def _fourier_features(t: Tensor, num_freqs: int = 8) -> Tensor:
    """Encode t ∈ [0,1] with sin/cos at increasing frequencies."""
    # t: (n,)
    freqs = 2 * math.pi * torch.arange(1, num_freqs + 1, device=t.device, dtype=t.dtype)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args), t[:, None]], dim=-1)


class VelocityField(nn.Module):
    """MLP v_theta(x, t) -> R^d. Concatenates Fourier-featured t with x."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        num_hidden: int = 4,
        num_freqs: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs
        t_dim = 2 * num_freqs + 1
        in_dim = dim + t_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_feat = _fourier_features(t, self.num_freqs)
        return self.net(torch.cat([x, t_feat], dim=-1))


class CNFModel(nn.Module):
    """Continuous normalizing flow + standard-normal prior.

    `sample(n, n_steps)` integrates the ODE with Euler from t=0 to t=1.
    """

    def __init__(self, velocity: VelocityField) -> None:
        super().__init__()
        self.v = velocity
        self.dim = velocity.dim

    def sample(self, n: int, device: torch.device | str = "cpu", n_steps: int = 100) -> Tensor:
        x = torch.randn(n, self.dim, device=device)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((n,), k * dt, device=device)
            v = self.v(x, t)
            x = x + dt * v
        return x

    def cfm_loss(self, x1: Tensor) -> Tensor:
        """Conditional flow matching loss with linear OT path."""
        return _cfm_loss(self.v, x1)


def _divergence_hutchinson(
    velocity: VelocityField,
    x: Tensor,
    t: Tensor,
    n_hutchinson: int = 1,
) -> Tensor:
    """Estimate divergence(trace(J_v)) with Hutchinson's estimator."""
    if n_hutchinson < 1:
        raise ValueError("n_hutchinson must be >= 1")

    x_req = x.detach().requires_grad_(True)
    div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(n_hutchinson):
        eps = torch.randint_like(x_req, low=0, high=2, dtype=torch.int64).to(x_req.dtype)
        eps = eps.mul(2).sub(1)  # Rademacher {-1, +1}
        v = velocity(x_req, t)
        dot = (v * eps).sum(dim=-1)
        (jvp,) = torch.autograd.grad(dot.sum(), x_req, create_graph=False, retain_graph=True)
        div = div + (jvp * eps).sum(dim=-1)
    return div / float(n_hutchinson)


def _cfm_loss(velocity: VelocityField, x1: Tensor) -> Tensor:
    n = x1.shape[0]
    t = torch.rand(n, device=x1.device)
    z0 = torch.randn_like(x1)
    x_t = (1 - t)[:, None] * z0 + t[:, None] * x1
    u_target = x1 - z0
    v_pred = velocity(x_t, t)
    return (v_pred - u_target).pow(2).sum(dim=-1).mean()


class CNFFlowModel(BaseDensityModel):
    """CNF wrapper exposing ``sample`` and ``log_prob`` like ``FlowModel``."""

    def __init__(self, velocity: VelocityField, prior: GaussianPrior | None = None) -> None:
        super().__init__()
        self.v = velocity
        self.dim = velocity.dim
        self.prior = prior if prior is not None else GaussianPrior(velocity.dim)

    def cfm_loss(self, x1: Tensor) -> Tensor:
        return _cfm_loss(self.v, x1)

    def _sample_euler(
        self,
        n: int,
        device: torch.device | str = "cpu",
        n_steps: int = 100,
        n_hutchinson: int = 1,
    ) -> tuple[Tensor, Tensor]:
        x = self.prior.sample(n, device=device)
        log_p = self.prior.log_prob(x)
        dt = 1.0 / n_steps

        for k in range(n_steps):
            t = torch.full((n,), k * dt, device=device, dtype=x.dtype)
            div = _divergence_hutchinson(self.v, x, t, n_hutchinson=n_hutchinson)
            v = self.v(x, t)
            x = x + dt * v
            log_p = log_p - dt * div
        return x, log_p

    def _sample_odeint(
        self,
        n: int,
        device: torch.device | str = "cpu",
        n_steps: int = 100,
    ) -> tuple[Tensor, Tensor]:
        if odeint is None:
            raise RuntimeError("torchdiffeq is not installed. Install with extras [ode].")
        x0 = self.prior.sample(n, device=device)
        t_span = torch.linspace(0.0, 1.0, n_steps + 1, device=x0.device, dtype=x0.dtype)

        class _Wrapper(nn.Module):
            def __init__(self, velocity: VelocityField) -> None:
                super().__init__()
                self.velocity = velocity

            def forward(self, t_scalar: Tensor, x_state: Tensor) -> Tensor:
                t = torch.full(
                    (x_state.shape[0],),
                    t_scalar.item(),
                    device=x_state.device,
                    dtype=x_state.dtype,
                )
                return self.velocity(x_state, t)

        xt = odeint(_Wrapper(self.v), x0, t_span, method="dopri5")
        x = xt[-1]
        # For ODEINT mode, obtain log_prob via backward integration API.
        log_q = self.log_prob(x, n_steps=n_steps, method="odeint")
        return x, log_q

    def sample(
        self,
        n: int,
        device: torch.device | str = "cpu",
        n_steps: int = 100,
        n_hutchinson: int = 1,
        method: str = "euler",
    ) -> tuple[Tensor, Tensor]:
        """Sample x ~ q and return (x, log_q(x))."""
        if method == "odeint":
            return self._sample_odeint(n, device=device, n_steps=n_steps)
        return self._sample_euler(n, device=device, n_steps=n_steps, n_hutchinson=n_hutchinson)

    def _log_prob_euler(
        self,
        x: Tensor,
        n_steps: int = 100,
        n_hutchinson: int = 1,
    ) -> Tensor:
        n = x.shape[0]
        z = x
        dt = 1.0 / n_steps
        div_int = torch.zeros(n, device=x.device, dtype=x.dtype)

        for k in range(n_steps):
            t_value = 1.0 - k * dt
            t = torch.full((n,), t_value, device=x.device, dtype=x.dtype)
            div = _divergence_hutchinson(self.v, z, t, n_hutchinson=n_hutchinson)
            v = self.v(z, t)
            z = z - dt * v
            div_int = div_int + dt * div
        return self.prior.log_prob(z) - div_int

    def _log_prob_odeint(self, x: Tensor, n_steps: int = 100) -> Tensor:
        if odeint is None:
            raise RuntimeError("torchdiffeq is not installed. Install with extras [ode].")
        t_span = torch.linspace(1.0, 0.0, n_steps + 1, device=x.device, dtype=x.dtype)

        class _BackwardDynamics(nn.Module):
            def __init__(self, outer: CNFFlowModel) -> None:
                super().__init__()
                self.outer = outer

            def forward(
                self,
                t_scalar: Tensor,
                states: tuple[Tensor, Tensor],
            ) -> tuple[Tensor, Tensor]:
                z, logp = states
                t = torch.full((z.shape[0],), t_scalar.item(), device=z.device, dtype=z.dtype)
                with torch.enable_grad():
                    z_req = z.detach().requires_grad_(True)
                    v = self.outer.v(z_req, t)
                    trace = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
                    for i in range(z.shape[1]):
                        (grad_i,) = torch.autograd.grad(
                            v[:, i].sum(),
                            z_req,
                            create_graph=False,
                            retain_graph=True,
                        )
                        trace = trace + grad_i[:, i]
                dz_dt = -v
                dlogp_dt = trace
                return dz_dt, dlogp_dt

        logp_init = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        zt, log_corr = odeint(_BackwardDynamics(self), (x, logp_init), t_span, method="dopri5")
        z0 = zt[-1]
        corr0 = log_corr[-1]
        return self.prior.log_prob(z0) + corr0

    def log_prob(
        self,
        x: Tensor,
        n_steps: int = 100,
        n_hutchinson: int = 1,
        method: str = "euler",
    ) -> Tensor:
        """Estimate log-density by integrating CNF dynamics backward."""
        if method == "odeint":
            return self._log_prob_odeint(x, n_steps=n_steps)
        return self._log_prob_euler(x, n_steps=n_steps, n_hutchinson=n_hutchinson)
