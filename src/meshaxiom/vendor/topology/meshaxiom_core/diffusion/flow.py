"""Rectified flow matching with velocity prediction (PAPER.txt:309-311).

Convention: ``t = 0`` is noise, ``t = 1`` is data,

    x_t = (1 - t) * x0 + t * x1,      v = x1 - x0,

and the network predicts ``v`` at ``(x_t, t)``.

Sampling integrates ``dx/dt = v`` from 0 to 1.  ``dpmpp2m`` needs the usual
diffusion parameterisation, obtained from the rectified-flow schedule by

    x_t = alpha_t * x1 + sigma_t * x0,   alpha_t = t,  sigma_t = 1 - t,
    log-SNR   lambda_t = log(alpha_t / sigma_t) = logit(t),
    data prediction  D(x_t, t) = x_t + (1 - t) * v.

DPM-Solver++(2M) in data-prediction form is then

    h   = lambda_next - lambda_cur,          r = h_prev / h,
    D~  = (1 + 1/(2r)) * D_cur - 1/(2r) * D_prev,
    x   = (sigma_next / sigma_cur) * x + alpha_next * (1 - exp(-h)) * D~,

which is *exact* whenever the data prediction is constant along the trajectory
(the first-order form is the flow-matching analogue of DDIM).  ``exp(-h)`` is
evaluated in closed form as ``t_cur (1-t_next) / ((1-t_cur) t_next)`` so that the
endpoints t=0 and t=1 stay exact; the logit itself is only needed for the
multistep ratio ``r`` and is clamped to ``[eps, 1-eps]`` there.
"""

from __future__ import annotations

import torch

LAMBDA_EPS = 1e-5


def occ_threshold(mode: str = "unit") -> float:
    """Decision threshold on the generated occupancy values (SPEC §5)."""
    return {"unit": 0.5, "signed": 0.0}[mode]


def encode_occ(occ: torch.Tensor, mode: str = "unit") -> torch.Tensor:
    """Binary occupancy -> the real-valued diffusion target."""
    return occ if mode == "unit" else occ * 2.0 - 1.0


def _bt(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast ``t`` — (B,), (B,N) or scalar — against ``x`` (B, N, C)."""
    if t.dim() == 0:
        return t
    return t.reshape(*t.shape, *([1] * (x.dim() - t.dim())))


class RectifiedFlow:
    """Straight-path flow matching; ``t_sampler`` is ``uniform`` or ``logitnorm``."""

    def __init__(self, t_sampler: str = "uniform", logitnorm_mean=0.0, logitnorm_std=1.0):
        self.t_sampler = t_sampler
        self.logitnorm_mean, self.logitnorm_std = logitnorm_mean, logitnorm_std

    def sample_t(self, shape, device, generator=None) -> torch.Tensor:
        if self.t_sampler == "uniform":  # paper is silent; uniform is the default
            return torch.rand(shape, device=device, generator=generator)
        if self.t_sampler == "logitnorm":  # NOTE(deviation): SD3-style, optional
            n = torch.randn(shape, device=device, generator=generator)
            return torch.sigmoid(n * self.logitnorm_std + self.logitnorm_mean)
        raise ValueError(f"unknown t sampler {self.t_sampler!r}")

    def add_noise(self, x1: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        tt = _bt(t, x1)
        return (1.0 - tt) * noise + tt * x1

    def training_target(self, x1: torch.Tensor, noise: torch.Tensor, t) -> torch.Tensor:
        """Velocity target; constant along the straight path, so ``t`` is unused."""
        return x1 - noise

    @torch.no_grad()
    def sample(self, model_fn, x_init: torch.Tensor, steps: int = 20, sampler: str = "dpmpp2m",
               cfg_scale: float = 1.0, uncond_fn=None) -> torch.Tensor:
        """Integrate ``model_fn(x_t, t) -> v`` from t=0 (``x_init``) to t=1."""
        x = x_init
        ts = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=torch.float32)

        def vel(xt, tv):
            tb = torch.full((xt.shape[0],), float(tv), device=xt.device)
            v = model_fn(xt, tb)
            if cfg_scale != 1.0 and uncond_fn is not None:
                vu = uncond_fn(xt, tb)
                v = vu + cfg_scale * (v - vu)
            return v.float()

        if sampler == "euler":
            for i in range(steps):
                x = x + (ts[i + 1] - ts[i]) * vel(x, ts[i])
            return x
        if sampler == "heun":
            for i in range(steps):
                dt = ts[i + 1] - ts[i]
                v1 = vel(x, ts[i])
                if i == steps - 1:  # t=1 is the degenerate end of the schedule: plain Euler
                    x = x + dt * v1
                    break
                v2 = vel(x + dt * v1, ts[i + 1])
                x = x + dt * 0.5 * (v1 + v2)
            return x
        if sampler != "dpmpp2m":
            raise ValueError(f"unknown sampler {sampler!r}")

        lam = torch.logit(ts.clamp(LAMBDA_EPS, 1.0 - LAMBDA_EPS))
        d_prev, h_prev = None, None
        for i in range(steps):
            tc, tn = ts[i], ts[i + 1]
            d = x + (1.0 - tc) * vel(x, tc)
            h = lam[i + 1] - lam[i]
            # "lower order final" (as in the reference DPM-Solver++): the last step's h
            # is inflated by the LAMBDA_EPS clamp (logit(1-eps) ~ 11.5), so r = h_prev/h
            # is tiny and the multistep correction becomes a huge extrapolation --
            # at 20 steps, D_use = 6.73 D_cur - 5.73 D_prev, amplifying any disagreement
            # between consecutive predictions ~6x right where it lands on the output.
            if d_prev is None or i == steps - 1:
                d_use = d
            else:
                r = h_prev / h
                d_use = (1 + 1 / (2 * r)) * d - (1 / (2 * r)) * d_prev
            emh = tc * (1.0 - tn) / ((1.0 - tc) * tn)  # exp(-h), in closed form
            x = (1.0 - tn) / (1.0 - tc) * x + tn * (1.0 - emh) * d_use
            d_prev, h_prev = d, h
        return x
