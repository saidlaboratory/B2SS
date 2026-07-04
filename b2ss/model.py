"""The B2SS decoder and its matched-capacity spatial-only control (proposal 4.5-4.6).

B2SS  = Transformer encoder (time-token) -> CV modulation gate -> Euler Neural-ODE
        readout -> 6-DOF kinematics.
Control = identical, except the CV gate is removed and the integration window tau
        is a single *learned* constant (per subject). Same architecture and
        (to within one scalar) the same parameter count — so any advantage is
        attributable to the information content of CV, not extra capacity.

Design note: the proposal calls channels the tokens but also says tau drives
"temporal attention masking". Those are only consistent if the tokens run along
*time*, so we tokenise over the 50 timepoints (each a 64-channel vector) and let
tau set a soft recency mask over them. Training is per-subject, so CV (hence tau)
is constant within a run — the attention mask is a single shared 2-D bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import FS, N_CHAN, WIN, N_KIN

MASK_GAMMA = 0.5      # steepness of the recency attention mask
ODE_STEPS = 20        # fixed Euler steps (shapes stay static; tau sets total time)


@dataclass
class DecoderConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    tau_min_ms: float = 20.0
    tau_max_ms: float = 100.0
    use_cv_gate: bool = True


def proposal_config(use_cv_gate: bool = True) -> DecoderConfig:
    """Full sizes from the proposal (d_model 256, 8 heads, 4 layers). Slow on CPU."""
    return DecoderConfig(d_model=256, nhead=8, num_layers=4, dim_ff=256,
                         use_cv_gate=use_cv_gate)


def _norm_cv(cv: torch.Tensor) -> torch.Tensor:
    """Map plausible CV range (~25-70 m/s) to roughly [-1, 1] for the gate."""
    return (cv - 47.5) / 22.5


class _EncoderLayer(nn.Module):
    """Pre-LN block: recency-masked multi-head attention + feed-forward."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff), nn.GELU(),
            nn.Linear(cfg.dim_ff, cfg.d_model))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class B2SSDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        c = self.cfg
        self.in_proj = nn.Linear(N_CHAN, c.d_model)
        self.pos = nn.Parameter(torch.zeros(1, WIN, c.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(_EncoderLayer(c) for _ in range(c.num_layers))
        self.ln_out = nn.LayerNorm(c.d_model)

        if c.use_cv_gate:
            # tau = tau_min + sigmoid(w*cv_n + b)*(tau_max-tau_min).
            # w<0 encodes the prior "faster CV -> shorter window".
            self.w_cv = nn.Parameter(torch.tensor(-1.0))
            self.b_cv = nn.Parameter(torch.tensor(0.0))
        else:
            self.tau_raw = nn.Parameter(torch.tensor(0.0))  # learned constant tau

        # Neural-ODE dynamics f_theta and readout.
        self.f = nn.Sequential(
            nn.Linear(c.d_model, c.d_model), nn.Tanh(),
            nn.Linear(c.d_model, c.d_model))
        self.readout = nn.Linear(c.d_model, N_KIN)

    # -- tau (integration window) in milliseconds -------------------------- #
    def tau_ms(self, cv: torch.Tensor | None) -> torch.Tensor:
        c = self.cfg
        if c.use_cv_gate:
            if cv is None:
                raise ValueError("B2SS decoder needs a CV input")
            gate = torch.sigmoid(self.w_cv * _norm_cv(cv).mean() + self.b_cv)
        else:
            gate = torch.sigmoid(self.tau_raw)
        return c.tau_min_ms + gate * (c.tau_max_ms - c.tau_min_ms)

    def _recency_mask(self, tau_ms: torch.Tensor, device) -> torch.Tensor:
        """Additive (WIN, WIN) attention bias: penalise keys older than the
        span implied by tau. Differentiable in tau."""
        span = tau_ms / 1000.0 * FS                       # samples
        j = torch.arange(WIN, device=device, dtype=tau_ms.dtype)
        age = (WIN - 1) - j                               # 0 = most recent
        bias = -MASK_GAMMA * torch.relu(age - span)       # (WIN,)
        return bias.unsqueeze(0).expand(WIN, WIN)         # broadcast over queries

    def forward(self, x: torch.Tensor, cv: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, WIN) -> tokens over time (B, WIN, C)
        h = self.in_proj(x.transpose(1, 2)) + self.pos
        tau = self.tau_ms(cv)
        mask = self._recency_mask(tau, x.device)
        for layer in self.layers:
            h = layer(h, mask)
        z = self.ln_out(h).mean(dim=1)                    # (B, d_model)

        # Euler Neural-ODE: integrate over total time tau, fixed step count.
        dt = (tau / 1000.0) / ODE_STEPS
        for _ in range(ODE_STEPS):
            z = z + dt * self.f(z)
        return self.readout(z)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_pair(cfg: DecoderConfig | None = None) -> tuple[B2SSDecoder, B2SSDecoder]:
    """(b2ss, control) — identical config apart from the CV gate."""
    base = cfg or DecoderConfig()
    b2ss = B2SSDecoder(DecoderConfig(**{**base.__dict__, "use_cv_gate": True}))
    ctrl = B2SSDecoder(DecoderConfig(**{**base.__dict__, "use_cv_gate": False}))
    return b2ss, ctrl


def _selfcheck() -> None:
    torch.manual_seed(0)
    b2ss, ctrl = make_pair()

    x = torch.randn(8, N_CHAN, WIN)
    cv = torch.full((8,), 60.0)
    assert b2ss(x, cv).shape == (8, N_KIN)
    assert ctrl(x).shape == (8, N_KIN)

    # tau stays within [tau_min, tau_max]
    for cvv in (25.0, 47.5, 70.0):
        t = b2ss.tau_ms(torch.tensor(cvv)).item()
        assert 20.0 - 1e-4 <= t <= 100.0 + 1e-4, t
    # prior: faster CV -> shorter window at init
    assert b2ss.tau_ms(torch.tensor(65.0)) < b2ss.tau_ms(torch.tensor(30.0))

    # matched capacity: within 1%
    pb, pc = count_params(b2ss), count_params(ctrl)
    assert abs(pb - pc) / pb < 0.01, (pb, pc)

    # gate is differentiable (gradient flows to w_cv)
    loss = b2ss(x, cv).pow(2).mean()
    loss.backward()
    assert b2ss.w_cv.grad is not None and torch.isfinite(b2ss.w_cv.grad)
    print(f"model.py self-check OK: params b2ss={pb} ctrl={pc} "
          f"(delta {abs(pb-pc)}); tau(65)<tau(30)")


if __name__ == "__main__":
    _selfcheck()
