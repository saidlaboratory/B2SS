"""The B2SS decoder, its baselines-as-ablations, and the matched-capacity control.

Architecture: conv patch-embed -> Transformer encoder (time tokens) -> CV
modulation gate -> Euler Neural-ODE readout -> task head (regression or
classification).

The CV gate sets a temporal integration window tau that (a) masks the encoder's
attention toward recent time and (b) sets the ODE integration time. Four gate
modes support a clean ablation of *where the window comes from*:

    'cv'      tau = f(measured CV)          <- B2SS (optionally uncertainty-aware)
    'learned' tau = one learned constant    <- matched-capacity spatial-only control
    'fixed'   tau = a fixed constant        <- no adaptation at all
    'none'    tau = full window (no mask)    <- plain Transformer+ODE baseline

CV is handled PER SAMPLE, so the model works both when CV is constant within a
run (per-subject regime) and when it varies trial to trial (heterogeneous regime,
where a single learned constant window is provably suboptimal). The gate is
uncertainty-aware: a noisy CV estimate (large cv_sd) is shrunk toward the
population mean, so tau falls back to the population-average window (Drakesmith
et al. 2019: trust CV only where the microstructure estimate is reliable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .data import FS, N_CHAN, WIN, N_KIN

MASK_GAMMA = 0.5          # steepness of the recency attention mask
# F1: tau maps to a span that is a FRACTION of the window, so the gate
# differentiates CV identically at any sampling rate / patch size (not an absolute
# ms->tokens count, which was degenerate in the EEG config).
SPAN_FRAC_MIN = 0.10      # tau_min -> attend ~10% of the window
SPAN_FRAC_MAX = 0.60      # tau_max -> ~60%
CV_POP_MEAN = 47.5        # m/s, population centre used by the gate + shrinkage
CV_POP_SCALE = 22.5       # m/s, maps plausible CV range to ~[-1, 1]
CV_SHRINK_SD = 10.0       # m/s, CV-estimate SD at which precision weight = 0.5

GATE_MODES = ("cv", "learned", "fixed", "none")


@dataclass
class DecoderConfig:
    n_chan: int = N_CHAN
    win: int = WIN                 # timepoints in the input window
    fs: int = FS                   # Hz
    patch: int = 1                 # conv patch stride over time (reduces #tokens)
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    ode_steps: int = 20
    dropout: float = 0.0           # regularisation; keep 0 for synthetic, raise for small real EEG
    tau_min_ms: float = 20.0
    tau_max_ms: float = 100.0
    fixed_tau_ms: float = 60.0     # used by gate_mode='fixed'
    gate_mode: str = "cv"
    task: str = "regression"       # 'regression' | 'classification'
    n_out: int = N_KIN             # regression targets
    n_classes: int = 2             # classification classes

    def __post_init__(self):
        assert self.gate_mode in GATE_MODES, self.gate_mode
        assert self.task in ("regression", "classification")
        assert self.win % self.patch == 0, "win must be divisible by patch"

    @property
    def n_tokens(self) -> int:
        return self.win // self.patch

    @property
    def out_dim(self) -> int:
        return self.n_out if self.task == "regression" else self.n_classes


def proposal_config(**kw) -> DecoderConfig:
    """Full sizes from the proposal (d_model 256, 8 heads, 4 layers). Slow on CPU."""
    base = dict(d_model=256, nhead=8, num_layers=4, dim_ff=256)
    base.update(kw)
    return DecoderConfig(**base)


class _EncoderLayer(nn.Module):
    """Pre-LN block: recency-masked multi-head attention + feed-forward."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.nhead = cfg.nhead
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, batch_first=True,
                                          dropout=cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_ff, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.ff(self.ln2(x))


class B2SSDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or DecoderConfig()
        c = self.cfg
        # Conv patch-embed over time: kernel=stride=patch. patch=1 => per-timepoint linear.
        self.embed = nn.Conv1d(c.n_chan, c.d_model, kernel_size=c.patch, stride=c.patch)
        self.pos = nn.Parameter(torch.zeros(1, c.n_tokens, c.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(_EncoderLayer(c) for _ in range(c.num_layers))
        self.ln_out = nn.LayerNorm(c.d_model)

        if c.gate_mode == "cv":
            # tau = tau_min + sigmoid(w*cv_n + b)*(tau_max-tau_min).
            # w<0 encodes the prior "faster CV -> shorter window".
            self.w_cv = nn.Parameter(torch.tensor(-1.0))
            self.b_cv = nn.Parameter(torch.tensor(0.0))
        elif c.gate_mode == "learned":
            self.tau_raw = nn.Parameter(torch.tensor(0.0))  # one learned constant tau

        # F3: non-autonomous Neural-ODE field dz/dt = f(z, t) — time fraction
        # appended, so it is a genuine time-dependent field, not a weight-tied
        # autonomous residual stack.
        self.f = nn.Sequential(
            nn.Linear(c.d_model + 1, c.d_model), nn.Tanh(),
            nn.Linear(c.d_model, c.d_model))
        self.head_drop = nn.Dropout(c.dropout)
        self.head = nn.Linear(c.d_model, c.out_dim)

    # -- tau (integration window) in ms, PER SAMPLE ------------------------ #
    def tau_ms(self, cv=None, cv_sd=None, batch: int = 1, device=None) -> torch.Tensor:
        c = self.cfg
        dev = device or self.pos.device
        if c.gate_mode == "cv":
            if cv is None:
                raise ValueError("gate_mode='cv' needs a CV input")
            cv = torch.as_tensor(cv, dtype=torch.float32, device=dev).reshape(-1)
            if cv_sd is not None:                            # uncertainty shrinkage
                sd = torch.as_tensor(cv_sd, dtype=torch.float32, device=dev).reshape(-1)
                w = 1.0 / (1.0 + (sd / CV_SHRINK_SD) ** 2)
                cv = CV_POP_MEAN + w * (cv - CV_POP_MEAN)
            gate = torch.sigmoid(self.w_cv * ((cv - CV_POP_MEAN) / CV_POP_SCALE) + self.b_cv)
            tau = c.tau_min_ms + gate * (c.tau_max_ms - c.tau_min_ms)
            return tau if tau.numel() > 1 else tau.expand(batch)
        if c.gate_mode == "learned":
            gate = torch.sigmoid(self.tau_raw)
            return (c.tau_min_ms + gate * (c.tau_max_ms - c.tau_min_ms)).expand(batch)
        if c.gate_mode == "fixed":
            return torch.full((batch,), c.fixed_tau_ms, device=dev)
        return torch.full((batch,), c.win / c.fs * 1000.0, device=dev)  # 'none' = full window

    def _recency_bias(self, tau_ms: torch.Tensor) -> torch.Tensor:
        """Per-token additive recency bias (B, T), <=0, from a config-invariant span
        (F1): tau maps to a fraction of the window, so the gate differentiates CV in
        any sampling-rate/patch config. Recent tokens ~0; tokens older than the span
        are penalised. Drives BOTH the attention mask and the pooling weights (F2)."""
        c = self.cfg
        T = c.n_tokens
        frac = SPAN_FRAC_MIN + (tau_ms - c.tau_min_ms) / (c.tau_max_ms - c.tau_min_ms) \
            * (SPAN_FRAC_MAX - SPAN_FRAC_MIN)
        span = frac.clamp(min=1e-3) * T                      # tokens, (B,)
        j = torch.arange(T, device=tau_ms.device, dtype=tau_ms.dtype)
        age = (T - 1) - j                                    # (T,), 0 = most recent
        return -MASK_GAMMA * torch.relu(age[None, :] - span[:, None])  # (B, T)

    def _attn_mask(self, bias: torch.Tensor):
        """None (no mask), 2-D (T,T) if uniform, or 3-D (B*nhead,T,T) per-sample."""
        c = self.cfg
        T = c.n_tokens
        if torch.all(bias == 0):                             # full window -> no mask
            return None
        if bias.shape[0] > 1 and not torch.allclose(bias, bias[:1].expand_as(bias)):
            m = bias.unsqueeze(1).expand(bias.shape[0], T, T)        # (B,T,T)
            return m.repeat_interleave(c.nhead, dim=0)               # (B*nhead,T,T)
        return bias[0].unsqueeze(0).expand(T, T)                     # (T,T) shared

    def forward(self, x: torch.Tensor, cv=None, cv_sd=None) -> torch.Tensor:
        # x: (B, C, WIN) -> patch-embed -> tokens (B, T, d_model)
        h = self.embed(x).transpose(1, 2) + self.pos
        B = x.shape[0]
        tau = self.tau_ms(cv, cv_sd, batch=B, device=x.device)   # (B,)
        bias = self._recency_bias(tau)                           # (B, T)
        mask = self._attn_mask(bias)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.ln_out(h)
        # F2: recency-weighted pool so tau gates the READOUT, not just attention.
        # 'none'/'fixed'-full -> bias all-zero -> softmax uniform -> plain mean.
        pool_w = torch.softmax(bias, dim=-1).unsqueeze(-1)      # (B, T, 1)
        z = (h * pool_w).sum(dim=1)                             # (B, d_model)

        # F3: integrate the non-autonomous field over total time tau (fixed steps).
        dt = (tau / 1000.0 / self.cfg.ode_steps).unsqueeze(1)  # (B, 1)
        for k in range(self.cfg.ode_steps):
            t_frac = torch.full((B, 1), k / self.cfg.ode_steps, device=x.device, dtype=z.dtype)
            z = z + dt * self.f(torch.cat([z, t_frac], dim=1))
        return self.head(self.head_drop(z))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_variants(cfg: DecoderConfig | None = None) -> dict[str, B2SSDecoder]:
    """One decoder per gate mode (same everything else) for ablation."""
    base = cfg or DecoderConfig()
    return {m: B2SSDecoder(DecoderConfig(**{**base.__dict__, "gate_mode": m}))
            for m in GATE_MODES}


def make_pair(cfg: DecoderConfig | None = None) -> tuple[B2SSDecoder, B2SSDecoder]:
    """(b2ss 'cv', control 'learned') — the matched-capacity scientific comparison."""
    v = make_variants(cfg)
    return v["cv"], v["learned"]


def _selfcheck() -> None:
    torch.manual_seed(0)
    b2ss, ctrl = make_pair()
    x = torch.randn(8, N_CHAN, WIN)
    cv = torch.full((8,), 60.0)
    assert b2ss(x, cv).shape == (8, N_KIN)
    assert ctrl(x).shape == (8, N_KIN)

    # heterogeneous CV -> per-sample tau varies
    cv_het = torch.linspace(28.0, 68.0, 8)
    tau = b2ss.tau_ms(cv_het, batch=8)
    assert tau.shape == (8,) and tau[0] > tau[-1]           # slow->wide, fast->narrow
    assert b2ss(x, cv_het).shape == (8, N_KIN)              # per-sample mask path works

    with torch.no_grad():
        # uncertainty shrinkage: huge SD pulls tau toward the population-average window
        tau_conf = float(b2ss.tau_ms(torch.tensor(68.0), batch=1))
        tau_unc = float(b2ss.tau_ms(torch.tensor(68.0), cv_sd=torch.tensor(60.0), batch=1))
        tau_pop = float(b2ss.tau_ms(torch.tensor(CV_POP_MEAN), batch=1))
        assert abs(tau_unc - tau_pop) < abs(tau_conf - tau_pop)
        for cvv in (25.0, 47.5, 70.0):
            t = float(b2ss.tau_ms(torch.tensor(cvv), batch=1)[0])
            assert 20.0 - 1e-4 <= t <= 100.0 + 1e-4, t

    pb, pc = count_params(b2ss), count_params(ctrl)
    assert abs(pb - pc) / pb < 0.01, (pb, pc)

    # classification head
    clf = B2SSDecoder(DecoderConfig(task="classification", n_classes=4, gate_mode="cv"))
    assert clf(x, cv).shape == (8, 4)

    # 'none'/'fixed' variants forward-pass
    v = make_variants()
    assert v["none"](x).shape == (8, N_KIN)
    assert v["fixed"](x).shape == (8, N_KIN)

    b2ss(x, cv).pow(2).mean().backward()
    assert b2ss.w_cv.grad is not None and torch.isfinite(b2ss.w_cv.grad)
    print(f"model.py self-check OK: params cv={pb} learned={pc} (delta {abs(pb-pc)}); "
          f"modes={list(v)}; het+uncertainty+clf paths OK")


if __name__ == "__main__":
    _selfcheck()
