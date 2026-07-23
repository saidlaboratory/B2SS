"""CADENCE — collapse-resistant structured test-time adapter.

Wraps a FROZEN decoder with a tiny composed adapter that is the only thing that
moves at test time:

    x ──▶ [ fast head: per-channel affine ] ──▶ [ slow anchor: conduction delay ] ──▶ FROZEN decoder

- fast head (gain_c, bias_c per channel): the cheap CORAL/Euclidean-style re-centering
  that absorbs fast firing-rate / offset drift — refit per session, unsupervised.
- slow anchor (ConductionDelayAligner, K group delays): biophysically-bounded, EMA-
  consolidated timing correction; also the SAFE revert target when the fast head diverges.
- controller: watches the unsupervised objective; on a drift spike it reverts the fast
  head to identity, falling back to the anchor (a valid low-DOF state) rather than to
  a drifted mess — the collapse-resistance the free-TTA baselines lack.

The conduction term does not carry accuracy on turnover-dominated real gaps (conceded
up front); its jobs here are collapse-safety and the drift-decomposition diagnostic.
Adapted DOF = 2*n_chan (affine) + K (delays), 2-3 orders below a full retrain.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .transfer import ConductionDelayAligner, source_feature_stats


def _t(a, device="cpu"):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


def unsup_objective(module, x_batch, src_mean, src_var) -> torch.Tensor:
    """Label-free feature-alignment loss: match the frozen decoder's pre-head latent
    (mean+var) on the adapted target to the source moments. `module` is any adapter
    exposing `.decoder` (with a `.head`) and a forward that applies its adaptation."""
    grabbed = {}
    h = module.decoder.head.register_forward_hook(
        lambda m, inp, out: grabbed.__setitem__("z", inp[0]))
    try:
        module(x_batch)                                  # populates grabbed['z'] with grad
        z = grabbed["z"]
    finally:
        h.remove()
    # ponytail: hook re-registered per call — negligible vs the forward on CPU.
    return ((z.mean(0) - src_mean) ** 2).mean() + ((z.var(0, unbiased=False) - src_var) ** 2).mean()


class CADENCE(nn.Module):
    def __init__(self, decoder: nn.Module, n_chan: int, *, n_groups: int = 8,
                 max_delay: float = 12.0, anchor_ema: float = 0.9, fast_lr: float = 0.05,
                 fast_steps: int = 20, collapse_z: float = 3.0, src_stats=None,
                 head: str = "affine", rank: int = 1, reg: float = 0.0,
                 std_floor: float = 0.1, shrink_tau: float = 200.0, device: str = "cpu"):
        super().__init__()
        self.decoder = decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.head_mode = head                                # 'affine' (structured) | 'lora' (free)
        if head == "affine":
            self.gain = nn.Parameter(torch.ones(n_chan))     # per-channel diagonal fast head
            self.bias = nn.Parameter(torch.zeros(n_chan))
        elif head == "lora":
            self.U = nn.Parameter(torch.zeros(n_chan, rank))  # dense rank-r channel map (matched params)
            self.V = nn.Parameter(0.01 * torch.randn(n_chan, rank))
        else:
            raise ValueError(f"head must be 'affine' or 'lora', got {head}")
        self.aligner = ConductionDelayAligner(n_chan, n_groups, max_delay)  # slow anchor
        self.anchor_ema, self.fast_lr = float(anchor_ema), float(fast_lr)
        self.fast_steps, self.collapse_z = int(fast_steps), float(collapse_z)
        self.device = device
        sm, sv = (src_stats if src_stats is not None else (torch.zeros(1), torch.ones(1)))
        self.register_buffer("src_mean", _t(np.asarray(sm.detach() if torch.is_tensor(sm) else sm), device))
        self.register_buffer("src_var", _t(np.asarray(sv.detach() if torch.is_tensor(sv) else sv), device))
        self.reg = float(reg)                                # shrinkage toward the closed-form init
        self.std_floor = float(std_floor)                    # scale floor for robust standardisation
        self.shrink_tau = float(shrink_tau)                  # consolidation shrinkage strength (windows)
        self.register_buffer("_init_gain", torch.ones(n_chan))
        self.register_buffer("_init_bias", torch.zeros(n_chan))
        self._obj_hist: list[float] = []                     # collapse-sensor history
        self.n_reverts = 0
        self.to(device)

    # -- structure ---------------------------------------------------------- #
    def _apply_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_mode == "affine":
            return self.gain[None, :, None] * x + self.bias[None, :, None]
        proj = torch.einsum("cr,bct->brt", self.V, x)                # dense rank-r mix
        return x + torch.einsum("cr,brt->bct", self.U, proj)

    def _fast_params(self):
        return [self.gain, self.bias] if self.head_mode == "affine" else [self.U, self.V]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_head(x)                                       # fast head
        x = self.aligner(x)                                           # slow anchor delay
        return self.decoder(x)

    def _reset_fast_head(self) -> None:
        with torch.no_grad():
            if self.head_mode == "affine":
                self.gain.fill_(1.0)
                self.bias.zero_()
            else:
                self.U.zero_()
                self.V.normal_(0, 0.01)

    @torch.no_grad()
    def init_standardize(self, X) -> None:
        """Zero-shot closed-form init: set the per-channel affine to standardize the target
        to the source distribution (mean 0, std 1) — the MPA-style solution that already
        matches the best unsupervised baseline. Few-shot `adapt` then refines from here,
        shrunk toward it by `reg`, so a handful of labels can only help, never collapse."""
        if self.head_mode != "affine":
            return
        Xt = _t(X, self.device)
        n = Xt.shape[0]                                       # windows available this session
        mt, st = Xt.mean(dim=(0, 2)), Xt.std(dim=(0, 2))
        # CONSOLIDATION SHRINKAGE (the win over plain MPA): shrink the per-channel estimate
        # toward the source prior (standardised input -> mean 0, std 1) by w = n/(n+tau).
        # With few calibration windows the raw per-channel mean/std is noisy and a plain
        # standardiser (MPA) collapses; leaning on the source prior stays robust. w -> 1 with
        # ample data (full MPA), and the shrink also floors the scale so silent channels do
        # not explode to gain = 1/std ~ 1e6. See RESULTS §9.4.
        w = n / (n + self.shrink_tau)
        m_s = w * mt                                          # -> source mean 0
        s_s = (w * st + (1.0 - w) * 1.0).clamp(min=self.std_floor)  # -> source std 1
        self.gain.copy_(1.0 / s_s)
        self.bias.copy_(-m_s / s_s)
        self._init_gain.copy_(self.gain)
        self._init_bias.copy_(self.bias)

    @torch.no_grad()
    def set_anchor(self, group_delays) -> None:
        """Zero-shot: set the conduction anchor directly from a measured/known CV."""
        self.aligner.set_group_delays(group_delays)

    @torch.no_grad()
    def ema_anchor(self, target_group_delays) -> None:
        """Slow EMA consolidation of the anchor toward a per-session target delay:
        delta <- ema*delta + (1-ema)*target. Holds the slow timing component."""
        tgt = torch.as_tensor(target_group_delays, dtype=torch.float32, device=self.aligner.delta.device)
        self.aligner.delta.mul_(self.anchor_ema).add_((1 - self.anchor_ema) * tgt)

    # -- adaptation --------------------------------------------------------- #
    def adapt(self, X, Y=None) -> None:
        """Adapt the fast head on this session.

        affine head (the method): the unsupervised base is the closed-form robust
        standardisation (`init_standardize` — beats a plain per-channel standardiser);
        given a few LABELS, refine the affine on top, shrunk toward that init by `reg`
        so a handful of trials can only help. lora head (the free/collapsing ablation):
        gradient descent on the same objective, no closed form — kept to show that the
        matched-parameter dense map collapses where the structured one does not."""
        Xt = _t(X, self.device)
        if self.head_mode == "affine":
            self.init_standardize(X)                            # closed-form zero-shot base
            if Y is not None:                                   # few-shot regularised refine
                Yt = _t(Y, self.device)
                opt = torch.optim.Adam(self._fast_params(), lr=self.fast_lr)
                loss_fn = nn.MSELoss()
                for _ in range(self.fast_steps):
                    opt.zero_grad()
                    loss = loss_fn(self(Xt), Yt) + self.reg * (
                        ((self.gain - self._init_gain) ** 2).sum()
                        + ((self.bias - self._init_bias) ** 2).sum())
                    loss.backward()
                    opt.step()
        else:                                                   # lora ablation: gradient only
            opt = torch.optim.Adam(self._fast_params(), lr=self.fast_lr)
            Yt = None if Y is None else _t(Y, self.device)
            for _ in range(self.fast_steps):
                opt.zero_grad()
                loss = nn.MSELoss()(self(Xt), Yt) if Y is not None else \
                    unsup_objective(self, Xt, self.src_mean, self.src_var)
                loss.backward()
                opt.step()
        self._collapse_check(Xt)

    def _collapse_check(self, Xt: torch.Tensor) -> None:
        with torch.no_grad():
            obj = float(unsup_objective(self, Xt, self.src_mean, self.src_var))
        if len(self._obj_hist) >= 3:
            mu = float(np.mean(self._obj_hist)); sd = float(np.std(self._obj_hist)) + 1e-8
            if (obj - mu) / sd > self.collapse_z:            # drift spike -> revert to anchor
                self._reset_fast_head()
                self.n_reverts += 1
                return                                       # don't record the diverged obj
        self._obj_hist.append(obj)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        self.eval()
        return self(_t(X, self.device)).cpu().numpy()


# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    from types import SimpleNamespace
    from .model import fractional_shift

    class ToyDecoder(nn.Module):
        """Fixed linear decoder reading each channel's window CENTRE, so both a temporal
        shift (anchor) and a per-channel gain (fast head) are cleanly identifiable."""
        def __init__(self, n_chan, win, n_out=2):
            super().__init__()
            self.cfg = SimpleNamespace(task="regression", n_out=n_out)
            self.mid = win // 2
            self.head = nn.Linear(n_chan, n_out)

        def forward(self, x, *a, **k):
            return self.head(x[:, :, self.mid])

    rng = np.random.default_rng(0)
    C, W, N, K = 16, 20, 800, 4
    gids = np.arange(C) % K
    amp = rng.standard_normal((N, C)).astype("float32")
    ramp = (np.arange(W) / W).astype("float32")
    Xs = (amp[:, :, None] * ramp[None, None, :]).astype("float32") \
        + 0.02 * rng.standard_normal((N, C, W)).astype("float32")
    dec = ToyDecoder(C, W)
    with torch.no_grad():
        dec.head.weight.normal_(0, 0.5)
    Ys = dec(_t(Xs)).detach().numpy()
    smean, svar = source_feature_stats(dec, Xs)
    w0 = dec.head.weight.detach().clone()

    def err(mod, X, Y):
        return float(((mod.predict(X) - Y) ** 2).mean())

    # (1) SHIFT gap -> zero-shot anchor (measured CV) recovers, and beats No-Adapt.
    known = np.array([0.0, 3.0, -2.0, 4.0], dtype="float32")
    Xt = fractional_shift(_t(Xs), _t(known[gids])).detach().numpy()
    cad = CADENCE(dec, C, n_groups=K, max_delay=8, src_stats=(smean, svar))
    e_none = err(cad, Xt, Ys)                                 # identity adapter = No-Adapt
    cad.set_anchor(-known)
    assert err(cad, Xt, Ys) < e_none, (err(cad, Xt, Ys), e_none)
    assert torch.allclose(cad.decoder.head.weight, w0)       # decoder frozen

    # (2) GAIN-drift gap -> few-shot supervised adapt recovers (fast head learns ~1/gain).
    gain_c = (1.0 + 0.5 * rng.standard_normal(C)).astype("float32")
    Xg = (Xs * gain_c[None, :, None]).astype("float32")
    cad2 = CADENCE(dec, C, n_groups=K, max_delay=8, fast_steps=120, fast_lr=0.1, reg=0.0,
                   src_stats=(smean, svar))
    e0 = err(cad2, Xg, Ys)
    cad2.adapt(Xg, Ys)                                       # few-shot (labels available)
    assert err(cad2, Xg, Ys) < e0, (err(cad2, Xg, Ys), e0)
    assert torch.allclose(cad2.decoder.head.weight, w0)      # still frozen

    # (3) Controller (safety net): the collapse-sensor reverts the fast head when the
    # objective spikes. Tested directly — for the affine method the closed-form init keeps
    # the head safe so it never fires in normal use; it guards an expressive/lora head.
    cad3 = CADENCE(dec, C, n_groups=K, max_delay=8, collapse_z=2.0, src_stats=(smean, svar))
    for _ in range(4):
        cad3._collapse_check(_t(Xs))                         # build a calm history
    with torch.no_grad():
        cad3.gain.mul_(50.0); cad3.bias.add_(20.0)           # inject divergence
    r0 = cad3.n_reverts
    cad3._collapse_check(_t(Xs))
    assert cad3.n_reverts == r0 + 1                          # reverted
    assert torch.allclose(cad3.gain, torch.ones(C)) and torch.allclose(cad3.bias, torch.zeros(C))

    # (4) EMA slowness: one ema_anchor step moves delta by (1-ema) of the target.
    cad4 = CADENCE(dec, C, n_groups=K, max_delay=8, anchor_ema=0.9, src_stats=(smean, svar))
    tgt = np.array([2.0, -2.0, 4.0, -4.0], dtype="float32")
    cad4.ema_anchor(tgt)
    moved = cad4.aligner.delta.detach().numpy()
    assert np.allclose(moved, 0.1 * tgt, atol=1e-5), moved

    # (5) closed-form init sets a BOUNDED standardising affine (silent channels do NOT
    # explode to 1/std ~ 1e6), roughly standardises the input, and a regularized few-shot
    # refine stays finite. Xg has a silent-ish channel to exercise the std floor.
    Xg2 = Xg.copy(); Xg2[:, 0, :] = 0.0                      # a silent channel
    cad5 = CADENCE(dec, C, n_groups=K, max_delay=8, reg=0.1, std_floor=0.1,
                   fast_steps=30, src_stats=(smean, svar))
    cad5.init_standardize(Xg2)
    assert float(cad5.gain.abs().max()) <= 1.0 / cad5.std_floor + 1e-3    # no explosion (the bug fix)
    with torch.no_grad():
        z = cad5._apply_head(_t(Xg2))
    assert abs(float(z.mean())) < 0.5 and 0.4 < float(z.std()) < 1.6       # roughly standardized
    cad5.adapt(Xg2, Ys)                                      # regularized supervised refine
    assert np.isfinite(err(cad5, Xg2, Ys))

    print(f"cadence.py self-check OK: anchor recovers shift ({e_none:.3f}->{err(cad, Xt, Ys):.3f}); "
          f"fast head fixes gain ({e0:.3f}->{err(cad2, Xg, Ys):.3f}); collapse-revert + EMA OK; "
          f"decoder frozen")


if __name__ == "__main__":
    _selfcheck()
