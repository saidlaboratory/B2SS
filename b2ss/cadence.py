"""CADENCE — consolidation-shrinkage test-time adapter.

Wraps a FROZEN decoder with a per-channel affine head that is the only thing moving at
test time:

    x ──▶ [ per-channel affine, fit by consolidation shrinkage ] ──▶ FROZEN decoder

The whole method is how that affine is set. A plain per-session standardiser (MPA) takes
each channel's mean/std straight from the current session's calibration windows; with few
windows those estimates are noisy and the resulting decode degrades badly. CADENCE shrinks
each estimate toward the source prior with weight w = n/(n+tau) (empirical-Bayes /
James-Stein), so the adapter starts near the identity and earns its way to a full
standardiser as calibration data arrives. Adapted DOF = 2*n_chan, 2-3 orders below a
retrain.

The controller (collapse sensor) reverts the head to identity when the unsupervised
objective spikes more than `collapse_z` sigma above its running history. It is NOT dormant:
on the real Indy stream it fires on roughly 3 of 10 session visits, and those visits are
exactly the ones where the adapter would otherwise have lost to the frozen decoder — the
revert turns a loss into a tie. So CADENCE's "never worse than No-Adapt" property is
produced by the shrinkage AND the controller together, not by the shrinkage alone. (An
earlier version of this docstring claimed the controller never fired for the closed-form
head. That was wrong, and it mis-attributed the one property of this method that survives.)

EARLIER DESIGN, REMOVED: this adapter used to compose the affine with a
`ConductionDelayAligner` "slow anchor". On every stream we ran, no measured conduction
velocity exists, the anchor's delays stayed at zero, and it was an exact identity op — so
it is gone from the method. Conduction survives where it is actually load-bearing: as the
drift-decomposition diagnostic in `scripts/run_decomposition_figure.py`, built on
`transfer.ConductionDelayAligner`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .transfer import source_feature_stats


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
    def __init__(self, decoder: nn.Module, n_chan: int, *, fast_lr: float = 0.05,
                 fast_steps: int = 20, collapse_z: float = 3.0, src_stats=None,
                 head: str = "affine", rank: int = 1, reg: float = 0.0,
                 std_floor: float = 0.1, shrink_tau: float = 200.0, shrink: str = "fixed",
                 device: str = "cpu"):
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
        self.fast_lr = float(fast_lr)
        self.fast_steps, self.collapse_z = int(fast_steps), float(collapse_z)
        self.device = device
        sm, sv = (src_stats if src_stats is not None else (torch.zeros(1), torch.ones(1)))
        self.register_buffer("src_mean", _t(np.asarray(sm.detach() if torch.is_tensor(sm) else sm), device))
        self.register_buffer("src_var", _t(np.asarray(sv.detach() if torch.is_tensor(sv) else sv), device))
        self.reg = float(reg)                                # shrinkage toward the closed-form init
        self.std_floor = float(std_floor)                    # scale floor for robust standardisation
        self.shrink_tau = float(shrink_tau)                  # consolidation shrinkage strength (windows)
        if shrink not in ("fixed", "eb"):
            raise ValueError(f"shrink must be 'fixed' or 'eb', got {shrink}")
        self.shrink = shrink                                 # see _shrink_weight
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
        return self.decoder(self._apply_head(x))

    def _reset_fast_head(self) -> None:
        with torch.no_grad():
            if self.head_mode == "affine":
                self.gain.fill_(1.0)
                self.bias.zero_()
            else:
                self.U.zero_()
                self.V.normal_(0, 0.01)

    @torch.no_grad()
    def _shrink_weight(self, n, mt, st):
        """How far to trust this session's per-channel estimates over the source prior.
        Returns (w_mean, w_scale) — the mean and the scale need different answers.

        `fixed`  w = n/(n+tau) for both. One scalar, one knob. A single tau is a compromise
                 across calibration budgets: the sweep prefers tau≈800 at n=500 and tau≈200
                 at n<=100 (RESULTS §I.3), which is what motivated the `eb` mode.

        `eb`     Efron-Morris empirical Bayes, no hyperparameter. The input is
                 source-standardised, so the prior is exactly "mean 0, scale 1" and the
                 observed dispersion of the estimates around it is an estimate of the true
                 drift PLUS sampling noise. Subtract the noise to get the prior variance,
                 then weight by the usual posterior ratio:

                     mean :  var_c = s_c^2/n            prior v_m = <m_c^2> - <var_c>
                     scale:  var_c = s_c^2/(2n)         prior v_s = <(s_c-1)^2> - <var_c>
                     w = v / (v + var_c)

                 MEASURED RESULT: `eb` does NOT work here. It collapses onto a plain
                 standardiser (+0.010 R2 over MPA at n=25) and loses to `fixed` by −0.400
                 (0/8 sessions). The reason is instructive and is why `fixed` is still the
                 default: EB assumes the session estimate is unbiased and merely noisy, so
                 when the between-session drift is genuinely large it concludes "trust the
                 data" and sets w -> 1. But the first n windows of a session are not a
                 noisy-but-centred sample of it, they are a chronologically biased one, and
                 no amount of correct variance accounting can see a bias. A fixed tau
                 hedges against that bias by refusing to commit — which is the right
                 behaviour for the wrong stated reason. Kept as a runnable negative result.

                 The mean weight is per-channel (the sampling variance of a mean genuinely
                 scales with the channel's variance); the scale weight is per-session,
                 because the *relative* error of a sample std is 1/sqrt(2n) for every
                 channel regardless of how loud it is. Getting that backwards — shrinking
                 the scale per-channel by the channel's own variance — tells you to TRUST a
                 quiet channel's tiny std, which is precisely the estimate that explodes the
                 gain. Both weights -> 1 as n -> inf, recovering a plain standardiser.
        """
        if self.shrink == "fixed":
            w = torch.as_tensor(n / (n + self.shrink_tau), dtype=torch.float32, device=mt.device)
            return w, w
        n = max(int(n), 1)
        var_m = (st ** 2) / n                                    # sampling var of the mean
        v_m = ((mt ** 2).mean() - var_m.mean()).clamp(min=0.0)   # noise-corrected prior var
        w_mean = (v_m / (v_m + var_m)).clamp(0.0, 1.0)
        var_s = (st ** 2) / (2 * n)                              # sampling var of the std
        v_s = (((st - 1.0) ** 2).mean() - var_s.mean()).clamp(min=0.0)
        w_scale = (v_s / (v_s + var_s.mean())).clamp(0.0, 1.0)
        return w_mean, w_scale

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
        # CONSOLIDATION SHRINKAGE: pull each per-channel estimate toward the source prior
        # (source-standardised input -> mean 0, std 1). With few calibration windows the raw
        # per-channel mean/std is noisy and a plain standardiser is worse than not adapting;
        # leaning on the prior keeps it safe, and the shrink also bounds the scale so a
        # silent channel cannot explode to gain = 1/std. w -> 1 with ample data (full MPA).
        w_m, w_s = self._shrink_weight(n, mt, st)
        m_s = w_m * mt                                        # -> source mean 0
        s_s = (w_s * st + (1.0 - w_s) * 1.0).clamp(min=self.std_floor)  # -> source std 1
        self.gain.copy_(1.0 / s_s)
        self.bias.copy_(-m_s / s_s)
        self._init_gain.copy_(self.gain)
        self._init_bias.copy_(self.bias)

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

    class ToyDecoder(nn.Module):
        """Fixed linear decoder reading each channel's window CENTRE, so a per-channel
        gain is cleanly identifiable."""
        def __init__(self, n_chan, win, n_out=2):
            super().__init__()
            self.cfg = SimpleNamespace(task="regression", n_out=n_out)
            self.mid = win // 2
            self.head = nn.Linear(n_chan, n_out)

        def forward(self, x, *a, **k):
            return self.head(x[:, :, self.mid])

    rng = np.random.default_rng(0)
    C, W, N = 16, 20, 800
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

    # (1) THE METHOD: consolidation shrinkage interpolates between doing nothing (few
    # windows -> w = n/(n+tau) small -> head near identity) and a full standardiser
    # (ample windows -> w -> 1 -> the MPA solution). Both ends must be right.
    gain_c = (1.0 + 0.5 * rng.standard_normal(C)).astype("float32")
    Xg = (Xs * gain_c[None, :, None]).astype("float32")
    cad = CADENCE(dec, C, shrink_tau=200.0, src_stats=(smean, svar))
    cad.init_standardize(Xg[:5])                             # n=5 << tau -> barely moves
    g, b = cad.gain.detach(), cad.bias.detach()
    assert float((g - 1).abs().max()) < 0.1 and float(b.abs().max()) < 0.1
    cad.init_standardize(Xg)                                 # n=800 >> tau -> ~full standardise
    with torch.no_grad():
        z = cad._apply_head(_t(Xg))
    assert abs(float(z.mean())) < 0.2 and 0.8 < float(z.std()) < 1.25
    assert torch.allclose(cad.decoder.head.weight, w0)       # decoder frozen throughout

    # (2) GAIN-drift gap -> few-shot supervised adapt recovers (fast head learns ~1/gain).
    cad2 = CADENCE(dec, C, fast_steps=120, fast_lr=0.1, reg=0.0, src_stats=(smean, svar))
    e0 = err(cad2, Xg, Ys)
    cad2.adapt(Xg, Ys)                                       # few-shot (labels available)
    assert err(cad2, Xg, Ys) < e0, (err(cad2, Xg, Ys), e0)
    assert torch.allclose(cad2.decoder.head.weight, w0)      # still frozen

    # (3) Controller (safety net): the collapse-sensor reverts the head when the objective
    # spikes. It is live for the closed-form affine too — on the real Indy stream it fires
    # on ~3 of 10 visits, which is where the zero-regret property comes from.
    cad3 = CADENCE(dec, C, collapse_z=2.0, src_stats=(smean, svar))
    for _ in range(4):
        cad3._collapse_check(_t(Xs))                         # build a calm history
    with torch.no_grad():
        cad3.gain.mul_(50.0); cad3.bias.add_(20.0)           # inject divergence
    r0 = cad3.n_reverts
    cad3._collapse_check(_t(Xs))
    assert cad3.n_reverts == r0 + 1                          # reverted
    assert torch.allclose(cad3.gain, torch.ones(C)) and torch.allclose(cad3.bias, torch.zeros(C))

    # (3b) empirical-Bayes shrinkage: no tau. More windows must mean more trust in the
    # session, and the weights must stay in [0, 1] and keep the gain bounded.
    cad_eb = CADENCE(dec, C, shrink="eb", src_stats=(smean, svar))
    Xq = Xg.copy(); Xq[:, 0, :] *= 0.02                      # channel 0 barely fires

    def wts(mod, A):
        t = _t(A)
        return mod._shrink_weight(len(A), t.mean((0, 2)), t.std((0, 2)))

    (wm_few, ws_few), (wm_many, ws_many) = wts(cad_eb, Xq[:8]), wts(cad_eb, Xq)
    assert float(ws_many) > float(ws_few)                    # more data -> trust the scale more
    for w in (wm_few, wm_many, ws_few, ws_many):
        assert 0.0 <= float(w.min()) and float(w.max()) <= 1.0

    # No real mean drift here (the target is the source times a per-channel gain), so the
    # observed spread of the means is pure sampling noise and EB must attribute it as such:
    # w_mean stays ~0 at EVERY n. That is the estimator working, not failing — a fixed tau
    # would happily fit that noise once n got large.
    assert float(wm_few.mean()) < 0.05 and float(wm_many.mean()) < 0.05
    # inject a genuine offset and it must now trust the data, increasingly so with n
    Xo = (Xq + rng.normal(0, 0.5, C).astype("float32")[None, :, None]).astype("float32")
    wm_o_few, wm_o_many = wts(cad_eb, Xo[:8])[0], wts(cad_eb, Xo)[0]
    assert float(wm_o_many.mean()) > float(wm_o_few.mean()) > 0.05
    cad_eb.init_standardize(Xq)                              # must stay bounded like `fixed`
    assert float(cad_eb.gain.detach().abs().max()) <= 1.0 / cad_eb.std_floor + 1e-3
    # the scale weight is per-session (a scalar), NOT per-channel: the relative error of a
    # sample std is 1/sqrt(2n) for every channel, so a quiet channel's tiny std is not
    # better estimated than a loud one's. Per-channel here would trust exactly the estimate
    # that blows the gain up.
    assert ws_many.numel() == 1 and wm_many.numel() == C

    # (4) memoryless by construction: two sessions in a row leave no trace of the first,
    # which is why the stream BWT is exactly 0 (a property, not an achievement — MPA
    # shares it; Tent, which carries state forward, does not).
    cad4 = CADENCE(dec, C, src_stats=(smean, svar))
    cad4.adapt(Xs)
    g_solo = cad4.gain.detach().clone()
    cad4.adapt(Xg); cad4.adapt(Xs)
    assert torch.allclose(cad4.gain, g_solo)

    # (5) closed-form init sets a BOUNDED standardising affine (silent channels do NOT
    # explode to 1/std ~ 1e6), roughly standardises the input, and a regularized few-shot
    # refine stays finite. Xg has a silent-ish channel to exercise the std floor.
    Xg2 = Xg.copy(); Xg2[:, 0, :] = 0.0                      # a silent channel
    cad5 = CADENCE(dec, C, reg=0.1, std_floor=0.1, fast_steps=30, src_stats=(smean, svar))
    cad5.init_standardize(Xg2)
    assert float(cad5.gain.detach().abs().max()) <= 1.0 / cad5.std_floor + 1e-3   # no explosion
    with torch.no_grad():
        z = cad5._apply_head(_t(Xg2))
    assert abs(float(z.mean())) < 0.5 and 0.4 < float(z.std()) < 1.6       # roughly standardized
    cad5.adapt(Xg2, Ys)                                      # regularized supervised refine
    assert np.isfinite(err(cad5, Xg2, Ys))

    print(f"cadence.py self-check OK: shrinkage spans identity->standardiser; head fixes gain "
          f"({e0:.3f}->{err(cad2, Xg, Ys):.3f}); collapse-revert + memoryless refit; decoder frozen")


if __name__ == "__main__":
    _selfcheck()
