"""Test-time-adaptation baselines for the CADENCE stream.

All share the Adapter protocol (.adapt(X, Y=None) / .predict(X)) so they drop into
b2ss.stream.run_stream, and all wrap a FROZEN decoder — only adapter-owned params move.
Tent and CoTTA are RE-DERIVED for a BN-free GRU velocity regressor (no entropy/BN to
exploit) against the SAME source-latent-moment objective CADENCE uses, so the comparison
isolates the adaptation MECHANISM (carry-forward vs teacher-restore vs scheduled reset vs
structured+revert), not the objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cadence import CADENCE, _t, unsup_objective
from .train import predict as _predict


def _freeze(dec: nn.Module) -> nn.Module:
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


class NoAdapt:
    """Frozen decoder; adaptation is a no-op. Accuracy floor / stability ceiling."""
    def __init__(self, decoder):
        self.decoder = _freeze(decoder)

    def adapt(self, X, Y=None):
        pass

    def predict(self, X):
        return _predict(self.decoder, X)


class _AffineBase(nn.Module):
    """Frozen decoder + a per-channel affine — the free, unstructured analogue of
    CADENCE's fast head (no conduction anchor, no collapse revert). Subclasses set the
    adaptation policy."""
    def __init__(self, decoder, n_chan, src_stats, lr=0.05, steps=20, device="cpu"):
        super().__init__()
        self.decoder = _freeze(decoder)
        self.gain = nn.Parameter(torch.ones(n_chan))
        self.bias = nn.Parameter(torch.zeros(n_chan))
        self.lr, self.steps, self.device = lr, steps, device
        sm, sv = src_stats
        self.register_buffer("src_mean", torch.as_tensor(sm, dtype=torch.float32))
        self.register_buffer("src_var", torch.as_tensor(sv, dtype=torch.float32))
        self.to(device)

    def forward(self, x):
        return self.decoder(self.gain[None, :, None] * x + self.bias[None, :, None])

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(_t(X, self.device)).cpu().numpy()


class Tent(_AffineBase):
    """Affine adapted to match source moments, CARRIED FORWARD across the stream with
    no reset and no safety net — the canonical error-accumulation foil."""
    def adapt(self, X, Y=None):
        Xt = _t(X, self.device)
        opt = torch.optim.Adam([self.gain, self.bias], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            unsup_objective(self, Xt, self.src_mean, self.src_var).backward()
            opt.step()


class CoTTA(_AffineBase):
    """Mean-teacher (EMA of the adapted affine) + stochastic restoration (random params
    reset to source each step). Predicts with the teacher — the anti-forgetting reference."""
    def __init__(self, *a, ema=0.99, restore_p=0.02, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("t_gain", torch.ones_like(self.gain))
        self.register_buffer("t_bias", torch.zeros_like(self.bias))
        self.ema, self.restore_p = float(ema), float(restore_p)

    def adapt(self, X, Y=None):
        Xt = _t(X, self.device)
        opt = torch.optim.Adam([self.gain, self.bias], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            unsup_objective(self, Xt, self.src_mean, self.src_var).backward()
            opt.step()
            with torch.no_grad():
                self.gain[torch.rand_like(self.gain) < self.restore_p] = 1.0   # stochastic restore
                self.bias[torch.rand_like(self.bias) < self.restore_p] = 0.0
                self.t_gain.mul_(self.ema).add_((1 - self.ema) * self.gain)     # EMA teacher
                self.t_bias.mul_(self.ema).add_((1 - self.ema) * self.bias)

    @torch.no_grad()
    def predict(self, X):
        Xt = _t(X, self.device)
        x = self.t_gain[None, :, None] * Xt + self.t_bias[None, :, None]
        return self.decoder(x).cpu().numpy()


class RDumb:
    """Wrap any adapter and reset it to source every `reset_every` visits (NeurIPS 2023
    — the credibility gate: a win must beat scheduled reset over a long stream)."""
    def __init__(self, inner_factory, reset_every: int = 3):
        self.factory, self.reset_every = inner_factory, int(reset_every)
        self.inner, self.t = inner_factory(), 0

    def adapt(self, X, Y=None):
        if self.t > 0 and self.t % self.reset_every == 0:
            self.inner = self.factory()                      # reset to source
        self.inner.adapt(X, Y)
        self.t += 1

    def predict(self, X):
        return self.inner.predict(X)


def free_lora(decoder, n_chan, src_stats, **kw):
    """The make-or-break ablation: CADENCE with the diagonal affine swapped for a dense
    rank-1 channel map at MATCHED parameter count. Isolates whether interpretable
    STRUCTURE (not param count or the anchor/revert machinery) carries the stability."""
    return CADENCE(decoder, n_chan, head="lora", rank=1, src_stats=src_stats, **kw)


# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    import numpy as np
    from .baselines import GRUDecoder
    from .transfer import source_feature_stats

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    C, W, N = 6, 20, 400
    Xs = rng.standard_normal((N, C, W)).astype("float32")
    dec = GRUDecoder(C, n_out=2, hidden=16, layers=1)
    smean, svar = source_feature_stats(dec, Xs)
    w0 = dec.head.weight.detach().clone()

    # No-Adapt: predict == raw decoder; adapt leaves weights byte-identical
    na = NoAdapt(dec)
    assert np.allclose(na.predict(Xs), _predict(dec, Xs))
    na.adapt(Xs)
    assert torch.equal(dec.head.weight, w0)

    # gain-drifted target the affine can fix -> unsup objective drops for Tent & CoTTA
    gain_c = (1.0 + 0.5 * rng.standard_normal(C)).astype("float32")
    Xg = (Xs * gain_c[None, :, None]).astype("float32")

    def obj(mod):
        with torch.no_grad():
            return float(unsup_objective(mod, _t(Xg), mod.src_mean, mod.src_var))

    tent = Tent(dec, C, (smean, svar), lr=0.1, steps=40)
    o0 = obj(tent); tent.adapt(Xg); assert obj(tent) < o0, (obj(tent), o0)
    g1 = tent.gain.detach().clone()
    tent.adapt(Xg)                                           # carries forward (no reset)
    assert not torch.equal(tent.gain, g1)                   # state persisted & kept moving
    assert torch.equal(dec.head.weight, w0)                 # decoder frozen

    cotta = CoTTA(dec, C, (smean, svar), lr=0.1, steps=40, restore_p=0.05)
    oc = obj(cotta); cotta.adapt(Xg); assert obj(cotta) <= oc + 1e-6
    # teacher is a genuine EMA: between identity and the student
    assert (cotta.t_gain - 1.0).abs().sum() <= (cotta.gain - 1.0).abs().sum() + 1e-4

    # free-LoRA: matched param count to a CADENCE affine head (2C each) and is a CADENCE
    ref = CADENCE(dec, C, src_stats=(smean, svar))
    fl = free_lora(dec, C, (smean, svar))
    n_affine = ref.gain.numel() + ref.bias.numel()
    n_lora = fl.U.numel() + fl.V.numel()
    assert abs(n_lora - n_affine) / n_affine <= 0.2, (n_lora, n_affine)
    assert hasattr(fl, "adapt") and hasattr(fl, "predict")

    # RDumb: resets the inner to a fresh source-state on schedule (reset_every visits)
    def factory():
        return Tent(dec, C, (smean, svar), lr=0.1, steps=20)
    rd = RDumb(factory, reset_every=2)
    rd.adapt(Xg); rd.adapt(Xg)                              # calls 1,2 -> inner A (no reset yet)
    id_a = id(rd.inner)
    rd.adapt(Xg)                                            # call 3: t%2==0 -> reset to fresh B
    assert id(rd.inner) != id_a                             # reset fired on schedule
    rd.adapt(Xg)                                            # call 4: no reset -> still B
    assert id(rd.inner) != id_a

    print(f"tta_baselines.py self-check OK: No-Adapt frozen; Tent carries forward "
          f"(obj {o0:.3f}->{obj(tent):.3f}); CoTTA teacher bounded; free-LoRA matched "
          f"({n_lora}~{n_affine} params); RDumb reset OK")


if __name__ == "__main__":
    _selfcheck()
