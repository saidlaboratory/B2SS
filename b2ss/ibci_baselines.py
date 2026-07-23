"""Strong intracortical-BCI stabilizers as continual-stream baselines.

These are mechanism-faithful reimplementations on the Adapter protocol (.adapt/.predict)
over a FROZEN decoder — not the authors' exact code — labeled as *-style and validated
on a sanity slice (each must beat No-Adapt on a drifted session):

  MPA-style  (Membrane Potential Alignment, arXiv 2606.14866): align each session's
             per-channel input/membrane moments to the source distribution, CLOSED-FORM
             (recompute standardization per session; the AdaBN/Euclidean-alignment family).
             Label-free, no gradient — the cheap strong baseline.

  NoMAD-style (Karpowicz et al., Nat Commun 2025): a learned full-rank linear READIN fit
             UNSUPERVISED to map the new session onto the source latent manifold, matching
             the frozen decoder's pre-head latent mean+covariance (CORAL). The expressive
             strong baseline reviewers will demand — and CADENCE's nearest neighbour.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .cadence import _t
from .train import predict as _predict


def _freeze(dec):
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def source_input_stats(X):
    """Per-channel input mean/std over (samples, time). X: (N, C, W)."""
    X = np.asarray(X, np.float32)
    return X.mean((0, 2)), X.std((0, 2)) + 1e-6


def _latent(decoder, x_tensor):
    """Grab the frozen decoder's pre-head latent (B, H) for x (a tensor, grad-preserving)."""
    grabbed = {}
    h = decoder.head.register_forward_hook(lambda m, inp, out: grabbed.__setitem__("z", inp[0]))
    try:
        decoder(x_tensor)
    finally:
        h.remove()
    return grabbed["z"]


def source_latent_moments(decoder, X, device="cpu"):
    """Source latent mean (H,) and covariance (H, H) for the NoMAD CORAL objective."""
    with torch.no_grad():
        z = _latent(_freeze(decoder), _t(X, device))
    mean = z.mean(0)
    zc = z - mean
    cov = (zc.t() @ zc) / max(1, len(z) - 1)
    return mean, cov


class MPA:
    """Closed-form per-channel standardization of each session to the source input
    distribution (membrane/input moment alignment). Label-free, no gradient.

    `std_floor` matters more than it looks. On a sparse multi-electrode array a sizeable
    minority of channels are near-silent over a short calibration slice (~15% of the 96
    Indy electrodes have sd < 0.1 over 25 windows), and dividing by their raw sd sends the
    aligned input to ±1e3. Without the floor this baseline does not merely degrade at small
    N — it diverges to negative R². Pass std_floor=0.0 to reproduce that failure mode.
    """
    def __init__(self, decoder, src_stats_x, device="cpu", std_floor: float = 0.1):
        self.decoder = _freeze(decoder)
        self.mu_s, self.sd_s = (np.asarray(a, np.float32) for a in src_stats_x)
        self.device = device
        self.std_floor = float(std_floor)
        self.mu_t, self.sd_t = self.mu_s, self.sd_s          # identity until adapted

    def adapt(self, X, Y=None):
        mu, sd = source_input_stats(X)                       # this session's own moments
        self.mu_t, self.sd_t = mu, np.maximum(sd, self.std_floor)

    def _align(self, X):
        X = np.asarray(X, np.float32)
        z = (X - self.mu_t[None, :, None]) / self.sd_t[None, :, None]
        return (z * self.sd_s[None, :, None] + self.mu_s[None, :, None]).astype(np.float32)

    def predict(self, X):
        return _predict(self.decoder, self._align(X))


class NoMAD(nn.Module):
    """Learned full-rank linear readin W (C,C), fit unsupervised to match the frozen
    decoder's latent mean+covariance to the source (CORAL manifold alignment)."""
    def __init__(self, decoder, n_chan, src_moments, lr=0.02, steps=40, device="cpu"):
        super().__init__()
        self.decoder = _freeze(decoder)
        self.W = nn.Parameter(torch.eye(n_chan))
        self.lr, self.steps, self.device = lr, steps, device
        m, c = src_moments
        self.register_buffer("src_mean", torch.as_tensor(m, dtype=torch.float32))
        self.register_buffer("src_cov", torch.as_tensor(c, dtype=torch.float32))
        self.to(device)

    def forward(self, x):
        return self.decoder(torch.einsum("cd,bdt->bct", self.W, x))

    def _coral(self, x):
        z = _latent(self.decoder, torch.einsum("cd,bdt->bct", self.W, x))
        mean = z.mean(0)
        zc = z - mean
        cov = (zc.t() @ zc) / max(1, len(z) - 1)
        return ((cov - self.src_cov) ** 2).sum() + ((mean - self.src_mean) ** 2).sum()

    def adapt(self, X, Y=None):
        Xt = _t(X, self.device)
        opt = torch.optim.Adam([self.W], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            self._coral(Xt).backward()
            opt.step()

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        return self(_t(X, self.device)).cpu().numpy()


# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    from .baselines import GRUDecoder

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    C, W, N = 8, 20, 400
    Xs = rng.standard_normal((N, C, W)).astype("float32")
    dec = GRUDecoder(C, n_out=2, hidden=16, layers=1)
    w0 = dec.head.weight.detach().clone()
    Ys = _predict(dec, Xs)                                    # self-consistent labels

    # a drifted target: per-channel gain + shift (the real firing-rate-drift analogue)
    gain = (1.0 + 0.6 * rng.standard_normal(C)).astype("float32")
    shift = (0.5 * rng.standard_normal(C)).astype("float32")
    Xt = (Xs * gain[None, :, None] + shift[None, :, None]).astype("float32")

    def err(mod, X):
        return float(((mod.predict(X) - Ys) ** 2).mean())

    e_none = float(((_predict(dec, Xt) - Ys) ** 2).mean())    # No-Adapt on the drifted target

    # MPA: closed-form standardization undoes the per-channel gain/shift -> beats No-Adapt
    mpa = MPA(dec, source_input_stats(Xs))
    mpa.adapt(Xt)
    assert err(mpa, Xt) < e_none, (err(mpa, Xt), e_none)
    assert torch.equal(dec.head.weight, w0)                  # frozen

    # the std floor is load-bearing. Calibrate on a short slice where channel 0 happens to
    # be quiet, then decode the rest of the session where it isn't: the unfloored aligner
    # divides that channel's real deviations by ~0 and explodes. Floored stays bounded.
    Xq = Xt[:25].copy()
    Xq[:, 0, :] = Xq[:, 0, :].mean() + 1e-3 * rng.standard_normal(Xq[:, 0, :].shape)
    hot, cold = MPA(dec, source_input_stats(Xs), std_floor=0.0), MPA(dec, source_input_stats(Xs))
    hot.adapt(Xq); cold.adapt(Xq)
    assert np.abs(hot._align(Xt)).max() > 100 * np.abs(cold._align(Xt)).max()
    assert err(cold, Xt) < err(hot, Xt)                      # and the floor is what saves R²

    # NoMAD: learned readin matching source latent moments -> beats No-Adapt, decoder frozen
    nomad = NoMAD(dec, C, source_latent_moments(dec, Xs), lr=0.05, steps=80)
    nomad.adapt(Xt)
    assert err(nomad, Xt) < e_none, (err(nomad, Xt), e_none)
    assert torch.equal(dec.head.weight, w0)

    print(f"ibci_baselines.py self-check OK: No-Adapt {e_none:.3f} | "
          f"MPA {err(mpa, Xt):.3f} | NoMAD {err(nomad, Xt):.3f} (both beat No-Adapt); decoder frozen")


if __name__ == "__main__":
    _selfcheck()
