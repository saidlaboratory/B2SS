"""Phase 10 — conduction-aware transfer normalization.

Train a decoder once on a source pool, FREEZE it, and adapt only a tiny
conduction-delay normaliser per target subject/session. The normaliser applies
low-dimensional per-tract-group temporal delays (delta in R^K, K<<n_chan), obtained
in one of three modes:

    zero-shot     delta = measured/known conduction  (no target neural data)
    few-shot      fit delta to a few LABELED target trials (decoder frozen)
    unsupervised  fit delta to UNLABELED target data by matching the frozen
                  decoder's latent statistics to the source (CORAL-style moments)

The novelty is that the adaptation space is low-dim and conduction-structured, so it
is data-efficient and regularised. See PIVOT.md / the design spec.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import fractional_shift

DEFAULT_GROUPS = 8
DEFAULT_MAX_DELAY = 12.0


class ConductionDelayAligner(nn.Module):
    """Per-tract-group temporal delays applied to (B, C, W). K learnable group
    delays expand to per-channel delays via a fixed channel->group map."""

    def __init__(self, n_chan: int, n_groups: int = DEFAULT_GROUPS,
                 max_delay: float = DEFAULT_MAX_DELAY):
        super().__init__()
        self.n_chan, self.n_groups, self.max_delay = n_chan, n_groups, float(max_delay)
        self.register_buffer("group_ids", torch.arange(n_chan) % n_groups)
        self.delta = nn.Parameter(torch.zeros(n_groups))     # group delays, bins

    def per_channel_delays(self) -> torch.Tensor:
        return self.delta.clamp(-self.max_delay, self.max_delay)[self.group_ids]

    @torch.no_grad()
    def set_group_delays(self, group_delays) -> None:
        g = torch.as_tensor(group_delays, dtype=torch.float32, device=self.delta.device)
        assert g.numel() == self.n_groups, (g.numel(), self.n_groups)
        self.delta.copy_(g.clamp(-self.max_delay, self.max_delay))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fractional_shift(x, self.per_channel_delays())


class TransferNormalizer(nn.Module):
    """Wraps a FROZEN decoder with a conduction-delay aligner (+ optional spatial
    aligner). Only the aligner adapts per target; the decoder never changes."""

    def __init__(self, decoder: nn.Module, n_chan: int, n_groups: int = DEFAULT_GROUPS,
                 max_delay: float = DEFAULT_MAX_DELAY, spatial: nn.Module | None = None):
        super().__init__()
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()
        self.spatial = spatial                               # optional (EEG); borrowed
        self.aligner = ConductionDelayAligner(n_chan, n_groups, max_delay)

    def trainable_parameters(self):
        ps = list(self.aligner.parameters())
        if self.spatial is not None:
            ps += [p for p in self.spatial.parameters() if p.requires_grad]
        return ps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial is not None:
            x = self.spatial(x)
        return self.decoder(self.aligner(x))


# --------------------------------------------------------------------------- #
# DelayFitter — the three modes
# --------------------------------------------------------------------------- #
def _t(a, device="cpu", dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def set_measured(norm: TransferNormalizer, group_delays) -> None:
    """Zero-shot: set delta from a measured/known per-group conduction."""
    norm.aligner.set_group_delays(group_delays)


def _loss_fn(task):
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


def fit_supervised(norm: TransferNormalizer, X, Y, *, epochs=60, lr=0.05,
                   batch_size=256, device="cpu", seed=0) -> list[float]:
    """Few-shot: optimise delta (decoder frozen) on a few LABELED target trials."""
    torch.manual_seed(seed)
    norm.to(device)
    clf = getattr(norm.decoder.cfg, "task", "regression") == "classification"
    Xt = _t(X, device)
    Yt = _t(Y, device, torch.long if clf else torch.float32)
    opt = torch.optim.Adam(norm.trainable_parameters(), lr=lr)
    loss_fn = _loss_fn("classification" if clf else "regression")
    g = torch.Generator().manual_seed(seed)
    hist = []
    for _ in range(epochs):
        norm.aligner.train()
        for b in torch.randperm(len(Xt), generator=g).split(batch_size):
            opt.zero_grad()
            loss = loss_fn(norm(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
        with torch.no_grad():
            hist.append(float(loss_fn(norm(Xt), Yt)))
    return hist


def source_feature_stats(decoder: nn.Module, X_source, device="cpu"):
    """Mean/var of the frozen decoder's pre-head latent over source data (for the
    unsupervised objective). Captured via a forward hook on `decoder.head`."""
    decoder.to(device).eval()
    grabbed = {}
    h = decoder.head.register_forward_hook(lambda m, inp, out: grabbed.__setitem__("z", inp[0].detach()))
    try:
        with torch.no_grad():
            decoder(_t(X_source, device))
        z = grabbed["z"]
    finally:
        h.remove()
    return z.mean(0), z.var(0, unbiased=False)


def fit_unsupervised(norm: TransferNormalizer, X_target, src_mean, src_var, *,
                     epochs=80, lr=0.05, batch_size=256, device="cpu", seed=0) -> list[float]:
    """Unsupervised: optimise delta so the frozen decoder's latent on aligned target
    data matches the source latent moments (CORAL-style). No target labels."""
    torch.manual_seed(seed)
    norm.to(device)
    Xt = _t(X_target, device)
    grabbed = {}
    h = norm.decoder.head.register_forward_hook(lambda m, inp, out: grabbed.__setitem__("z", inp[0]))
    opt = torch.optim.Adam(norm.trainable_parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    hist = []
    try:
        for _ in range(epochs):
            for b in torch.randperm(len(Xt), generator=g).split(batch_size):
                opt.zero_grad()
                norm(Xt[b])                                  # populates grabbed['z'] (with grad)
                z = grabbed["z"]
                loss = ((z.mean(0) - src_mean) ** 2).mean() + ((z.var(0, unbiased=False) - src_var) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                norm(Xt)
                z = grabbed["z"]
                hist.append(float(((z.mean(0) - src_mean) ** 2).mean()
                                  + ((z.var(0, unbiased=False) - src_var) ** 2).mean()))
    finally:
        h.remove()
    return hist


# --------------------------------------------------------------------------- #
def _selfcheck() -> None:
    from types import SimpleNamespace

    class ToyDecoder(nn.Module):
        """Fixed linear decoder reading the WINDOW CENTRE of each channel, so a
        temporal shift is cleanly identifiable. .head exposes the latent (the
        centre-timepoint channel vector)."""
        def __init__(self, n_chan, win, n_out=2):
            super().__init__()
            self.cfg = SimpleNamespace(task="regression", n_out=n_out)
            self.mid = win // 2
            self.head = nn.Linear(n_chan, n_out)

        def forward(self, x, *a, **k):
            return self.head(x[:, :, self.mid])

    rng = np.random.default_rng(0)
    C, W, N, K = 16, 20, 500, 4
    gids = (np.arange(C) % K)
    # per-channel ramp: X[n,c,t] = amp[n,c] * t/W (+noise). Non-stationary -> a shift
    # changes the centre value, so both label-fit and variance-match can identify it.
    amp = rng.standard_normal((N, C)).astype("float32")
    ramp = (np.arange(W) / W).astype("float32")
    Xs = (amp[:, :, None] * ramp[None, None, :]).astype("float32") + 0.02 * rng.standard_normal((N, C, W)).astype("float32")
    dec = ToyDecoder(C, W)
    with torch.no_grad():
        dec.head.weight.normal_(0, 0.5)
    Ys = dec(_t(Xs)).detach().numpy()

    # target = source shifted by a KNOWN per-group delay; aligner should undo it (delta = -known)
    known = np.array([0.0, 3.0, -2.0, 4.0], dtype="float32")
    Xt = fractional_shift(_t(Xs), _t(known[gids])).detach().numpy()
    Yt = Ys                                                 # only inputs shifted

    # freeze check
    norm = TransferNormalizer(dec, C, n_groups=K, max_delay=8)
    assert all(not p.requires_grad for p in norm.decoder.parameters())
    w0 = norm.decoder.head.weight.detach().clone()

    # zero-shot: measured undo delays = -known -> recovers source-level loss
    set_measured(norm, -known)
    mse_zero = float(((norm(_t(Xt)).detach().numpy() - Yt) ** 2).mean())
    mse_none = float(((TransferNormalizer(dec, C, K, 8)(_t(Xt)).detach().numpy() - Yt) ** 2).mean())
    assert mse_zero < mse_none, (mse_zero, mse_none)        # alignment helps

    # few-shot: fit delta from labels -> recovers the undo delays, decoder unchanged
    norm2 = TransferNormalizer(dec, C, K, 8)
    hist = fit_supervised(norm2, Xt, Yt, epochs=120, lr=0.1, seed=0)
    assert hist[-1] < hist[0]
    assert torch.allclose(norm2.decoder.head.weight, w0)   # decoder frozen
    rec = norm2.aligner.delta.detach().numpy()
    assert np.mean(np.abs(rec - (-known))) < 1.0, rec       # recovered ~ -known

    # unsupervised: match source latent moments -> moves delta toward the undo
    smean, svar = source_feature_stats(dec, Xs)
    norm3 = TransferNormalizer(dec, C, K, 8)
    uh = fit_unsupervised(norm3, Xt, smean, svar, epochs=150, lr=0.1, seed=0)
    assert uh[-1] < uh[0]
    err_after = np.mean(np.abs(norm3.aligner.delta.detach().numpy() - (-known)))
    assert err_after < np.mean(np.abs(known)) , err_after   # closer to truth than zero-init
    print(f"transfer.py self-check OK: zero-shot mse {mse_zero:.3f}<{mse_none:.3f}; "
          f"few-shot recovered delta≈{rec.round(1)}; unsup err {err_after:.2f}; decoder frozen")


if __name__ == "__main__":
    _selfcheck()
