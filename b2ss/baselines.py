"""Published baselines for the real-EEG benchmark.

  EEGNet (Lawhern et al. 2018) — faithful PyTorch reimplementation of the
    compact CNN (temporal conv -> depthwise spatial conv -> separable conv),
    with the paper's max-norm constraints. Competitive but tiny (~few k params).
  CSP + LDA — the classical filter-bank-CSP-family baseline (mne CSP + sklearn).

Both expose the same call shape as the B2SS decoder so run_real_benchmark.py can
line them up. EEGNet plugs into train.fit() via a minimal .cfg and a constrain_()
hook (max-norm enforced after each optimizer step).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _maxnorm_(w: torch.Tensor, maxval: float, dim):
    norm = w.norm(2, dim=dim, keepdim=True)
    w.mul_((maxval / (norm + 1e-8)).clamp(max=1.0))


class EEGNet(nn.Module):
    def __init__(self, n_chan: int, n_time: int, n_classes: int = 2, fs: float = 160.0,
                 F1: int = 8, D: int = 2, F2: int = 16, drop: float = 0.5):
        super().__init__()
        self.cfg = SimpleNamespace(task="classification", n_classes=n_classes)
        kern = int(fs // 2)
        self.conv1 = nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depth = nn.Conv2d(F1, F1 * D, (n_chan, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1, self.drop1 = nn.AvgPool2d((1, 4)), nn.Dropout(drop)
        self.sep_dw = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_pw = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2, self.drop2 = nn.AvgPool2d((1, 8)), nn.Dropout(drop)
        with torch.no_grad():                        # size the classifier from a dummy pass
            flat = self._features(torch.zeros(1, 1, n_chan, n_time)).shape[1]
        self.head = nn.Linear(flat, n_classes)

    def _features(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.depth(x)))))
        x = self.drop2(self.pool2(F.elu(self.bn3(self.sep_pw(self.sep_dw(x))))))
        return torch.flatten(x, 1)

    def forward(self, x, cv=None, cv_sd=None):       # x: (B, C, T); cv ignored
        return self.head(self._features(x.unsqueeze(1)))

    @torch.no_grad()
    def constrain_(self):
        _maxnorm_(self.depth.weight, 1.0, dim=(1, 2, 3))   # spatial filters
        _maxnorm_(self.head.weight, 0.25, dim=1)           # dense units


def eegnet_param_count(n_chan=64, n_time=480, n_classes=2) -> int:
    m = EEGNet(n_chan, n_time, n_classes)
    return sum(p.numel() for p in m.parameters())


class GRUDecoder(nn.Module):
    """Recurrent baseline for continuous kinematic decoding (the LFADS/POSSM family
    is intracortical-RNN-based; this is the lightweight stand-in). Optionally carries
    a ChannelDelay front-end (Phase 8) so the CV/delay idea can ride a *competitive*
    decoder — the "does CV help a good decoder?" test."""

    def __init__(self, n_chan: int, n_out: int = 2, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2,
                 align_mode: str = "none", max_delay_bins: float = 10.0):
        super().__init__()
        from .model import ChannelDelay
        self.cfg = SimpleNamespace(task="regression", n_out=n_out)
        self.align = ChannelDelay(n_chan, max_delay_bins, align_mode)
        self.gru = nn.GRU(n_chan, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x, cv=None, cv_sd=None, delays=None):   # x: (B, C, W)
        o, _ = self.gru(self.align(x, delays).transpose(1, 2))   # (B, W, hidden)
        return self.head(self.drop(o[:, -1]))


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha: float = 1.0) -> float:
    """Linear baseline: Ridge on the flattened window. Returns variance-weighted R²."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    ntr, nte = len(Xtr), len(Xte)
    clf = Ridge(alpha=alpha).fit(Xtr.reshape(ntr, -1), Ytr)
    return float(r2_score(Yte, clf.predict(Xte.reshape(nte, -1)),
                          multioutput="variance_weighted"))


def csp_lda_accuracy(Xtr, ytr, Xte, yte, n_components: int = 6) -> float:
    """Classical CSP + LDA. X: (n, C, T)."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    import mne
    mne.set_log_level("ERROR")
    clf = Pipeline([("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
                    ("lda", LinearDiscriminantAnalysis())])
    clf.fit(Xtr.astype(np.float64), ytr)
    return float((clf.predict(Xte.astype(np.float64)) == yte).mean())


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(1) == y).mean())


def _selfcheck() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 64, 480)
    m = EEGNet(64, 480, 2)
    assert m(x).shape == (6, 2)
    m(x).sum().backward(); m.constrain_()
    assert m.depth.weight.norm(2, dim=(1, 2, 3)).max() <= 1.0 + 1e-4
    assert m.head.weight.norm(2, dim=1).max() <= 0.25 + 1e-4
    p = eegnet_param_count()
    assert 1500 < p < 6000, p            # tiny, per the paper

    # CSP+LDA separates two clearly-different covariance classes
    rng = np.random.default_rng(0)
    A = rng.standard_normal((64, 64)); Bm = rng.standard_normal((64, 64))
    Xa = np.array([A @ rng.standard_normal((64, 480)) for _ in range(30)])
    Xb = np.array([Bm @ rng.standard_normal((64, 480)) for _ in range(30)])
    X = np.concatenate([Xa, Xb]); y = np.array([0] * 30 + [1] * 30)
    tr = rng.permutation(60)[:40]; te = np.setdiff1d(np.arange(60), tr)
    acc = csp_lda_accuracy(X[tr], y[tr], X[te], y[te])
    assert acc > 0.7, acc
    print(f"baselines.py self-check OK: EEGNet {p} params, forward+maxnorm OK; "
          f"CSP+LDA acc={acc:.2f} on synthetic covariance classes")


if __name__ == "__main__":
    _selfcheck()
