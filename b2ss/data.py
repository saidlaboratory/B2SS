"""Synthetic neural data with a ground-truth CV->latency law (see BRIEF.md section 8).

The whole point of this module is to bake in the relationship the proposal
*hypothesises* so the decoder can be exercised in-silico: a subject's conduction
velocity sets the width of the temporal window over which the latent "intent"
signal drives kinematics. Faster CV  ->  narrower, more recent window (lower
latency); slower CV  ->  wider window (integrate longer). A decoder that uses the
right window decodes best.

This is a correctness/plausibility testbed, NOT evidence for the hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d

FS = 250          # Hz, after resampling (proposal 4.5)
N_CHAN = 64       # EEG channels
WIN = 50          # timepoints per window = 200 ms @ 250 Hz
N_KIN = 6         # 6-DOF wrist kinematics
D_LATENT = 3      # dimensionality of the latent motor "intent"

# CV -> integration width (samples). 20-100 ms tau range = 5-25 samples @ 250 Hz,
# which is also the model's tau_min/tau_max span, so oracle width and gate tau
# live on the same scale.
CV_LO, CV_HI = 25.0, 70.0     # m/s, spanning the plausible CST range
W_MIN, W_MAX = 5, 25          # samples


def cv_to_width(cv: float) -> int:
    """Faster CV -> narrower (more recent) integration window."""
    frac = np.clip((cv - CV_LO) / (CV_HI - CV_LO), 0.0, 1.0)
    return int(round(W_MAX - frac * (W_MAX - W_MIN)))


def sample_cvs(n: int, *, mean: float = 45.0, sd: float = 9.0,
               lo: float = CV_LO, hi: float = CV_HI, seed: int = 0) -> np.ndarray:
    """Per-subject CVs with realistic spread (sd/mean = 20% >= H1's 15% CoV).

    Normally distributed, so most subjects cluster mid-range — representative of a
    real cohort, but it under-exercises the CV mechanism (fast/slow tails are rare).
    """
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, sd, n), lo, hi)


def spread_cvs(n: int, *, lo: float = 28.0, hi: float = 68.0, seed: int = 0) -> np.ndarray:
    """CVs spanning the plausible range with light jitter — evaluates the decoder
    across the whole CV landscape (fast and slow tracts), as the g-ratio atlas does.
    """
    rng = np.random.default_rng(seed)
    base = np.linspace(lo, hi, n)
    return np.clip(base + rng.normal(0.0, (hi - lo) / (4 * n), n), CV_LO, CV_HI)


def _causal_boxcar(x: np.ndarray, width: int) -> np.ndarray:
    """Causal moving average of width `width` along axis 1. x: (d, L)."""
    d, L = x.shape
    c = np.cumsum(x, axis=1)
    out = np.empty_like(x)
    for t in range(L):
        lo = max(0, t - width + 1)
        s = c[:, t] - (c[:, lo - 1] if lo > 0 else 0.0)
        out[:, t] = s / (t - lo + 1)
    return out


def _gen_continuous(A: np.ndarray, B: np.ndarray, width: int, L: int,
                    rng: np.random.Generator, eeg_noise: float,
                    kin_noise: float) -> tuple[np.ndarray, np.ndarray]:
    """One continuous segment. Returns eeg (C, L), kin (N_KIN, L).

    intent : smooth latent trajectory (D_LATENT, L)
    eeg[:,t] = A @ intent[:,t]                 + noise   (per-timepoint encoding)
    kin[:,t] = B @ boxcar(intent, width)[:,t]  + noise   (CV-windowed readout)
    """
    intent = gaussian_filter1d(rng.standard_normal((D_LATENT, L)), sigma=3.0, axis=1)
    intent /= intent.std(axis=1, keepdims=True) + 1e-8
    eeg = A @ intent + rng.normal(0.0, eeg_noise, (A.shape[0], L))
    kin = B @ _causal_boxcar(intent, width) + rng.normal(0.0, kin_noise, (B.shape[0], L))
    return eeg.astype(np.float32), kin.astype(np.float32)


def _windows(eeg: np.ndarray, kin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slice a continuous segment into overlapping windows.

    X: (n, C, WIN); Y: (n, N_KIN) — target = kinematics at each window's end.
    """
    xs = sliding_window_view(eeg, WIN, axis=1)          # (C, n, WIN)
    X = np.transpose(xs, (1, 0, 2)).copy()              # (n, C, WIN)
    Y = kin[:, WIN - 1:].T.copy()                       # (n, N_KIN)
    return X.astype(np.float32), Y.astype(np.float32)


@dataclass
class SubjectData:
    cv: float
    width: int                 # ground-truth integration width (samples)
    A: np.ndarray              # (C, D_LATENT) EEG mixing
    B: np.ndarray              # (N_KIN, D_LATENT) kinematics readout
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    cont_eeg: np.ndarray       # (C, L) held-out continuous segment for latency
    cont_kin: np.ndarray       # (N_KIN, L)


def make_subject(cv: float, *, n_train: int = 600, n_test: int = 200,
                 cont_len: int = 400, eeg_noise: float = 1.0,
                 kin_noise: float = 0.05, seed: int = 0) -> SubjectData:
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((N_CHAN, D_LATENT)) / np.sqrt(D_LATENT)).astype(np.float32)
    B = rng.standard_normal((N_KIN, D_LATENT)).astype(np.float32)
    width = cv_to_width(cv)

    e_tr, k_tr = _gen_continuous(A, B, width, n_train + WIN - 1, rng, eeg_noise, kin_noise)
    e_te, k_te = _gen_continuous(A, B, width, n_test + WIN - 1, rng, eeg_noise, kin_noise)
    e_co, k_co = _gen_continuous(A, B, width, cont_len, rng, eeg_noise, kin_noise)

    X_tr, Y_tr = _windows(e_tr, k_tr)
    X_te, Y_te = _windows(e_te, k_te)
    return SubjectData(cv=float(cv), width=width, A=A, B=B,
                       X_train=X_tr, Y_train=Y_tr, X_test=X_te, Y_test=Y_te,
                       cont_eeg=e_co, cont_kin=k_co)


def oracle_predict(X: np.ndarray, A: np.ndarray, B: np.ndarray, width: int) -> np.ndarray:
    """Best-case linear readout given a chosen integration `width`.

    Recovers intent per-timepoint (pinv A), causally averages over `width`, maps
    through B, reads out the window end. Correct width  ->  lowest MSE. Used to
    prove the synthetic data actually encodes exploitable CV structure — no
    training involved.
    """
    Ap = np.linalg.pinv(A)                       # (D, C)
    preds = np.empty((X.shape[0], B.shape[0]), dtype=np.float32)
    for i in range(X.shape[0]):
        intent_hat = Ap @ X[i]                   # (D, WIN)
        box = _causal_boxcar(intent_hat, width)  # (D, WIN)
        preds[i] = B @ box[:, -1]
    return preds


def _selfcheck() -> None:
    # width decreases with CV
    assert cv_to_width(30) > cv_to_width(65), "faster CV must give a narrower window"

    # CoV of a sampled cohort clears H1's 15% bar
    cvs = sample_cvs(30, seed=1)
    assert cvs.std() / cvs.mean() >= 0.15, "cohort CoV should be >= 15%"

    s = make_subject(cv=50.0, seed=2)
    assert s.X_train.shape == (600, N_CHAN, WIN), s.X_train.shape
    assert s.Y_train.shape == (600, N_KIN), s.Y_train.shape

    # The CV structure is exploitable: per-subject *correct* width beats a single
    # fixed width, averaged over a cohort with varied CV.
    correct, fixed = [], []
    for i, cv in enumerate(sample_cvs(8, seed=3)):
        sub = make_subject(cv, n_train=50, n_test=200, seed=100 + i)
        correct.append(_mse(oracle_predict(sub.X_test, sub.A, sub.B, sub.width), sub.Y_test))
        fixed.append(_mse(oracle_predict(sub.X_test, sub.A, sub.B, W_MAX), sub.Y_test))
    assert np.mean(correct) < np.mean(fixed), (np.mean(correct), np.mean(fixed))
    print(f"data.py self-check OK: correct-width MSE {np.mean(correct):.4f} "
          f"< fixed-width MSE {np.mean(fixed):.4f}")


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


if __name__ == "__main__":
    _selfcheck()
