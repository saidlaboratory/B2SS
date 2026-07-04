"""Decoding metrics (proposal 5.3): MSE, Pearson r, effective decoder latency."""

from __future__ import annotations

import numpy as np

from .data import FS


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Pearson correlation across kinematic dimensions. pred/true: (N, DOF)."""
    rs = []
    for d in range(pred.shape[1]):
        p, t = pred[:, d], true[:, d]
        if p.std() < 1e-8 or t.std() < 1e-8:
            continue
        rs.append(np.corrcoef(p, t)[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def _best_lag(p: np.ndarray, t: np.ndarray, max_lag: int) -> int:
    p = p - p.mean()
    t = t - t.mean()
    denom = (np.linalg.norm(p) * np.linalg.norm(t)) + 1e-12
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:                       # pred delayed by `lag`
            a, b = p[lag:], t[:len(t) - lag] if lag else t
        else:
            a, b = p[:len(p) + lag], t[-lag:]
        if len(a) < 2:
            continue
        c = float(np.dot(a, b) / denom)
        if c > best:
            best, best_lag = c, lag
    return best_lag


def xcorr_lag(pred_seq: np.ndarray, true_seq: np.ndarray, max_lag: int = 25) -> float:
    """Effective decoder latency = cross-correlation peak lag (samples).

    Positive => predicted kinematics lag behind the true kinematics. pred/true:
    (DOF, L) continuous decodes. Averaged over DOF.
    """
    lags = [_best_lag(pred_seq[d], true_seq[d], max_lag) for d in range(pred_seq.shape[0])]
    return float(np.mean(lags))


def samples_to_ms(samples: float) -> float:
    return samples / FS * 1000.0


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    true = rng.standard_normal((6, 300)).astype(np.float32)
    delay = 4
    pred = np.zeros_like(true)
    pred[:, delay:] = true[:, :-delay]          # pred is true delayed by 4 samples
    lag = xcorr_lag(pred, true, max_lag=15)
    assert abs(lag - delay) < 1e-6, lag

    # perfect prediction: r=1, mse=0
    y = rng.standard_normal((50, 6)).astype(np.float32)
    assert mse(y, y) == 0.0
    assert abs(pearson_r(y, y) - 1.0) < 1e-6
    print(f"eval.py self-check OK: recovered lag={lag} (true {delay}), "
          f"{samples_to_ms(lag):.1f} ms")


if __name__ == "__main__":
    _selfcheck()
