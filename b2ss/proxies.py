"""A CV proxy from EEG alone — no MRI, no TMS (the "adoptable by any lab" path).

The sensorimotor mu-rhythm peak frequency is used as a non-invasive, indirect
proxy for conduction velocity. The rationale: Nunez's global standing-wave theory
predicts resonant EEG frequency proportional to cortico-cortical conduction
velocity over loop length (f ~ nu/L), and Valdes-Hernandez et al. (2010) found
individual peak alpha frequency correlates with white-matter fractional
anisotropy. IMPORTANT: this link is real but *weak-to-moderate, correlational,
and mechanistically ambiguous* (alpha frequency is also set by thalamocortical
membrane dynamics). Treat the proxy as an indirect surrogate, not a calibrated
per-person CV. See BACKGROUND.md.

Method: a lightweight version of Corcoran et al. (2018) restingIAF — Welch PSD
over sensorimotor channels, remove the 1/f background, pick the 7-13 Hz peak with
parabolic refinement.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .model import CV_POP_MEAN, CV_POP_SCALE

MU_BAND = (7.0, 13.0)


def mu_peak_frequency(eeg: np.ndarray, fs: float, band=MU_BAND,
                      fit_range=(2.0, 40.0)) -> float:
    """Peak frequency (Hz) in `band` after removing the 1/f background.

    eeg: (channels, time) — pass sensorimotor channels (e.g. C3/Cz/C4). Returns
    NaN if no clear in-band peak is found.
    """
    eeg = np.atleast_2d(np.asarray(eeg, float))
    nper = int(min(eeg.shape[1], fs * 2))
    f, pxx = welch(eeg, fs=fs, nperseg=nper, noverlap=nper // 2, axis=1)
    psd = pxx.mean(0)

    fit = (f >= fit_range[0]) & (f <= fit_range[1]) & (f > 0)
    logf, logp = np.log10(f[fit]), np.log10(psd[fit] + 1e-20)
    slope, intercept = np.polyfit(logf, logp, 1)          # 1/f background
    resid = np.full_like(f, -np.inf)
    resid[fit] = logp - (slope * logf + intercept)

    in_band = (f >= band[0]) & (f <= band[1]) & fit
    if not in_band.any():
        return float("nan")
    idx = np.where(in_band)[0]
    k = idx[np.argmax(resid[idx])]
    if k <= 0 or k >= len(f) - 1 or resid[k] <= 0:         # must beat background, interior
        return float("nan")
    # parabolic interpolation on the residual peak
    y0, y1, y2 = resid[k - 1], resid[k], resid[k + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(f[k] + delta * (f[1] - f[0]))


def frequencies_to_pseudo_cv(freqs) -> np.ndarray:
    """Map per-subject mu peak frequencies onto the CV scale the gate expects.

    Monotone (higher frequency -> higher pseudo-CV), via a z-score across the
    cohort placed on the CV population mean/scale. This is a proxy mapping, not a
    physical CV in m/s. NaNs (no peak found) fall back to the population mean.
    """
    f = np.asarray(freqs, float)
    good = np.isfinite(f)
    if good.sum() < 2:
        return np.full_like(f, CV_POP_MEAN)
    mu, sd = f[good].mean(), f[good].std()
    z = np.zeros_like(f)
    z[good] = (f[good] - mu) / (sd + 1e-8)
    return CV_POP_MEAN + z * CV_POP_SCALE


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    fs, T = 160.0, 160 * 8
    t = np.arange(T) / fs
    # 1/f-ish background + a 10.5 Hz oscillation on 3 channels
    def make(fpeak):
        bg = np.cumsum(rng.standard_normal((3, T)), axis=1) * 0.02
        return bg + 0.5 * np.sin(2 * np.pi * fpeak * t)[None, :]
    assert abs(mu_peak_frequency(make(10.5), fs) - 10.5) < 0.5

    # monotonicity: faster peak -> higher pseudo-CV
    peaks = np.array([9.0, 10.0, 11.0, 12.0])
    pcv = frequencies_to_pseudo_cv(peaks)
    assert np.all(np.diff(pcv) > 0), pcv
    # NaN handling
    pcv2 = frequencies_to_pseudo_cv([9.0, np.nan, 12.0])
    assert np.isfinite(pcv2).all()
    print(f"proxies.py self-check OK: recovered mu peak ~10.5 Hz; "
          f"pseudo-CV spans {pcv.min():.1f}-{pcv.max():.1f} m/s")


if __name__ == "__main__":
    _selfcheck()
