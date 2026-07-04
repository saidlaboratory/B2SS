"""Conduction-velocity estimation (proposal section 4.4).

Two independent estimators plus a Monte-Carlo bootstrap for uncertainty:

1. g-ratio -> CV via the proposal's stated relation  CV = k * v(g),
   v(g) = sqrt(1 - g**2) / g.  `k` is a physical calibration constant, tuned
   against human recordings (see BACKGROUND.md). Deriving CV from g-ratio follows
   the g-ratio conduction literature: Rushton (1951); Berman, Filo & Mezer (2019),
   "Modelling conduction delays ... using MRI-measured g-ratio"; Drakesmith et al.
   (2019). NB: the proposal's reference list cites the *other* Berman 2019 paper
   (corpus-callosum age/sex), which does not contain this formula — see
   BACKGROUND.md "Citation accuracy notes".
2. TMS-EEG combined estimate  CV = path_length / (mep_lat - cortical - nmj).

All velocities are in m/s, lengths in metres, times in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

NMJ_DELAY_S = 0.8e-3  # neuromuscular-junction delay, fixed (proposal 4.4)


# --------------------------------------------------------------------------- #
# g-ratio -> conduction velocity
# --------------------------------------------------------------------------- #
def velocity_factor(g: float | np.ndarray) -> np.ndarray:
    """Berman g-ratio velocity factor v(g) = sqrt(1 - g**2) / g.

    Monotonically *decreasing* on (0, 1): a g-ratio near 1 (little myelin) gives
    a small factor (slow), near 0 (heavy myelin) a large one.
    """
    g = np.asarray(g, dtype=float)
    if np.any((g <= 0) | (g >= 1)):
        raise ValueError("g-ratio must lie strictly in (0, 1)")
    return np.sqrt(1.0 - g**2) / g


def cv_from_gratio(g: float | np.ndarray, k: float = 40.0) -> np.ndarray:
    """CV = k * v(g). `k` (m/s) is calibrated against human recordings.

    Default k=40 places a typical CST g-ratio (~0.7) near ~40 m/s, in the
    plausible corticospinal range (tens of m/s).
    """
    return k * velocity_factor(g)


def cv_fixed_fiber_diameter(g: float | np.ndarray, fiber_diameter: float = 1.0,
                            k: float = 1.0) -> np.ndarray:
    """Classic Rushton (1951) CV for a *fixed fibre diameter*.

    CV proportional to (inner axon diameter) * sqrt(ln(1/g))
             = g * D * sqrt(-ln g)  (up to constant k).

    Unlike `velocity_factor`, this is peaked in g: it is maximised at
    g = exp(-1/2) ~= 0.607 — the textbook "optimal g-ratio". Kept as a
    biophysical sanity anchor (tested in tests/), not used by the pipeline.
    """
    g = np.asarray(g, dtype=float)
    if np.any((g <= 0) | (g >= 1)):
        raise ValueError("g-ratio must lie strictly in (0, 1)")
    return k * g * fiber_diameter * np.sqrt(-np.log(g))


# --------------------------------------------------------------------------- #
# TMS-EEG combined estimate
# --------------------------------------------------------------------------- #
def combined_cv(path_length_m: float, mep_latency_s: float,
                cortical_delay_s: float, nmj_delay_s: float = NMJ_DELAY_S) -> float:
    """CV = path length / effective conduction time.

    Effective time = MEP latency - cortical synaptic delay - NMJ delay.
    """
    effective_t = mep_latency_s - cortical_delay_s - nmj_delay_s
    if effective_t <= 0:
        raise ValueError(
            f"non-positive effective conduction time ({effective_t*1e3:.2f} ms); "
            "check latency vs. delay inputs")
    return path_length_m / effective_t


@dataclass(frozen=True)
class CVEstimate:
    mean: float
    ci_low: float
    ci_high: float
    sd: float

    def __str__(self) -> str:
        return (f"{self.mean:.1f} m/s "
                f"(95% CI {self.ci_low:.1f}-{self.ci_high:.1f}, SD {self.sd:.1f})")


def bootstrap_cv(path_length_m: float, mep_latencies_s: np.ndarray,
                 cortical_delay_s: float, *, path_length_sd_m: float = 0.0,
                 cortical_delay_sd_s: float = 0.0, n_resamples: int = 500,
                 ci: float = 0.95, seed: int | None = 0) -> CVEstimate:
    """Monte-Carlo bootstrap CV estimate with a confidence interval (proposal 4.4).

    Resamples the per-trial MEP latencies with replacement and (optionally)
    perturbs path length and cortical delay by their stated uncertainties,
    propagating everything into a per-subject CV distribution.
    """
    mep = np.asarray(mep_latencies_s, dtype=float)
    if mep.ndim != 1 or mep.size == 0:
        raise ValueError("mep_latencies_s must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)

    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        lat = rng.choice(mep, size=mep.size, replace=True).mean()
        plen = path_length_m + rng.normal(0.0, path_length_sd_m)
        cort = cortical_delay_s + rng.normal(0.0, cortical_delay_sd_s)
        try:
            samples[i] = combined_cv(plen, lat, cort)
        except ValueError:
            samples[i] = np.nan

    samples = samples[np.isfinite(samples)]
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return CVEstimate(
        mean=float(np.mean(samples)),
        ci_low=float(np.quantile(samples, lo)),
        ci_high=float(np.quantile(samples, hi)),
        sd=float(np.std(samples, ddof=1)),
    )


def _selfcheck() -> None:
    # Berman factor is monotonically decreasing in g.
    gs = np.linspace(0.4, 0.9, 20)
    vf = velocity_factor(gs)
    assert np.all(np.diff(vf) < 0), "v(g) must decrease with g"

    # Rushton fixed-diameter CV peaks at g = exp(-1/2) ~= 0.607.
    gg = np.linspace(0.3, 0.95, 4000)
    g_star = gg[int(np.argmax(cv_fixed_fiber_diameter(gg)))]
    assert abs(g_star - math.exp(-0.5)) < 0.02, f"peak g={g_star}, expected ~0.607"

    # Combined estimate: 0.4 m path, 20 ms MEP, 5 ms cortical, 0.8 ms NMJ.
    cv = combined_cv(0.40, 20e-3, 5e-3)
    assert 25 < cv < 35, cv  # 0.40 / (20-5-0.8)ms ~= 28 m/s

    # Bootstrap CI brackets the point estimate and is reproducible.
    lat = np.full(50, 20e-3) + np.random.default_rng(1).normal(0, 1e-3, 50)
    est = bootstrap_cv(0.40, lat, 5e-3, path_length_sd_m=0.02,
                       cortical_delay_sd_s=1e-3, seed=0)
    assert est.ci_low < est.mean < est.ci_high, est
    est2 = bootstrap_cv(0.40, lat, 5e-3, path_length_sd_m=0.02,
                        cortical_delay_sd_s=1e-3, seed=0)
    assert est.mean == est2.mean, "bootstrap must be reproducible under fixed seed"
    print("cv.py self-check OK:", est)


if __name__ == "__main__":
    _selfcheck()
