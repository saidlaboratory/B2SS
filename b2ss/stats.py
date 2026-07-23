"""Statistical harness for the pre-registered analysis plan (proposal §6).

Implements the confirmatory machinery so the plan is runnable, and corrects the
power analysis using the *verified* effect sizes (Clark 2022 is r=0.18 ~ d 0.37,
not d 0.45 — see BACKGROUND.md):

  - paired-design power / required N via the noncentral-t distribution
  - ICC(2,1) two-way-random absolute-agreement reliability (H1 test-retest)
  - linear mixed model optimal_tau ~ CV + (1|subject) with marginal Delta R^2 + LRT (H3)
  - Bonferroni (H4) and Benjamini-Hochberg FDR (H5, H6) corrections
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Power (within-subjects / paired)
# --------------------------------------------------------------------------- #
def mean_ci(x, ci: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) via Student-t interval. For aggregating across seeds/subjects."""
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = x.std(ddof=1) / np.sqrt(n)
    h = float(stats.t.ppf(1 - (1 - ci) / 2, n - 1) * se)
    return m, m - h, m + h


def power_paired(d: float, n: int, alpha: float = 0.05, two_sided: bool = True) -> float:
    """Power of a paired t-test: noncentral t, df=n-1, ncp = d*sqrt(n)."""
    df, ncp = n - 1, d * np.sqrt(n)
    if two_sided:
        tc = stats.t.ppf(1 - alpha / 2, df)
        return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    tc = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(tc, df, ncp))


def required_n_paired(d: float, alpha: float = 0.05, power: float = 0.80,
                      two_sided: bool = True, n_max: int = 5000) -> int:
    """Smallest N giving >= `power` for a paired t-test at effect size d."""
    for n in range(3, n_max):
        if power_paired(d, n, alpha, two_sided) >= power:
            return n
    return n_max


def power_table() -> list[dict]:
    """Required N for the effect sizes in play, at alpha .05 and Bonferroni .0167.
    The corrected Clark effect (d=0.37) needs ~60-79 subjects — the proposal's
    N=30 is underpowered for it; the optimistic d=0.55 needs ~28-38."""
    rows = []
    for label, d in [("proposal d=0.45 (overstated)", 0.45),
                     ("corrected Clark d=0.37", 0.37),
                     ("target d=0.55", 0.55)]:
        rows.append({
            "effect": label, "d": d,
            "N @a=0.05": required_n_paired(d, 0.05),
            "N @a=0.0167 (Bonf 3)": required_n_paired(d, 0.05 / 3),
            "power @N=30, a=.05": round(power_paired(d, 30, 0.05), 2),
        })
    return rows


# --------------------------------------------------------------------------- #
# Reliability: ICC(2,1) two-way random, absolute agreement, single measure
# --------------------------------------------------------------------------- #
def icc21(table: np.ndarray) -> dict:
    """table: (n_subjects, k_sessions). Returns ICC(2,1), ICC(2,k), and a
    bootstrap 95% CI over subjects (Koo & Li interpret via the CI, not the point)."""
    x = np.asarray(table, float)
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)                 # rows/subjects
    ms_c = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)                 # cols/sessions
    ss_e = ((x - grand) ** 2).sum() - k * ((x.mean(1) - grand) ** 2).sum() \
        - n * ((x.mean(0) - grand) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    icc1 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + (k / n) * (ms_c - ms_e))
    icck = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        try:
            boots.append(icc21_point(x[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"icc21": float(icc1), "icc2k": float(icck),
            "ci95": (float(lo), float(hi)), "ms_r": ms_r, "ms_c": ms_c, "ms_e": ms_e}


def icc21_point(x: np.ndarray) -> float:
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    ss_e = ((x - grand) ** 2).sum() - k * ((x.mean(1) - grand) ** 2).sum() \
        - n * ((x.mean(0) - grand) ** 2).sum()
    ms_e = ss_e / ((n - 1) * (k - 1))
    return (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + (k / n) * (ms_c - ms_e))


def paired_by_unit(a, b, n_units: int) -> dict:
    """Paired comparison of two methods over the EXPERIMENTAL UNIT, averaging seeds away.

    `a`, `b` are flat result lists in (seed-major, unit-minor) order — the layout the
    Indy scripts accumulate, `n_units` values per seed. Seeds are a nuisance dimension:
    re-running the same sessions under a different init does not produce new sessions, so
    pooling seed x unit as if they were independent draws inflates n and shrinks the CI.
    Average within unit, then test across units.

    Returns mean difference, paired-t p, units won, and the CI of the difference.
    """
    a = np.asarray(a, float).reshape(-1, n_units).mean(0)
    b = np.asarray(b, float).reshape(-1, n_units).mean(0)
    d = a - b
    p = float(stats.ttest_rel(a, b).pvalue) if n_units > 1 and d.std() > 0 else 1.0
    m, lo, hi = mean_ci(d)
    return {"delta": m, "ci": [lo, hi], "p": p,
            "won": int((d > 0).sum()), "n": int(n_units)}


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #
def bonferroni(pvals, alpha: float = 0.05) -> dict:
    p = np.asarray(pvals, float)
    return {"reject": p <= alpha / len(p), "alpha_each": alpha / len(p)}


def benjamini_hochberg(pvals, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thresh
    reject = np.zeros(m, bool)
    if passed.any():
        kmax = np.max(np.where(passed))
        reject[order[:kmax + 1]] = True
    return reject


# --------------------------------------------------------------------------- #
# H3 mixed model: optimal_tau ~ CV + (1|subject), marginal Delta R^2 + LRT
# --------------------------------------------------------------------------- #
def mixed_delta_r2(df, outcome: str = "optimal_tau", predictor: str = "cv",
                   group: str = "subject") -> dict:
    """Nakagawa marginal Delta R^2 for `predictor`, plus a likelihood-ratio test."""
    import statsmodels.formula.api as smf

    full = smf.mixedlm(f"{outcome} ~ {predictor}", df, groups=df[group]).fit(reml=False)
    null = smf.mixedlm(f"{outcome} ~ 1", df, groups=df[group]).fit(reml=False)

    def marginal_r2(res):
        var_f = np.var(res.model.exog @ res.fe_params, ddof=0)
        var_u = float(res.cov_re.iloc[0, 0])
        return var_f / (var_f + var_u + res.scale)

    lr = 2 * (full.llf - null.llf)
    return {"delta_r2": marginal_r2(full) - marginal_r2(null),
            "marginal_r2_full": marginal_r2(full),
            "lrt_chi2": float(lr), "lrt_p": float(stats.chi2.sf(lr, df=1)),
            "beta_cv": float(full.fe_params[predictor])}


def _selfcheck() -> None:
    # power: matches the verified spec (d=0.37 -> 60, d=0.55 -> 28 at a=.05)
    assert required_n_paired(0.37, 0.05) == 60, required_n_paired(0.37, 0.05)
    assert required_n_paired(0.55, 0.05) == 28, required_n_paired(0.55, 0.05)
    assert required_n_paired(0.5, 0.05) == 34  # Cohen's table check
    assert 0.49 < power_paired(0.37, 30, 0.05) < 0.51

    # ICC: high reliability when subjects differ a lot, sessions barely
    rng = np.random.default_rng(0)
    trait = rng.normal(50, 10, 30)[:, None]
    table = trait + rng.normal(0, 1.0, (30, 2))    # tiny within-subject noise
    r = icc21(table)
    assert r["icc21"] > 0.9, r["icc21"]
    assert r["ci95"][0] <= r["icc21"] <= r["ci95"][1]

    # BH: with one tiny p and rest large, exactly the tiny one survives
    rej = benjamini_hochberg([0.001, 0.4, 0.6, 0.8], q=0.05)
    assert rej.tolist() == [True, False, False, False]

    # paired_by_unit: a small consistent per-unit gain is significant; seeds must not
    # inflate n (3 seeds x 4 units is still n=4), and the sign convention is a - b.
    units = np.array([0.40, 0.55, 0.31, 0.62])
    gain = np.array([0.01, 0.02, 0.04, 0.05])                   # effect varies BY UNIT
    A = np.concatenate([units + gain + rng.normal(0, 0.002, 4) for _ in range(3)])
    B = np.concatenate([units + rng.normal(0, 0.002, 4) for _ in range(3)])
    pb = paired_by_unit(A, B, 4)
    assert pb["n"] == 4 and pb["won"] == 4 and pb["p"] < 0.05
    assert abs(pb["delta"] - gain.mean()) < 0.005, pb
    assert paired_by_unit(B, A, 4)["delta"] < 0                 # antisymmetric
    # pooling seed x unit as n=12 triple-counts 4 sessions and shrinks the interval
    naive = mean_ci(A - B)
    assert (pb["ci"][1] - pb["ci"][0]) > 2 * (naive[2] - naive[1])

    # mixed model: CV that drives tau -> positive delta R^2, significant LRT
    import pandas as pd
    rows = []
    for s in range(20):
        u = rng.normal(0, 5)
        for _ in range(10):
            cv = rng.uniform(28, 68)
            rows.append({"subject": s, "cv": cv,
                         "optimal_tau": 100 - 0.8 * cv + u + rng.normal(0, 3)})
    res = mixed_delta_r2(pd.DataFrame(rows))
    assert res["delta_r2"] > 0.1 and res["lrt_p"] < 0.001, res
    print(f"stats.py self-check OK: N(d=.37)=60, N(d=.55)=28; ICC={r['icc21']:.3f} "
          f"CI{tuple(round(c,2) for c in r['ci95'])}; H3 dR2={res['delta_r2']:.2f} p={res['lrt_p']:.1e}")


if __name__ == "__main__":
    _selfcheck()
