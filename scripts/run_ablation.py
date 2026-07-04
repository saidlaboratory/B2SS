#!/usr/bin/env python3
"""Gate ablation — where should the integration window come from? (multi-seed)

Four gate modes (cv / learned / fixed / none) in two regimes:
  Study A (homogeneous, data-efficiency): CV constant per subject -> a learned
    constant CAN approach the CV-derived window, so the CV advantage is a PRIOR /
    data-efficiency benefit (largest at low data).
  Study B (heterogeneous): CV varies per trial -> a single constant window is
    provably suboptimal, so CV carries INFORMATION (gap persists with more data).

Runs multiple seeds and reports mean +/- 95% CI. Writes JSON + PNG (error bars).

    python scripts/run_ablation.py                 # 5 seeds
    python scripts/run_ablation.py --quick         # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b2ss.data import make_subject, make_heterogeneous, spread_cvs
from b2ss.eval import mse
from b2ss.model import DecoderConfig, B2SSDecoder, GATE_MODES
from b2ss.stats import mean_ci
from b2ss.train import fit, predict

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _train_eval(mode, X, Y, cv_tr, Xte, Yte, cv_te, epochs, seed):
    m = B2SSDecoder(DecoderConfig(gate_mode=mode))
    fit(m, X, Y, cv=cv_tr, epochs=epochs, seed=seed)
    return mse(predict(m, Xte, cv_te), Yte)


def study_a_seed(subjects, sizes, epochs, seed):
    cvs = spread_cvs(subjects, seed=seed)
    out = {m: {n: [] for n in sizes} for m in GATE_MODES}
    for n in sizes:
        for i, cv in enumerate(cvs):
            sub = make_subject(cv, n_train=n, n_test=200, seed=seed * 100 + i)
            for mode in GATE_MODES:
                out[mode][n].append(_train_eval(mode, sub.X_train, sub.Y_train, sub.cv,
                                                sub.X_test, sub.Y_test, sub.cv, epochs, seed))
    return {m: {n: float(np.mean(out[m][n])) for n in sizes} for m in GATE_MODES}


def study_b_seed(sizes, epochs, seed):
    out = {m: {n: None for n in sizes} for m in GATE_MODES}
    for n in sizes:
        het = make_heterogeneous(n_train=n, n_test=400, seed=seed)
        for mode in GATE_MODES:
            out[mode][n] = _train_eval(mode, het.X_train, het.Y_train, het.cv_train,
                                       het.X_test, het.Y_test, het.cv_test, epochs, seed)
    return out


def aggregate(per_seed, sizes):
    """per_seed: list of {mode:{n:mse}} -> {mode:{n:(mean,lo,hi)}}."""
    return {m: {n: mean_ci([s[m][n] for s in per_seed]) for n in sizes} for m in GATE_MODES}


def plot(agg, sizes, title, path):
    plt.figure(figsize=(6, 4))
    for m in GATE_MODES:
        means = [agg[m][n][0] for n in sizes]
        err = [[agg[m][n][0] - agg[m][n][1] for n in sizes],
               [agg[m][n][2] - agg[m][n][0] for n in sizes]]
        plt.errorbar(sizes, means, yerr=err, marker="o", capsize=3, label=m)
    plt.xscale("log"); plt.xlabel("training windows / subject"); plt.ylabel("test MSE")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sizes = [40, 160] if args.quick else [40, 80, 160, 320]
    subjects = 2 if args.quick else args.subjects
    seeds = list(range(2 if args.quick else args.seeds))
    RESULTS.mkdir(exist_ok=True)

    a_seeds, b_seeds = [], []
    for s in seeds:
        print(f"\n=== seed {s} ===")
        a = study_a_seed(subjects, sizes, args.epochs, s); a_seeds.append(a)
        print("  A: " + "  ".join(f"{m}={a[m][sizes[-1]]:.3f}" for m in GATE_MODES))
        b = study_b_seed(sizes, args.epochs, s); b_seeds.append(b)
        print("  B: " + "  ".join(f"{m}={b[m][sizes[-1]]:.3f}" for m in GATE_MODES))

    A, B = aggregate(a_seeds, sizes), aggregate(b_seeds, sizes)
    plot(A, sizes, f"Study A — homogeneous (prior benefit), {len(seeds)} seeds",
         RESULTS / "ablation_data_efficiency.png")
    plot(B, sizes, f"Study B — heterogeneous CV (information), {len(seeds)} seeds",
         RESULTS / "ablation_heterogeneous.png")

    nmax = sizes[-1]
    (RESULTS / "ablation.json").write_text(json.dumps({
        "seeds": len(seeds), "sizes": sizes,
        "study_a": {m: {int(n): A[m][n] for n in sizes} for m in GATE_MODES},
        "study_b": {m: {int(n): B[m][n] for n in sizes} for m in GATE_MODES},
    }, indent=2))

    def fmt(agg, n): return {m: agg[m][n] for m in GATE_MODES}
    print(f"\nSummary at n_train={nmax} (mean [95% CI], {len(seeds)} seeds)")
    for label, agg in (("A homogeneous", A), ("B heterogeneous", B)):
        print(f"  {label}:")
        for m in GATE_MODES:
            mn, lo, hi = agg[m][nmax]
            print(f"    {m:>8}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")
    a_gap = A["learned"][nmax][0] - A["cv"][nmax][0]
    b_gap = B["learned"][nmax][0] - B["cv"][nmax][0]
    a_lo = A["learned"][sizes[0]][0] - A["cv"][sizes[0]][0]
    print(f"\n  A cv-vs-learned gap: {a_lo:+.3f} @n={sizes[0]} -> {a_gap:+.3f} @n={nmax} "
          f"(homogeneous: CV a persistent prior; does not clearly shrink)")
    print(f"  B cv-vs-learned gap @n={nmax}: {b_gap:+.3f} (heterogeneous: CV is information)")
    print("  figures + results/ablation.json written.\n")


if __name__ == "__main__":
    main()
