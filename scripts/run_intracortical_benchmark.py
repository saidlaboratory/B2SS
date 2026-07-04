#!/usr/bin/env python3
"""The decisive experiment (P2): B2SS as a CONTINUOUS kinematic decoder.

Decodes 2-D hand velocity from motor-cortex spikes (NLB MC_Maze_Small) — the
regression regime the architecture is actually built for. This is the fair test of
"is the decoder any good?", which the small-trial EEG classification could not be.

  * Competitiveness: B2SS vs Ridge (linear) and a GRU, by velocity R².
  * Gate (exploratory): does a per-trial context signal (reaction time) fed to the
    CV gate beat a learned constant window? Honest report either way — rt is a
    context proxy, not a measured conduction velocity.

    python scripts/run_intracortical_benchmark.py            # ~20-30 min CPU
    python scripts/run_intracortical_benchmark.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b2ss.intracortical import load_maze, make_windows
from b2ss.model import DecoderConfig, B2SSDecoder, count_params, CV_POP_MEAN, CV_POP_SCALE
from b2ss.baselines import GRUDecoder, ridge_r2
from b2ss.stats import mean_ci
from b2ss.train import fit, predict

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20                    # bins (400 ms @ 50 Hz), covers the ~100 ms neural->kin lag
B2SS_MODES = ("none", "learned", "cv")


def cfg(n_chan, mode):
    return DecoderConfig(n_chan=n_chan, win=WIN, fs=50, patch=1, d_model=64, nhead=4,
                         num_layers=2, dropout=0.2, task="regression", n_out=2, gate_mode=mode)


def zstat(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def r2(model, Xte, Yte, cv=None):
    return float(r2_score(Yte, predict(model, Xte, cv=cv), multioutput="variance_weighted"))


def run_seed(Xtr, Ytr, cvtr, Xte, Yte, cvte, n_chan, epochs, seed):
    out = {}
    for mode in B2SS_MODES:
        m = B2SSDecoder(cfg(n_chan, mode))
        c_tr = cvtr if mode == "cv" else None
        c_te = cvte if mode == "cv" else None
        fit(m, Xtr, Ytr, cv=c_tr, epochs=epochs, lr=1e-3, batch_size=256, seed=seed)
        out[f"b2ss-{mode}"] = r2(m, Xte, Yte, c_te)
    g = GRUDecoder(n_chan)
    fit(g, Xtr, Ytr, epochs=epochs, lr=1e-3, batch_size=256, seed=seed)
    out["gru"] = r2(g, Xte, Yte)
    out["ridge"] = ridge_r2(Xtr, Ytr, Xte, Yte)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.seeds = 15, 1
    RESULTS.mkdir(exist_ok=True)

    d = load_maze()
    X, Y, split, rt = make_windows(d, WIN)
    tr, te = split == 0, split == 1
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    # standardise spikes (per channel) and velocity with TRAIN stats (R² is invariant)
    xmu, xsd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
    ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
    Xtr_z, Xte_z = zstat(Xtr, xmu, xsd), zstat(Xte, xmu, xsd)
    Ytr_z, Yte_z = zstat(Ytr, ymu, ysd), zstat(Yte, ymu, ysd)

    # per-window context (reaction time) -> pseudo-CV on the gate's scale
    rt_tr, rt_te = rt[tr], rt[te]
    mu, sd = np.nanmean(rt_tr), np.nanstd(rt_tr) + 1e-6
    cvtr = CV_POP_MEAN + (rt_tr - mu) / sd * CV_POP_SCALE
    cvte = CV_POP_MEAN + (rt_te - mu) / sd * CV_POP_SCALE

    n_chan = X.shape[1]
    models = [f"b2ss-{m}" for m in B2SS_MODES] + ["gru", "ridge"]
    print(f"\nMC_Maze_Small velocity decoding | train {len(Xtr)} / val {len(Xte)} windows | "
          f"{n_chan} units | B2SS {count_params(B2SSDecoder(cfg(n_chan,'cv'))):,} params\n")

    per_seed = []
    for s in range(args.seeds):
        r = run_seed(Xtr_z, Ytr_z, cvtr, Xte_z, Yte_z, cvte, n_chan, args.epochs, s)
        per_seed.append(r)
        print(f"seed {s}: " + "  ".join(f"{k}={r[k]:.3f}" for k in models))

    ci = {m: mean_ci([ps[m] for ps in per_seed]) for m in models}
    print(f"\nVelocity R² — mean [95% CI] across {args.seeds} seeds (higher better)")
    for m in models:
        mn, lo, hi = ci[m]
        print(f"  {m:>12}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")

    gate = ci["b2ss-cv"][0] - ci["b2ss-learned"][0]
    comp = ci["b2ss-none"][0] - ci["gru"][0]
    print(f"\n  competitiveness: b2ss-none − gru = {comp:+.3f}; vs ridge = "
          f"{ci['b2ss-none'][0]-ci['ridge'][0]:+.3f}")
    print(f"  gate (exploratory): b2ss-cv − b2ss-learned = {gate:+.3f} "
          f"({'context helps' if gate>0 else 'no benefit from rt context'})")

    plt.figure(figsize=(6.5, 4))
    xs = np.arange(len(models))
    means = [ci[m][0] for m in models]
    err = [[ci[m][0]-ci[m][1] for m in models], [ci[m][2]-ci[m][0] for m in models]]
    plt.bar(xs, means, yerr=err, capsize=4)
    plt.xticks(xs, models, rotation=20); plt.ylabel("velocity R² (variance-weighted)")
    plt.title(f"MC_Maze_Small hand-velocity decoding ({args.seeds} seeds)")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig = RESULTS / "intracortical_benchmark.png"; plt.savefig(fig, dpi=120); plt.close()

    (RESULTS / "intracortical_benchmark.json").write_text(json.dumps({
        "dataset": "MC_Maze_Small (DANDI 000140)", "win_bins": WIN, "bin_ms": 20,
        "n_units": int(n_chan), "n_train": int(len(Xtr)), "n_val": int(len(Xte)),
        "epochs": args.epochs, "seeds": args.seeds,
        "velocity_r2_mean_ci": {m: ci[m] for m in models},
        "gate_minus_learned": gate, "b2ss_vs_gru": comp,
    }, indent=2))
    print(f"\nNote: NLB leaderboard ~0.90 R² uses 5 ms bins + LFADS-smoothed rates + a "
          f"tuned decoder on trial-aligned data; our 20 ms continuous setup is simpler "
          f"and will read lower. What matters is B2SS vs the same-input baselines.")
    print(f"figure: {fig.name}; data: results/intracortical_benchmark.json\n")


if __name__ == "__main__":
    main()
