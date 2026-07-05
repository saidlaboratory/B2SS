#!/usr/bin/env python3
"""Injected-latency bridge (Phase 8) — a DATA-EFFICIENCY test of measured-CV alignment.

Take REAL MC_Maze spikes, inject a KNOWN fixed per-group conduction latency, then
ask: does giving a decoder the measured delays (align_mode='cv') help vs. not
aligning ('none') or learning them ('learned')?

Key lesson from a first pass: a *fixed* per-channel delay is trivially LEARNABLE
from abundant data (an unaligned decoder just absorbs it), so measured alignment can
only help where a prior helps — when data is scarce (or, ideally, cross-subject,
which one MC_Maze session can't provide). So we sweep training-set size and look for
a low-data advantage that shrinks as data grows. Honest either way; no forcing.

    python scripts/run_latency_bridge.py            # ~25-40 min CPU
    python scripts/run_latency_bridge.py --quick
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

from b2ss.intracortical import load_maze, make_windows, inject_group_latency, MazeData
from b2ss.model import DecoderConfig, B2SSDecoder, count_params
from b2ss.baselines import GRUDecoder
from b2ss.stats import mean_ci
from b2ss.train import fit, predict

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN, MAXD = 20, 12
VARIANTS = ("b2ss-none", "b2ss-learned", "b2ss-cv", "gru-none", "gru+cv-align")


def b2cfg(n_chan, align):
    return DecoderConfig(n_chan=n_chan, win=WIN, fs=50, patch=1, d_model=64, nhead=4,
                         num_layers=2, dropout=0.2, task="regression", n_out=2,
                         gate_mode="none", align_mode=align, max_delay_bins=MAXD)


def zc(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def build(model, Xtr, Ytr, Xte, Yte, epochs, seed, delays=None):
    fit(model, Xtr, Ytr, delays=delays, epochs=epochs, lr=1e-3, batch_size=256, seed=seed)
    return float(r2_score(Yte, predict(model, Xte, delays=delays),
                          multioutput="variance_weighted"))


def one(name, n_chan, Xtr, Ytr, Xte, Yte, epochs, seed, delays):
    if name == "b2ss-none":
        return build(B2SSDecoder(b2cfg(n_chan, "none")), Xtr, Ytr, Xte, Yte, epochs, seed)
    if name == "b2ss-learned":
        return build(B2SSDecoder(b2cfg(n_chan, "learned")), Xtr, Ytr, Xte, Yte, epochs, seed)
    if name == "b2ss-cv":
        return build(B2SSDecoder(b2cfg(n_chan, "cv")), Xtr, Ytr, Xte, Yte, epochs, seed, delays)
    if name == "gru-none":
        return build(GRUDecoder(n_chan), Xtr, Ytr, Xte, Yte, epochs, seed)
    return build(GRUDecoder(n_chan, align_mode="cv", max_delay_bins=MAXD),
                 Xtr, Ytr, Xte, Yte, epochs, seed, delays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--max-delay", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    sizes = [300, 3000] if args.quick else [200, 600, 2000, 10000]
    if args.quick:
        args.epochs, args.seeds = 20, 1
    RESULTS.mkdir(exist_ok=True)

    d = load_maze()
    n_chan = d.spikes.shape[1]
    inj_spikes, align_delays = inject_group_latency(
        d.spikes, n_groups=args.groups, max_delay_bins=args.max_delay, seed=0)
    X, Y, split, _ = make_windows(MazeData(inj_spikes, d.vel, d.split, d.rt, d.fs), WIN)
    tr, te = split == 0, split == 1
    Xtr_all, Ytr_all, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    xmu, xsd = Xtr_all.mean((0, 2), keepdims=True), Xtr_all.std((0, 2), keepdims=True) + 1e-6
    ymu, ysd = Ytr_all.mean(0), Ytr_all.std(0) + 1e-6
    Xtr_all, Ytr_all = zc(Xtr_all, xmu, xsd), zc(Ytr_all, ymu, ysd)
    Xte, Yte = zc(Xte, xmu, xsd), zc(Yte, ymu, ysd)
    sizes = [n for n in sizes if n <= len(Xtr_all)] or [len(Xtr_all)]

    print(f"\nInjected-latency data-efficiency bridge | {n_chan} units, {args.groups} groups, "
          f"δ∈[0,{args.max_delay}] | {len(Xtr_all)} train windows | "
          f"B2SS {count_params(B2SSDecoder(b2cfg(n_chan,'cv'))):,} params\n")

    # results[variant][size] = list over seeds
    res = {v: {n: [] for n in sizes} for v in VARIANTS}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for n in sizes:
            idx = rng.choice(len(Xtr_all), size=n, replace=False)
            Xs, Ys = Xtr_all[idx], Ytr_all[idx]
            for v in VARIANTS:
                res[v][n].append(one(v, n_chan, Xs, Ys, Xte, Yte, args.epochs, seed, align_delays))
        print(f"seed {seed} done: " + "  ".join(
            f"{v}@{sizes[0]}={res[v][sizes[0]][-1]:.3f}" for v in ("b2ss-none", "b2ss-cv")))

    ci = {v: {n: mean_ci(res[v][n]) for n in sizes} for v in VARIANTS}
    print(f"\nVelocity R² — mean [95% CI], {args.seeds} seeds. Gap = cv−none per family.\n")
    hdr = "size  " + "  ".join(f"{v:>13}" for v in VARIANTS)
    print(hdr)
    for n in sizes:
        print(f"{n:>5} " + "  ".join(f"{ci[v][n][0]:>13.3f}" for v in VARIANTS))
    print("\n  data-efficiency gaps (cv − none):")
    for n in sizes:
        gb = ci["b2ss-cv"][n][0] - ci["b2ss-none"][n][0]
        gg = ci["gru+cv-align"][n][0] - ci["gru-none"][n][0]
        print(f"    n={n:>5}: b2ss {gb:+.3f}   gru {gg:+.3f}")
    lo, hi = sizes[0], sizes[-1]
    b_lo = ci["b2ss-cv"][lo][0] - ci["b2ss-none"][lo][0]
    b_hi = ci["b2ss-cv"][hi][0] - ci["b2ss-none"][hi][0]
    verdict = ("measured-CV alignment helps at LOW data and fades with more (prior benefit)"
               if b_lo > 0.02 and b_lo > b_hi else
               "no data-efficiency benefit from measured-CV alignment — idea unsupported here")
    print(f"\n  VERDICT: {verdict}")

    plt.figure(figsize=(6.5, 4))
    for v in VARIANTS:
        plt.plot(sizes, [ci[v][n][0] for n in sizes], "o-", label=v)
    plt.xscale("log"); plt.xlabel("training windows"); plt.ylabel("velocity R²")
    plt.title(f"Injected-latency data-efficiency ({args.seeds} seeds)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    fig = RESULTS / "latency_bridge.png"; plt.savefig(fig, dpi=120); plt.close()

    (RESULTS / "latency_bridge.json").write_text(json.dumps({
        "n_groups": args.groups, "max_delay_bins": args.max_delay, "win": WIN,
        "sizes": sizes, "seeds": args.seeds, "epochs": args.epochs,
        "velocity_r2": {v: {int(n): ci[v][n] for n in sizes} for v in VARIANTS},
        "gap_cv_minus_none_lowdata": b_lo, "gap_cv_minus_none_highdata": b_hi,
    }, indent=2))
    print(f"\nNote: a fixed per-channel delay is learnable from abundant data, so any "
          f"benefit should appear at LOW data (prior) and fade — the honest test.")
    print(f"figure: {fig.name}; data: results/latency_bridge.json\n")


if __name__ == "__main__":
    main()
