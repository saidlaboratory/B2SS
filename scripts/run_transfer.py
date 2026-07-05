#!/usr/bin/env python3
"""Cross-subject / zero-shot transfer (Phase 9) — the one regime not yet excluded.

Within-subject, a decoder just LEARNS the conduction delays from data, so measured CV
adds nothing (Phase 8). But a decoder trained on SOURCE subjects and applied
zero-shot to a held-out TARGET cannot learn the target's delays — it never sees the
target's training data. So IF subjects differ in conduction delay, telling the model
the target's measured CV (to normalise it into a common frame) should enable transfer
that fails without it. This is the decisive test of whether measured CV has any value.

Controlled proof-of-mechanism on REAL MC_Maze spikes: split the recording into K
disjoint pseudo-subjects (they differ in trial content), inject a KNOWN distinct
per-subject conduction delay into each, then leave-one-subject-out transfer:
  * NONE     — train on source subjects' (delayed) data, zero-shot the (delayed) target.
  * CV-ALIGN — align every subject by its measured delays into a common frame first.
Metric: zero-shot velocity R² on the held-out subject. If CV-ALIGN >> NONE, measured
CV enables cross-subject transfer.

HONEST SCOPE: the cross-subject conduction difference here is *injected and known*, so
this is an upper bound / proof-of-mechanism — real inter-subject differences are
richer than a delay, and a real measured CV is imperfect. It shows the mechanism can
work in the transfer regime, not that it will on real cohorts.

    python scripts/run_transfer.py            # ~30-50 min CPU
    python scripts/run_transfer.py --quick
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

from b2ss.intracortical import (load_maze, make_windows, inject_group_latency,
                                shift_channels, MazeData)
from b2ss.model import DecoderConfig, B2SSDecoder, count_params
from b2ss.baselines import GRUDecoder
from b2ss.stats import mean_ci
from b2ss.train import fit, predict

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20


def subject_windows(spikes, vel, fs, train_frac=0.8):
    """One pseudo-subject's windows with an internal time-split (0=train, 1=eval)."""
    n = spikes.shape[0]
    split = np.where(np.arange(n) < int(n * train_frac), 0, 1).astype(np.int8)
    X, Y, sp, _ = make_windows(MazeData(spikes, vel, split, np.zeros(n, np.float32), fs), WIN)
    return X, Y, sp


def b2cfg(n_chan):
    return DecoderConfig(n_chan=n_chan, win=WIN, fs=50, patch=1, d_model=64, nhead=4,
                         num_layers=2, dropout=0.2, task="regression", n_out=2,
                         gate_mode="none", align_mode="none")


def make_model(kind, n_chan):
    return GRUDecoder(n_chan) if kind == "gru" else B2SSDecoder(b2cfg(n_chan))


def zc(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def transfer_r2(kind, Xtr, Ytr, Xte, Yte, epochs, seed):
    m = make_model(kind, Xtr.shape[1])
    fit(m, Xtr, Ytr, epochs=epochs, lr=1e-3, batch_size=256, seed=seed)
    return float(r2_score(Yte, predict(m, Xte), multioutput="variance_weighted"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=5)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--max-delay", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.subjects, args.epochs, args.seeds = 3, 20, 1
    RESULTS.mkdir(exist_ok=True)

    d = load_maze()
    n_chan = d.spikes.shape[1]
    T = d.spikes.shape[0]
    bounds = np.linspace(0, T, args.subjects + 1).astype(int)

    # Build K pseudo-subjects: disjoint segments + a distinct injected conduction delay.
    subs = []
    for k in range(args.subjects):
        sl = slice(bounds[k], bounds[k + 1])
        seg, vel = d.spikes[sl], d.vel[sl]
        inj, align = inject_group_latency(seg, n_groups=args.groups,
                                          max_delay_bins=args.max_delay, seed=100 + k)
        aligned = shift_channels(inj, align)                 # measured-CV alignment -> common frame
        subs.append({"none": inj, "cv": aligned, "vel": vel})

    print(f"\nCross-subject zero-shot transfer | {args.subjects} pseudo-subjects, "
          f"{n_chan} units, δ∈[0,{args.max_delay}] | B2SS {count_params(B2SSDecoder(b2cfg(n_chan))):,} params\n")

    # window every subject under both conditions once
    W = {c: [] for c in ("none", "cv")}
    for s in subs:
        for c in ("none", "cv"):
            W[c].append(subject_windows(s[c], s["vel"], d.fs))

    kinds = ["gru", "b2ss"]
    res = {f"{kind}-{cond}": [] for kind in kinds for cond in ("none", "cv")}
    for seed in range(args.seeds):
        for k in range(args.subjects):                       # leave-one-subject-out
            for cond in ("none", "cv"):
                Xtr = np.concatenate([W[cond][j][0][W[cond][j][2] == 0] for j in range(args.subjects) if j != k])
                Ytr = np.concatenate([W[cond][j][1][W[cond][j][2] == 0] for j in range(args.subjects) if j != k])
                Xk, Yk, spk = W[cond][k]
                Xte, Yte = Xk[spk == 1], Yk[spk == 1]        # zero-shot: held-out subject's eval
                mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
                ymu, ysd = Ytr.mean(0), Ytr.std(0) + 1e-6
                Xtr_z, Ytr_z = zc(Xtr, mu, sd), zc(Ytr, ymu, ysd)
                Xte_z, Yte_z = zc(Xte, mu, sd), zc(Yte, ymu, ysd)
                for kind in kinds:
                    r = transfer_r2(kind, Xtr_z, Ytr_z, Xte_z, Yte_z, args.epochs, seed)
                    res[f"{kind}-{cond}"].append(r)
        print(f"seed {seed} done: " + "  ".join(
            f"{m}={np.mean(res[m]):.3f}" for m in res))

    ci = {m: mean_ci(res[m]) for m in res}
    print(f"\nZero-shot transfer velocity R² — mean [95% CI] over {args.subjects} LOSO folds "
          f"× {args.seeds} seeds\n")
    for m in res:
        mn, lo, hi = ci[m]
        print(f"  {m:>10}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")

    gru_gain = ci["gru-cv"][0] - ci["gru-none"][0]
    b2_gain = ci["b2ss-cv"][0] - ci["b2ss-none"][0]
    print(f"\n  measured-CV alignment gain (cv − none): gru {gru_gain:+.3f}   b2ss {b2_gain:+.3f}")
    verdict = ("measured-CV alignment ENABLES cross-subject transfer"
               if gru_gain > 0.05 or b2_gain > 0.05 else
               "measured-CV alignment does not materially help transfer here")
    print(f"  VERDICT: {verdict}")

    plt.figure(figsize=(6, 4))
    order = ["gru-none", "gru-cv", "b2ss-none", "b2ss-cv"]
    xs = np.arange(len(order))
    means = [ci[m][0] for m in order]
    err = [[ci[m][0]-ci[m][1] for m in order], [ci[m][2]-ci[m][0] for m in order]]
    plt.bar(xs, means, yerr=err, capsize=4,
            color=["#bbb", "#4c9", "#bbb", "#4c9"])
    plt.xticks(xs, order, rotation=15); plt.ylabel("zero-shot transfer R²")
    plt.title(f"Cross-subject transfer ({args.subjects} pseudo-subjects, injected CV)")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig = RESULTS / "transfer.png"; plt.savefig(fig, dpi=120); plt.close()

    (RESULTS / "transfer.json").write_text(json.dumps({
        "subjects": args.subjects, "groups": args.groups, "max_delay_bins": args.max_delay,
        "epochs": args.epochs, "seeds": args.seeds,
        "transfer_r2_mean_ci": {m: ci[m] for m in res},
        "cv_gain_gru": gru_gain, "cv_gain_b2ss": b2_gain,
    }, indent=2))
    print(f"\nfigure: {fig.name}; data: results/transfer.json\n")


if __name__ == "__main__":
    main()
