#!/usr/bin/env python3
"""Phase 10.2 — the calibration-cost spectrum (controlled, faithful mechanism).

Real MC_Maze spikes → K pseudo-subjects, each with a distinct INJECTED per-tract
conduction delay. Canonicalise the source cohort to a common conduction frame,
train a decoder on it, FREEZE it. Then, per held-out target, vary ONLY the
target-side conduction alignment and measure velocity R²:

    no-norm            apply frozen decoder raw (naive transfer)
    zero-shot          align by the target's measured/known conduction (no data)
    unsupervised       fit delta to the target's UNLABELED data (moment-match)
    few-shot(n)        fit delta to n LABELED target trials
    full-retrain       train a fresh decoder on the target's own data (upper bound = the
                       calibration cost we are trying to avoid)
    free-delay(n)      few-shot with UNSTRUCTURED per-channel delays (ablation: does the
                       low-dim conduction grouping beat free delays?)

Headline: an accuracy-vs-target-labels curve — how much calibration does
conduction-normalisation save? Honest either way. Multi-seed, 95% CI.

    python scripts/run_transfer_modes.py            # ~30-45 min CPU
    python scripts/run_transfer_modes.py --quick
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
import torch

from b2ss.intracortical import load_maze, make_windows, MazeData
from b2ss.model import fractional_shift, DecoderConfig, B2SSDecoder
from b2ss.baselines import GRUDecoder
from b2ss.stats import mean_ci
from b2ss.train import fit, predict
from b2ss.transfer import (TransferNormalizer, set_measured, fit_supervised,
                           fit_unsupervised, source_feature_stats)

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20


def make_subjects(d: MazeData, K: int, n_groups: int, max_delay: int, seed: int):
    """Split the session into K time-blocks; inject a distinct per-group delay into
    each (real spikes, known ground-truth conduction). Returns per-subject windows
    (with train/eval split) + the injected per-group delays."""
    S, V, T, C = d.spikes, d.vel, d.spikes.shape[0], d.spikes.shape[1]
    gids = np.arange(C) % n_groups
    rng = np.random.default_rng(seed)
    bounds = np.linspace(0, T, K + 1).astype(int)
    subs = []
    for s in range(K):
        lo, hi = bounds[s], bounds[s + 1]
        Ss, Vs, sp = S[lo:hi], V[lo:hi], d.split[lo:hi]
        gd = rng.integers(0, max_delay + 1, n_groups).astype(np.float32)
        Sh = np.zeros_like(Ss)
        for c in range(C):
            dd = int(gd[gids[c]])
            Sh[dd:, c] = Ss[:len(Ss) - dd, c] if dd else 0
            if dd == 0:
                Sh[:, c] = Ss[:, c]
        X, Y, split, _ = make_windows(MazeData(Sh, Vs, sp, d.rt[lo:hi], d.fs), WIN)
        subs.append({"X": X, "Y": Y, "split": split, "gd": gd})
    return subs, gids


def zc(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def canonicalise(X, gd, n_groups):
    """Align a subject's windows to the zero-delay frame using its known delays."""
    C = X.shape[1]
    per_chan = torch.tensor(-gd[np.arange(C) % n_groups], dtype=torch.float32)
    with torch.no_grad():
        return fractional_shift(torch.tensor(X), per_chan).numpy().astype(np.float32)


def r2(model, X, Y):
    return float(r2_score(Y, predict(model, X), multioutput="variance_weighted"))


def r2_norm(norm, X, Y):
    with torch.no_grad():
        pred = norm(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    return float(r2_score(Y, pred, multioutput="variance_weighted"))


def new_gru(n_chan):
    return GRUDecoder(n_chan)


def run_fold(subs, gids, target, n_groups, max_delay, epochs, fewshot_ns, seed):
    C = subs[0]["X"].shape[1]
    src = [s for i, s in enumerate(subs) if i != target]
    tgt = subs[target]

    # source pool, canonicalised to the common frame, standardised on source stats
    Xs = np.concatenate([canonicalise(s["X"][s["split"] == 0], s["gd"], n_groups) for s in src])
    Ys = np.concatenate([s["Y"][s["split"] == 0] for s in src])
    xmu, xsd = Xs.mean((0, 2), keepdims=True), Xs.std((0, 2), keepdims=True) + 1e-6
    ymu, ysd = Ys.mean(0), Ys.std(0) + 1e-6
    Xs, Ys = zc(Xs, xmu, xsd), zc(Ys, ymu, ysd)

    decoder = new_gru(C)
    fit(decoder, Xs, Ys, epochs=epochs, lr=1e-3, batch_size=256, seed=seed)
    smean, svar = source_feature_stats(decoder, Xs)

    # target splits (raw, i.e. NOT pre-aligned — the normaliser must handle it)
    Xtr = zc(tgt["X"][tgt["split"] == 0], xmu, xsd)
    Ytr = zc(tgt["Y"][tgt["split"] == 0], ymu, ysd)
    Xte = zc(tgt["X"][tgt["split"] == 1], xmu, xsd)
    Yte = zc(tgt["Y"][tgt["split"] == 1], ymu, ysd)
    measured = -tgt["gd"]                                     # known undo delays (per group)

    out = {}
    def mk():
        return TransferNormalizer(decoder, C, n_groups=n_groups, max_delay=max_delay)

    out["no-norm"] = r2_norm(mk(), Xte, Yte)                  # delta = 0
    z = mk(); set_measured(z, measured); out["zero-shot"] = r2_norm(z, Xte, Yte)
    u = mk(); fit_unsupervised(u, Xtr, smean, svar, epochs=100, lr=0.1, seed=seed); out["unsup"] = r2_norm(u, Xte, Yte)
    for n in fewshot_ns:
        f = mk()
        idx = np.random.default_rng(seed).choice(len(Xtr), size=min(n, len(Xtr)), replace=False)
        fit_supervised(f, Xtr[idx], Ytr[idx], epochs=120, lr=0.1, seed=seed)
        out[f"few-{n}"] = r2_norm(f, Xte, Yte)
    # free-delay ablation: unstructured per-channel learned delays via few-shot at the largest n
    nmax = fewshot_ns[-1]
    free = TransferNormalizer(decoder, C, n_groups=C, max_delay=max_delay)   # K=C => per-channel
    idx = np.random.default_rng(seed).choice(len(Xtr), size=min(nmax, len(Xtr)), replace=False)
    fit_supervised(free, Xtr[idx], Ytr[idx], epochs=120, lr=0.1, seed=seed)
    out[f"free-delay-{nmax}"] = r2_norm(free, Xte, Yte)
    # full retrain on target's own data (calibration-cost upper bound)
    rt = new_gru(C); fit(rt, Xtr, Ytr, epochs=epochs, lr=1e-3, batch_size=256, seed=seed)
    out["full-retrain"] = r2(rt, Xte, Yte)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=4, help="K pseudo-subjects (LOSO)")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--max-delay", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    fewshot_ns = [5, 20, 100]
    if args.quick:
        args.subjects, args.epochs, args.seeds, fewshot_ns = 3, 20, 1, [20]
    RESULTS.mkdir(exist_ok=True)

    d = load_maze()
    order = ["no-norm", "zero-shot", "unsup"] + [f"few-{n}" for n in fewshot_ns] + \
            [f"free-delay-{fewshot_ns[-1]}", "full-retrain"]
    per = {m: [] for m in order}
    print(f"\nCalibration-cost spectrum | K={args.subjects} pseudo-subjects, "
          f"{args.groups} groups, δ∈[0,{args.max_delay}] | GRU backbone\n")
    for seed in range(args.seeds):
        subs, gids = make_subjects(d, args.subjects, args.groups, args.max_delay, seed)
        for t in range(args.subjects):
            o = run_fold(subs, gids, t, args.groups, args.max_delay, args.epochs, fewshot_ns, seed)
            for m in order:
                per[m].append(o[m])
        print(f"seed {seed}: " + "  ".join(f"{m}={np.mean(per[m][-args.subjects:]):.3f}" for m in ("no-norm", "zero-shot", "unsup", "full-retrain")))

    ci = {m: mean_ci(per[m]) for m in order}
    print(f"\nVelocity R² — mean [95% CI] over {args.subjects} folds × {args.seeds} seeds\n")
    for m in order:
        mn, lo, hi = ci[m]
        print(f"  {m:>16}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")
    gain = ci["zero-shot"][0] - ci["no-norm"][0]
    struct = ci[f"few-{fewshot_ns[-1]}"][0] - ci[f"free-delay-{fewshot_ns[-1]}"][0]
    print(f"\n  zero-shot gain over no-norm: {gain:+.3f}")
    print(f"  conduction-structure vs free-delay @n={fewshot_ns[-1]}: {struct:+.3f}")

    # calibration curve: R² vs target labels used (0 for zero/unsup, n for few, all for full)
    plt.figure(figsize=(6.8, 4))
    xs = [0] + fewshot_ns
    curve = [ci["zero-shot"][0]] + [ci[f"few-{n}"][0] for n in fewshot_ns]
    err = [[ci["zero-shot"][0]-ci["zero-shot"][1]] + [ci[f"few-{n}"][0]-ci[f"few-{n}"][1] for n in fewshot_ns],
           [ci["zero-shot"][2]-ci["zero-shot"][0]] + [ci[f"few-{n}"][2]-ci[f"few-{n}"][0] for n in fewshot_ns]]
    plt.errorbar(xs, curve, yerr=err, marker="o", capsize=3, label="conduction-norm")
    plt.axhline(ci["no-norm"][0], ls="--", c="gray", label="no-norm (naive transfer)")
    plt.axhline(ci["full-retrain"][0], ls=":", c="green", label="full retrain (calibration cost)")
    plt.axhline(ci["unsup"][0], ls="-.", c="purple", alpha=0.6, label="unsupervised")
    plt.xlabel("labeled target trials used"); plt.ylabel("transfer velocity R²")
    plt.title(f"Calibration-cost spectrum ({args.subjects} folds × {args.seeds} seeds)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    fig = RESULTS / "transfer_modes.png"; plt.savefig(fig, dpi=120); plt.close()

    (RESULTS / "transfer_modes.json").write_text(json.dumps({
        "K": args.subjects, "groups": args.groups, "max_delay": args.max_delay,
        "seeds": args.seeds, "fewshot_ns": fewshot_ns,
        "velocity_r2": {m: ci[m] for m in order},
        "zero_shot_gain_over_no_norm": gain, "structure_vs_free_delay": struct,
    }, indent=2))
    print(f"\nfigure: {fig.name}; data: results/transfer_modes.json\n")


if __name__ == "__main__":
    main()
