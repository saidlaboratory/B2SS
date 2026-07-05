#!/usr/bin/env python3
"""Phase 10.3 — REAL cross-session transfer (no injection).

MC_Maze Small/Medium/Large are three recording sessions of the same monkey on
different days. Aggregating spikes per ELECTRODE gives 67 channels that correspond
across sessions, so we can test transfer of a decoder trained on some sessions to a
held-out session — a genuine (non-injected) distribution shift.

No measured CV exists here, so only the data-driven modes apply: fit the conduction
delta from the target's few-shot labeled or unlabeled data. HONEST CAVEAT: real
cross-session shift is more than conduction timing (firing-rate drift, unit
turnover, tuning changes), so the conduction-delay normaliser addresses only one
axis — a modest or null result is expected and is the honest bound.

    python scripts/run_xsession.py            # ~20-40 min CPU
    python scripts/run_xsession.py --quick
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

from b2ss.intracortical import common_electrodes, load_session, make_windows
from b2ss.baselines import GRUDecoder
from b2ss.stats import mean_ci
from b2ss.train import fit, predict
from b2ss.transfer import (TransferNormalizer, fit_supervised, fit_unsupervised,
                           source_feature_stats)

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN, N_GROUPS, MAXD = 20, 8, 12
SESSIONS = ["small", "medium", "large"]


def zc(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def r2(model, X, Y):
    return float(r2_score(Y, predict(model, X), multioutput="variance_weighted"))


def r2_norm(norm, X, Y):
    with torch.no_grad():
        pred = norm(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    return float(r2_score(Y, pred, multioutput="variance_weighted"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--src-cap", type=int, default=15000, help="cap source windows for speed")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    fewshot_ns = [20, 100]
    if args.quick:
        args.epochs, args.seeds, args.src_cap, fewshot_ns = 15, 1, 4000, [50]
    RESULTS.mkdir(exist_ok=True)

    keep = common_electrodes()
    print(f"\nReal cross-session transfer | {len(keep)} common electrodes | "
          f"sessions {SESSIONS} (same monkey, different days)\n")
    win = {}
    for n in SESSIONS:
        d = load_session(n, keep)
        X, Y, split, _ = make_windows(d, WIN)
        win[n] = {"Xtr": X[split == 0], "Ytr": Y[split == 0],
                  "Xte": X[split == 1], "Yte": Y[split == 1]}

    order = ["no-norm", "unsup"] + [f"few-{n}" for n in fewshot_ns] + ["full-retrain"]
    per = {m: [] for m in order}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for target in SESSIONS:
            src = [s for s in SESSIONS if s != target]
            Xs = np.concatenate([win[s]["Xtr"] for s in src])
            Ys = np.concatenate([win[s]["Ytr"] for s in src])
            if len(Xs) > args.src_cap:
                idx = rng.choice(len(Xs), args.src_cap, replace=False)
                Xs, Ys = Xs[idx], Ys[idx]
            xmu, xsd = Xs.mean((0, 2), keepdims=True), Xs.std((0, 2), keepdims=True) + 1e-6
            ymu, ysd = Ys.mean(0), Ys.std(0) + 1e-6
            Xs, Ys = zc(Xs, xmu, xsd), zc(Ys, ymu, ysd)

            dec = GRUDecoder(len(keep))
            fit(dec, Xs, Ys, epochs=args.epochs, lr=1e-3, batch_size=256, seed=seed)
            smean, svar = source_feature_stats(dec, Xs)

            Xtr = zc(win[target]["Xtr"], xmu, xsd); Ytr = zc(win[target]["Ytr"], ymu, ysd)
            Xte = zc(win[target]["Xte"], xmu, xsd); Yte = zc(win[target]["Yte"], ymu, ysd)

            def mk():
                return TransferNormalizer(dec, len(keep), n_groups=N_GROUPS, max_delay=MAXD)

            res = {"no-norm": r2_norm(mk(), Xte, Yte)}
            u = mk(); fit_unsupervised(u, Xtr, smean, svar, epochs=100, lr=0.1, seed=seed)
            res["unsup"] = r2_norm(u, Xte, Yte)
            for n in fewshot_ns:
                f = mk()
                idx = rng.choice(len(Xtr), min(n, len(Xtr)), replace=False)
                fit_supervised(f, Xtr[idx], Ytr[idx], epochs=120, lr=0.1, seed=seed)
                res[f"few-{n}"] = r2_norm(f, Xte, Yte)
            rt = GRUDecoder(len(keep)); fit(rt, Xtr, Ytr, epochs=args.epochs, lr=1e-3, batch_size=256, seed=seed)
            res["full-retrain"] = r2(rt, Xte, Yte)
            for m in order:
                per[m].append(res[m])
            print(f"seed {seed} target={target:>6}: " + "  ".join(f"{m}={res[m]:.3f}" for m in order))

    ci = {m: mean_ci(per[m]) for m in order}
    print(f"\nCross-session transfer velocity R² — mean [95% CI] over 3 targets × {args.seeds} seeds\n")
    for m in order:
        mn, lo, hi = ci[m]
        print(f"  {m:>14}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")
    gain = max(ci[f"few-{fewshot_ns[-1]}"][0], ci["unsup"][0]) - ci["no-norm"][0]
    verdict = ("conduction δ-fit helps real cross-session transfer"
               if gain > 0.02 else
               "conduction δ-fit gives no real cross-session benefit (shift is > timing) — honest bound")
    print(f"\n  best δ-fit gain over no-norm: {gain:+.3f}\n  VERDICT: {verdict}")

    plt.figure(figsize=(6.5, 4))
    xs = np.arange(len(order))
    plt.bar(xs, [ci[m][0] for m in order],
            yerr=[[ci[m][0]-ci[m][1] for m in order], [ci[m][2]-ci[m][0] for m in order]], capsize=4)
    plt.xticks(xs, order, rotation=20, ha="right"); plt.ylabel("transfer velocity R²")
    plt.title(f"Real cross-session transfer (MC_Maze S/M/L, {args.seeds} seeds)")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig = RESULTS / "xsession.png"; plt.savefig(fig, dpi=120); plt.close()
    (RESULTS / "xsession.json").write_text(json.dumps({
        "electrodes": int(len(keep)), "sessions": SESSIONS, "seeds": args.seeds,
        "fewshot_ns": fewshot_ns, "velocity_r2": {m: ci[m] for m in order},
        "best_delta_fit_gain": gain,
    }, indent=2))
    print(f"\nfigure: {fig.name}; data: results/xsession.json\n")


if __name__ == "__main__":
    main()
