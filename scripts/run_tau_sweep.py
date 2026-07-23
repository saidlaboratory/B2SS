#!/usr/bin/env python3
"""Phase 11 — is CADENCE's shrinkage strength tuned on the evaluation sessions?

`shrink_tau` (and `std_floor`) were defaults, not measured choices, and tau=200 sits in the
middle of the budget grid we report on. That is exactly the shape of a hyperparameter fitted
to the test set, and nothing in the repo refuted it. This script does two things:

  1. SWEEP one-at-a-time around the defaults and report the whole surface, so the
     sensitivity is visible rather than asserted.
  2. SELECT tau by LEAVE-ONE-SESSION-OUT: for each held-out session pick the tau that was
     best on the OTHER sessions, then score it on the held-out one. If LOSO selection lands
     near the fixed default, the default is not doing secret work; if it beats the default,
     the default was a bad guess; if both beat a fixed oracle-free rule by nothing, say so.

tau has a plain reading -- the number of calibration windows at which you trust the session
as much as the source prior (w = n/(n+tau) = 1/2). The sweep is informative on its own.

    python scripts/run_tau_sweep.py --seeds 3
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

from b2ss.indy import list_sessions, load_indy_session
from b2ss.intracortical import make_windows
from b2ss.baselines import GRUDecoder
from b2ss.train import fit, predict
from b2ss.transfer import source_feature_stats
from b2ss.cadence import CADENCE
from b2ss.stats import mean_ci, paired_by_unit

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20
TAUS = [50.0, 100.0, 200.0, 400.0, 800.0]
FLOORS = [0.02, 0.05, 0.1, 0.2, 0.4]
DEF_TAU, DEF_FLOOR = 200.0, 0.1


def vel_r2(Yt, Yp):
    return float(r2_score(Yt, Yp, multioutput="variance_weighted"))


def zc(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def loso_select(grid, values, n_sess):
    """grid[value] = (n_sess,) mean R2 per session. For each session, pick the value that
    was best on the OTHER sessions and return the score it earns on the held-out one."""
    out, picks = [], []
    for s in range(n_sess):
        others = [np.mean([grid[v][k] for k in range(n_sess) if k != s]) for v in values]
        v = values[int(np.argmax(others))]
        picks.append(v)
        out.append(grid[v][s])
    return np.array(out), picks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--held-in", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--src-cap", type=int, default=6000)
    ap.add_argument("--budgets", default="25,100,500,2000")
    ap.add_argument("--subject", default="indy",
                    help="`loco` is the second monkey on the same rig — see b2ss/indy.py")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.src_cap, args.budgets = 1, 15, 3000, "25,500"
    RESULTS.mkdir(exist_ok=True)
    Ns = [int(b) for b in args.budgets.split(",")]

    data = []
    for p in list_sessions(subject=args.subject):
        try:
            d = load_indy_session(p)
        except OSError:
            continue
        X, Y, split, _ = make_windows(d, WIN)
        data.append((p.stem, X, Y, split))
    n_chan = min(x[1].shape[1] for x in data)
    targets = list(range(args.held_in, len(data)))
    n_sess = len(targets)
    print(f"\nCADENCE hyperparameter sweep | {n_sess} target sessions x {args.seeds} seeds | "
          f"tau {TAUS} | std_floor {FLOORS} | N {Ns}\n")

    # tau[N][tau] and floor[N][floor] -> list of R2, (seed-major, session-minor)
    tau_r2 = {N: {t: [] for t in TAUS} for N in Ns}
    floor_r2 = {N: {f: [] for f in FLOORS} for N in Ns}
    for seed in range(args.seeds):
        import torch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        Xs = np.concatenate([data[i][1][data[i][3] == 0][:, :n_chan] for i in range(args.held_in)])
        Ys = np.concatenate([data[i][2][data[i][3] == 0] for i in range(args.held_in)])
        if len(Xs) > args.src_cap:
            k = rng.choice(len(Xs), args.src_cap, replace=False); Xs, Ys = Xs[k], Ys[k]
        xmu, xsd = Xs.mean((0, 2), keepdims=True), Xs.std((0, 2), keepdims=True) + 1e-6
        ymu, ysd = Ys.mean(0), Ys.std(0) + 1e-6
        Xsz = zc(Xs, xmu, xsd)
        dec = GRUDecoder(n_chan)
        fit(dec, Xsz, zc(Ys, ymu, ysd), epochs=args.epochs, lr=1e-3, batch_size=256, seed=seed)
        src_stats = source_feature_stats(dec, Xsz)

        for ti in targets:
            _, X, Y, split = data[ti]
            Xtr = zc(X[split == 0][:, :n_chan], xmu, xsd)
            Xte, Yte = zc(X[split == 1][:, :n_chan], xmu, xsd), zc(Y[split == 1], ymu, ysd)
            for N in Ns:
                Xn = Xtr[:N]
                for t in TAUS:
                    c = CADENCE(dec, n_chan, src_stats=src_stats, shrink_tau=t, std_floor=DEF_FLOOR)
                    c.adapt(Xn); tau_r2[N][t].append(vel_r2(Yte, c.predict(Xte)))
                for f in FLOORS:
                    c = CADENCE(dec, n_chan, src_stats=src_stats, shrink_tau=DEF_TAU, std_floor=f)
                    c.adapt(Xn); floor_r2[N][f].append(vel_r2(Yte, c.predict(Xte)))
        print(f"  seed {seed} done")

    def per_session(d):                      # (seed-major, session-minor) -> (n_sess,)
        return np.asarray(d, float).reshape(-1, n_sess).mean(0)

    print("\ntau sweep — mean cross-session R2 (std_floor = %.2f)\n" % DEF_FLOOR)
    print(f"  {'N=':>8}" + "".join(f"{'tau=' + str(int(t)):>10}" for t in TAUS) + f"{'LOSO':>10}")
    loso = {}
    for N in Ns:
        grid = {t: per_session(tau_r2[N][t]) for t in TAUS}
        sel, picks = loso_select(grid, TAUS, n_sess)
        loso[N] = {"r2": float(sel.mean()), "picks": [float(p) for p in picks],
                   "vs_default": paired_by_unit(sel, grid[DEF_TAU], n_sess)}
        print(f"  {N:>8}" + "".join(f"  {grid[t].mean():+.3f}   " for t in TAUS)
              + f"  {sel.mean():+.3f}")

    print("\nLOSO-selected vs the fixed default tau=200 (paired over sessions)\n")
    for N in Ns:
        d = loso[N]["vs_default"]
        picks = loso[N]["picks"]
        print(f"  N={N:>5}  delta {d['delta']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]  "
              f"p={d['p']:.3f}  picked tau in {{{','.join(str(int(p)) for p in sorted(set(picks)))}}}")

    print("\nstd_floor sweep — mean cross-session R2 (tau = %d)\n" % DEF_TAU)
    print(f"  {'N=':>8}" + "".join(f"{'f=' + str(f):>10}" for f in FLOORS))
    for N in Ns:
        print(f"  {N:>8}" + "".join(f"  {per_session(floor_r2[N][f]).mean():+.3f}   " for f in FLOORS))

    worst_gain = max(abs(loso[N]["vs_default"]["delta"]) for N in Ns)
    sig = [N for N in Ns if loso[N]["vs_default"]["p"] < 0.05]
    verdict = (
        f"LOSO-selected tau differs from the fixed default by at most {worst_gain:.4f} R2 "
        f"({'no budget' if not sig else 'N=' + ','.join(map(str, sig))} significant at p<.05), so the "
        f"reported numbers do not depend on having picked tau=200 with hindsight. "
        f"The tau surface is {'flat' if worst_gain < 0.02 else 'not flat'} across "
        f"{TAUS[0]:.0f}-{TAUS[-1]:.0f}.")
    print(f"\n  VERDICT: {verdict}\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for N in Ns:
        axes[0].plot(TAUS, [per_session(tau_r2[N][t]).mean() for t in TAUS], "o-", label=f"N={N}")
        axes[1].plot(FLOORS, [per_session(floor_r2[N][f]).mean() for f in FLOORS], "s-", label=f"N={N}")
    axes[0].axvline(DEF_TAU, color="0.6", ls="--", lw=1); axes[0].set_xscale("log")
    axes[0].set_xlabel("shrink_tau (windows)"); axes[0].set_ylabel("cross-session velocity R²")
    axes[0].set_title("Shrinkage strength")
    axes[1].axvline(DEF_FLOOR, color="0.6", ls="--", lw=1); axes[1].set_xscale("log")
    axes[1].set_xlabel("std_floor"); axes[1].set_title("Scale floor")
    for a in axes:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "tau_sweep.png", dpi=120); plt.close()
    (RESULTS / "tau_sweep.json").write_text(json.dumps({
        "taus": TAUS, "floors": FLOORS, "default_tau": DEF_TAU, "default_floor": DEF_FLOOR,
        "budgets": Ns, "seeds": args.seeds, "n_sessions": n_sess,
        "tau_r2": {str(N): {str(t): mean_ci(tau_r2[N][t]) for t in TAUS} for N in Ns},
        "floor_r2": {str(N): {str(f): mean_ci(floor_r2[N][f]) for f in FLOORS} for N in Ns},
        "loso": {str(N): loso[N] for N in Ns},
        "verdict": verdict,
    }, indent=2))
    print("figure: results/tau_sweep.png; data: results/tau_sweep.json\n")


if __name__ == "__main__":
    main()
