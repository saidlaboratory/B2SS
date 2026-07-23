#!/usr/bin/env python3
"""Why per-session standardisation fails online: chronological BIAS, not sampling noise.

The obvious story for why a per-session standardiser (MPA/AdaBN/Euclidean alignment) breaks
down at small calibration budgets is that the per-channel estimates are noisy. That story is
wrong, and it is testable in one line of experimental design: hold the number of calibration
windows fixed and change only HOW they are drawn.

  first-N    the first N windows in time -- what an online BCI actually receives
  random-N   N windows drawn from across the session's training portion -- same N,
             same estimator, no temporal bias

If the failure were sampling noise, the two would agree: N windows is N windows. They do
not agree, and not by a little.

The consequence is a general one for test-time adaptation on streams, not a BCI detail:
**online calibration data is not a random sample of the session it calibrates for.** Any
adapter whose statistics assume otherwise -- including a textbook empirical-Bayes shrinkage,
which models variance and is blind to bias (see cadence._shrink_weight) -- will inherit the
bias. A conservative fixed shrinkage helps only because it declines to commit to a biased
estimate, which is the right behaviour for a different reason than the one usually given.

    python scripts/run_calibration_bias.py --seeds 3
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
from b2ss.ibci_baselines import MPA, source_input_stats
from b2ss.stats import mean_ci, paired_by_unit

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20
KEYS = ["mpa-first", "mpa-random", "cadence-first", "cadence-random"]


def vel_r2(Yt, Yp):
    return float(r2_score(Yt, Yp, multioutput="variance_weighted"))


def zc(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--held-in", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--src-cap", type=int, default=6000)
    ap.add_argument("--budgets", default="25,50,100,200,500")
    ap.add_argument("--subject", default="indy")
    ap.add_argument("--max-chan", type=int, default=0,
                    help="cap channels (loco=192 M1+S1; 96 for M1-only like-for-like)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.src_cap, args.budgets = 1, 15, 3000, "25,200"
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
    if args.max_chan:
        n_chan = min(n_chan, args.max_chan)
    targets = list(range(args.held_in, len(data)))
    U = len(targets)
    print(f"\nCalibration bias vs noise | {len(data)} sessions, {n_chan} electrodes | "
          f"{U} targets x {args.seeds} seeds | N {Ns}\n")

    res = {k: {n: [] for n in Ns} for k in KEYS}
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
        ss, si = source_feature_stats(dec, Xsz), source_input_stats(Xsz)

        for ti in targets:
            _, X, Y, sp = data[ti]
            Xtr = zc(X[sp == 0][:, :n_chan], xmu, xsd)
            Xte, Yte = zc(X[sp == 1][:, :n_chan], xmu, xsd), zc(Y[sp == 1], ymu, ysd)
            for n in Ns:
                draws = {"first": Xtr[:n],
                         "random": Xtr[rng.choice(len(Xtr), n, replace=False)]}
                for how, Xn in draws.items():
                    m = MPA(dec, si); m.adapt(Xn)
                    res[f"mpa-{how}"][n].append(vel_r2(Yte, m.predict(Xte)))
                    c = CADENCE(dec, n_chan, src_stats=ss); c.adapt(Xn)
                    res[f"cadence-{how}"][n].append(vel_r2(Yte, c.predict(Xte)))
        print(f"  seed {seed} done")

    ci = {k: {n: mean_ci(res[k][n]) for n in Ns} for k in KEYS}
    print(f"\nCross-session velocity R2 — mean over {U} targets x {args.seeds} seeds\n")
    print(f"  {'N=':>16}" + "".join(f"{n:>9}" for n in Ns))
    for k in KEYS:
        print(f"  {k:>16}" + "".join(f"  {ci[k][n][0]:+.3f}" for n in Ns))

    print("\nSAME N, only the draw differs — random minus first, paired over sessions\n")
    paired = {}
    for base in ("mpa", "cadence"):
        print(f"  {base}")
        paired[base] = {}
        for n in Ns:
            d = paired_by_unit(res[f"{base}-random"][n], res[f"{base}-first"][n], U)
            paired[base][n] = d
            print(f"    N={n:>4}  {d['delta']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  "
                  f"p={d['p']:.3f}  {d['won']}/{d['n']} sessions")

    n0 = Ns[0]
    d0 = paired["mpa"][n0]
    # gate the strong wording on actual significance — a positive point estimate on 3
    # sessions with p=0.3 is a direction, not a demonstration.
    sig = d0["p"] < 0.05 and d0["delta"] > 0
    directional = d0["delta"] > 0 and d0["won"] >= (d0["n"] + 1) // 2
    if sig:
        strength = ("so the online failure is CHRONOLOGICAL BIAS, not sampling noise")
    elif directional:
        strength = (f"the direction agrees ({d0['won']}/{d0['n']} sessions) but is NOT "
                    f"significant here (p={d0['p']:.3f}, {d0['n']} sessions) — a directional "
                    f"replication, underpowered, not a demonstration on its own")
    else:
        strength = (f"the effect does NOT replicate on this cohort (delta {d0['delta']:+.3f}, "
                    f"p={d0['p']:.3f}, {d0['won']}/{d0['n']}) — report straight")
    verdict = (
        f"At N={n0} a plain standardiser scores {ci['mpa-first'][n0][0]:+.3f} on the FIRST "
        f"{n0} windows and {ci['mpa-random'][n0][0]:+.3f} on {n0} RANDOM ones "
        f"(delta {d0['delta']:+.3f}, p={d0['p']:.3f}, {d0['won']}/{d0['n']} sessions); "
        f"{strength}.")
    print(f"\n  VERDICT: {verdict}\n")

    plt.figure(figsize=(7, 4.5))
    style = {"mpa-first": ("s--", "#d62728"), "mpa-random": ("s-", "#2ca02c"),
             "cadence-first": ("o--", "#1f77b4"), "cadence-random": ("o-", "#9467bd")}
    for k in KEYS:
        fmt, c = style[k]
        ys = [ci[k][n][0] for n in Ns]
        lo = [ci[k][n][0] - ci[k][n][1] for n in Ns]
        hi = [ci[k][n][2] - ci[k][n][0] for n in Ns]
        plt.errorbar(Ns, ys, yerr=[lo, hi], fmt=fmt, color=c, capsize=3, lw=1.8, label=k)
    plt.xscale("log")
    plt.xlabel("calibration windows N")
    plt.ylabel("cross-session velocity R²")
    plt.title("Same N, different draw: online calibration data is not a random sample")
    plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=8); plt.tight_layout()
    tag = "" if args.subject == "indy" else f"_{args.subject}"
    plt.savefig(RESULTS / f"calibration_bias{tag}.png", dpi=120); plt.close()
    (RESULTS / f"calibration_bias{tag}.json").write_text(json.dumps({
        "subject": args.subject, "max_chan": args.max_chan,
        "sessions": [d[0] for d in data], "n_chan": int(n_chan), "budgets": Ns,
        "seeds": args.seeds, "n_targets": U,
        "r2": {k: {str(n): ci[k][n] for n in Ns} for k in KEYS},
        "raw": {k: {str(n): res[k][n] for n in Ns} for k in KEYS},
        "paired_random_minus_first": {b: {str(n): paired[b][n] for n in Ns} for b in paired},
        "verdict": verdict,
    }, indent=2))
    print(f"figure: results/calibration_bias{tag}.png; data: results/calibration_bias{tag}.json\n")


if __name__ == "__main__":
    main()
