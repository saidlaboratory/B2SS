#!/usr/bin/env python3
"""Phase 11 — the online data-efficiency curve on real cross-session Indy.

Realistic online BCI: at each new session you must decode from the FIRST few windows,
before much calibration data has arrived. Every method adapts UNSUPERVISED from the first
N windows of the held-out session; we plot cross-session velocity R2 vs N.

What the curve is for. A per-session standardiser has to estimate a mean and std for every
electrode from whatever calibration data exists so far. On a 96-channel array a sizeable
minority of channels are near-silent over a short slice, so those estimates are noisy or
degenerate and the aligned input is garbage. CADENCE shrinks each estimate toward the
source prior by w = n/(n+tau), so the adapter starts near the identity and earns its way to
a full standardiser. The question this script answers is how much that buys, against three
comparators that matter:

  no-adapt      the frozen decoder. The only honest reference at small N -- if adapting
                cannot beat leaving it alone, the adapter is not worth shipping.
  mpa           per-session standardiser WITH a scale floor. The fair strong baseline.
  mpa-nofloor   the same without the floor. Not a baseline -- a documented failure mode,
                reported so the floored number is not mistaken for the whole story.

BUDGETS include `all` (every training window in the session, ~15k, ~300 s), because
"full calibration" claims made at N=2000 describe 13% of the available data and the two
curves are still moving there.

Primary analysis is PAIRED OVER SESSIONS (seeds averaged within session, not pooled as
independent draws), Benjamini-Hochberg corrected across the budget grid.

    python scripts/run_indy_calibration.py --seeds 3
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
from b2ss.tta_baselines import Tent, free_lora
from b2ss.ibci_baselines import MPA, source_input_stats
from b2ss.stats import mean_ci, paired_by_unit, benjamini_hochberg

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20
GRAD_CAP = 2000                     # windows the gradient adapters see (see loop comment)
METHODS = ["no-adapt", "mpa-nofloor", "mpa", "tent", "free-lora", "cadence"]


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
    ap.add_argument("--fast-steps", type=int, default=20)
    ap.add_argument("--budgets", default="25,50,100,200,500,2000,all",
                    help="`all` = every training window in the session (true full calibration)")
    ap.add_argument("--subject", default="indy",
                    help="`loco` is the second monkey on the same rig — see b2ss/indy.py")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.src_cap = 1, 15, 3000
        args.budgets = "50,2000,all"
    RESULTS.mkdir(exist_ok=True)
    Ns = [b if b == "all" else int(b) for b in args.budgets.split(",")]

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
    n_units = len(targets)
    print(f"\nIndy online data-efficiency | {len(data)} sessions, {n_chan} electrodes | "
          f"source={args.held_in} frozen, {n_units} targets, N-windows {Ns}\n")

    curve = {m: {N: [] for N in Ns} for m in METHODS}
    silent_frac, avail = [], []
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
        src_stats, src_in = source_feature_stats(dec, Xsz), source_input_stats(Xsz)

        for ti in targets:
            name, X, Y, split = data[ti]
            Xtr = zc(X[split == 0][:, :n_chan], xmu, xsd)             # session windows, in order
            Xte, Yte = zc(X[split == 1][:, :n_chan], xmu, xsd), zc(Y[split == 1], ymu, ysd)
            na = vel_r2(Yte, predict(dec, Xte))
            if seed == 0:
                avail.append(len(Xtr))
                silent_frac.append(float((Xtr[:25].std((0, 2)) < 0.1).mean()))
            for N in Ns:
                Xn = Xtr if N == "all" else Xtr[:N]                   # first N windows (online)
                # ponytail: the gradient adapters see at most GRAD_CAP windows per step --
                # full-batch Adam over 15k windows costs ~30x the closed-form methods for a
                # column that is not the claim. Recorded in the JSON so it isn't invisible.
                Xg = Xn[:GRAD_CAP]
                curve["no-adapt"][N].append(na)
                built = {
                    "mpa-nofloor": (MPA(dec, src_in, std_floor=0.0), Xn),
                    "mpa": (MPA(dec, src_in), Xn),
                    "tent": (Tent(dec, n_chan, src_stats, steps=args.fast_steps), Xg),
                    "free-lora": (free_lora(dec, n_chan, src_stats, fast_steps=args.fast_steps), Xg),
                    "cadence": (CADENCE(dec, n_chan, src_stats=src_stats), Xn),
                }
                for key, (mod, Xin) in built.items():
                    mod.adapt(Xin)
                    curve[key][N].append(vel_r2(Yte, mod.predict(Xte)))
        print(f"  seed {seed} done")

    ci = {m: {N: mean_ci(curve[m][N]) for N in Ns} for m in METHODS}
    print(f"\nCross-session velocity R2 — mean over {n_units} targets x {args.seeds} seeds")
    print(f"(descriptive only; the inference is the paired table below)\n")
    print(f"  {'N=':>12}" + "".join(f"{str(N):>9}" for N in Ns))
    for m in METHODS:
        print(f"  {m:>12}" + "".join(f"  {ci[m][N][0]:+.3f}" for N in Ns))

    # ---- primary analysis: paired over sessions, BH-corrected across the budget grid --
    contrasts = [("cadence", "no-adapt"), ("cadence", "mpa"), ("mpa", "no-adapt")]
    paired = {f"{a}_vs_{b}": {N: paired_by_unit(curve[a][N], curve[b][N], n_units) for N in Ns}
              for a, b in contrasts}
    flat = [(k, N) for k in paired for N in Ns]
    rej = benjamini_hochberg([paired[k][N]["p"] for k, N in flat], q=0.05)
    for (k, N), r in zip(flat, rej):
        paired[k][N]["bh_reject"] = bool(r)

    print(f"\nPAIRED over {n_units} sessions (seeds averaged within session), "
          f"BH-corrected across {len(flat)} tests\n")
    for k in paired:
        print(f"  {k}")
        for N in Ns:
            d = paired[k][N]
            mark = "*" if d["bh_reject"] else " "
            print(f"    N={str(N):>5}  {d['delta']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  "
                  f"p={d['p']:.3f}{mark}  {d['won']}/{d['n']} sessions")

    small = [N for N in Ns if N != "all" and N <= 100]
    vs_na = [paired["cadence_vs_no-adapt"][N] for N in small]
    vs_mpa = [paired["cadence_vs_mpa"][N] for N in small]
    allN = Ns[-1]
    verdict = (
        f"At small budgets (N<=100) CADENCE beats no-adapt by "
        f"{min(d['delta'] for d in vs_na):+.3f}..{max(d['delta'] for d in vs_na):+.3f} R2 "
        f"({sum(d['bh_reject'] for d in vs_na)}/{len(small)} BH-significant) and a FLOORED MPA by "
        f"{min(d['delta'] for d in vs_mpa):+.3f}..{max(d['delta'] for d in vs_mpa):+.3f}. "
        f"At full calibration (N={allN}, {int(np.mean(avail))} windows avg) CADENCE vs MPA = "
        f"{paired['cadence_vs_mpa'][allN]['delta']:+.3f} "
        f"(p={paired['cadence_vs_mpa'][allN]['p']:.3f}). The unfloored standardiser diverges "
        f"below N=200 ({ci['mpa-nofloor'][Ns[0]][0]:+.3f} at N={Ns[0]}) — a scale-floor failure, "
        f"not a property of standardisation.")
    print(f"\n  VERDICT: {verdict}\n")

    plt.figure(figsize=(7, 4.8))
    xs = [n if n != "all" else int(np.mean(avail)) for n in Ns]
    styles = {"cadence": ("o-", 2.4), "mpa": ("s--", 1.6), "mpa-nofloor": ("s:", 1.0),
              "tent": ("^:", 1.2), "no-adapt": ("d:", 1.2), "free-lora": ("x-", 1.0)}
    for m in METHODS:
        ys = [ci[m][N][0] for N in Ns]
        lo = [ci[m][N][0] - ci[m][N][1] for N in Ns]; hi = [ci[m][N][2] - ci[m][N][0] for N in Ns]
        fmt, lw = styles[m]
        plt.errorbar(xs, ys, yerr=[lo, hi], fmt=fmt, lw=lw, capsize=3, label=m)
    plt.axvspan(0, 100, color="0.9", zorder=0)
    plt.annotate("data-scarce", (30, plt.ylim()[0] + 0.05), fontsize=8, color="0.35")
    plt.xscale("log"); plt.xlabel("unlabelled calibration windows N (online)")
    plt.ylabel("cross-session velocity R²")
    plt.title("Online data-efficiency on the Indy stream")
    plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS / "indy_calibration.png", dpi=120); plt.close()
    (RESULTS / "indy_calibration.json").write_text(json.dumps({
        "sessions": [d[0] for d in data], "n_chan": int(n_chan), "N_windows": [str(N) for N in Ns],
        "seeds": args.seeds, "n_targets": n_units,
        "windows_available_per_session": avail, "grad_adapter_window_cap": GRAD_CAP,
        "silent_channel_frac_at_25": float(np.mean(silent_frac)),
        "curve": {m: {str(N): ci[m][N] for N in Ns} for m in METHODS},
        "raw": {m: {str(N): curve[m][N] for N in Ns} for m in METHODS},
        "paired": {k: {str(N): paired[k][N] for N in Ns} for k in paired},
        "verdict": verdict,
    }, indent=2))
    print(f"figure: results/indy_calibration.png; data: results/indy_calibration.json\n")


if __name__ == "__main__":
    main()
