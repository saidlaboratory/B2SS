#!/usr/bin/env python3
"""Phase 10.4 — EEG breadth: real cross-session transfer via MOABB (Zhou2016).

Uses MOABB only as a data loader (Zhou2016: 4 subjects x 3 sessions, 2-class motor
imagery, 14 EEG ch). Runs OUR transfer protocol: train a frozen classification
decoder on source sessions, transfer to a held-out session via conduction-delay
alignment (few-shot / unsupervised), vs no-norm and full-retrain. Accuracy.

HONEST SCOPE: conduction-delay alignment is a WEAK model of the EEG cross-session gap
(electrode placement, impedance, non-stationarity dominate). A modest/null result is
expected and bounds the claim to where conduction is the axis (intracortical). This
stage tests breadth, not a headline win.

Guarded: if MOABB is absent or the data can't be fetched, prints a message and exits
0 (never breaks the suite).

    python scripts/run_moabb_transfer.py            # ~20-30 min CPU (first run downloads)
    python scripts/run_moabb_transfer.py --quick
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--decim", type=int, default=5, help="time decimation (1251 -> ~250)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    fewshot_ns = [10, 40]
    if args.quick:
        args.subjects, args.epochs, args.seeds, fewshot_ns = 2, 20, 1, [40]

    try:
        from moabb.datasets import Zhou2016
        from moabb.paradigms import LeftRightImagery
    except Exception as e:
        print(f"[skip] MOABB not available ({type(e).__name__}: {e}). "
              f"Install with `pip install moabb` to run the EEG stage.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    from b2ss.model import DecoderConfig, B2SSDecoder
    from b2ss.stats import mean_ci
    from b2ss.train import fit, predict
    from b2ss.transfer import (TransferNormalizer, fit_supervised, fit_unsupervised,
                               source_feature_stats)
    RESULTS = Path(__file__).resolve().parent.parent / "results"
    RESULTS.mkdir(exist_ok=True)

    par, ds = LeftRightImagery(), Zhou2016()
    subjects = list(range(1, args.subjects + 1))
    try:
        loaded = {}
        for s in subjects:
            X, y, meta = par.get_data(dataset=ds, subjects=[s])
            X = X[:, :, ::args.decim].astype(np.float32)         # decimate time
            yb = (np.asarray(y) == "right_hand").astype(np.int64)
            loaded[s] = (X, yb, np.asarray(meta["session"]))
    except Exception as e:
        print(f"[skip] Could not fetch Zhou2016 ({type(e).__name__}: {e}). "
              f"The dataset host may be unreachable; the EEG stage is optional.")
        return

    n_chan, win = loaded[subjects[0]][0].shape[1:]
    print(f"\nEEG cross-session transfer (Zhou2016) | {n_chan} ch, win={win}, "
          f"{args.subjects} subjects x 3 sessions\n")

    def cfg():
        return DecoderConfig(n_chan=n_chan, win=win, fs=250 // args.decim,
                             patch=max(1, win // 25), d_model=64, nhead=4, num_layers=2,
                             dropout=0.3, task="classification", n_classes=2,
                             gate_mode="none", align_mode="none")

    def zc(a, mu, sd): return ((a - mu) / sd).astype(np.float32)

    order = ["no-norm", "unsup"] + [f"few-{n}" for n in fewshot_ns] + ["full-retrain"]
    per = {m: [] for m in order}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        for s in subjects:
            X, yb, sess = loaded[s]
            for held in sorted(set(sess)):
                src = sess != held
                Xs, Ys = X[src], yb[src]
                mu, sd = Xs.mean((0, 2), keepdims=True), Xs.std((0, 2), keepdims=True) + 1e-6
                Xs = zc(Xs, mu, sd)
                dec = B2SSDecoder(cfg())
                fit(dec, Xs, Ys, epochs=args.epochs, lr=1e-3, batch_size=64, seed=seed)
                smean, svar = source_feature_stats(dec, Xs)
                Xt, Yt = zc(X[~src], mu, sd), yb[~src]
                n_half = len(Xt) // 2
                Xcal, Ycal, Xev, Yev = Xt[:n_half], Yt[:n_half], Xt[n_half:], Yt[n_half:]

                def mk(): return TransferNormalizer(dec, n_chan, n_groups=7, max_delay=6)
                def acc(norm):
                    import torch
                    with torch.no_grad():
                        return accuracy_score(Yev, norm(torch.tensor(Xev)).argmax(1).numpy())
                res = {"no-norm": acc(mk())}
                u = mk(); fit_unsupervised(u, Xcal, smean, svar, epochs=80, lr=0.1, seed=seed); res["unsup"] = acc(u)
                for n in fewshot_ns:
                    f = mk(); idx = rng.choice(len(Xcal), min(n, len(Xcal)), replace=False)
                    fit_supervised(f, Xcal[idx], Ycal[idx], epochs=100, lr=0.1, seed=seed); res[f"few-{n}"] = acc(f)
                rt = B2SSDecoder(cfg()); fit(rt, Xcal, Ycal, epochs=args.epochs, lr=1e-3, batch_size=64, seed=seed)
                res["full-retrain"] = accuracy_score(Yev, predict(rt, Xev).argmax(1))
                for m in order:
                    per[m].append(res[m])
            print(f"seed {seed} subj {s}: " + "  ".join(f"{m}={np.mean(per[m][-3:]):.3f}" for m in ("no-norm", f"few-{fewshot_ns[-1]}", "full-retrain")))

    ci = {m: mean_ci(per[m]) for m in order}
    print(f"\nEEG cross-session accuracy — mean [95% CI]\n")
    for m in order:
        mn, lo, hi = ci[m]
        print(f"  {m:>14}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")
    gain = max(ci[f"few-{fewshot_ns[-1]}"][0], ci["unsup"][0]) - ci["no-norm"][0]
    print(f"\n  best δ-fit gain over no-norm: {gain:+.3f}  "
          f"({'helps EEG' if gain > 0.02 else 'no EEG benefit — bounds the claim to intracortical (expected)'})")

    plt.figure(figsize=(6, 4))
    xs = np.arange(len(order))
    plt.bar(xs, [ci[m][0] for m in order],
            yerr=[[ci[m][0]-ci[m][1] for m in order], [ci[m][2]-ci[m][0] for m in order]], capsize=4)
    plt.axhline(0.5, ls="--", c="gray", label="chance")
    plt.xticks(xs, order, rotation=20, ha="right"); plt.ylabel("cross-session accuracy")
    plt.title(f"EEG cross-session transfer (Zhou2016, {args.seeds} seeds)")
    plt.legend(); plt.tight_layout()
    fig = RESULTS / "moabb_transfer.png"; plt.savefig(fig, dpi=120); plt.close()
    (RESULTS / "moabb_transfer.json").write_text(json.dumps({
        "dataset": "Zhou2016", "n_chan": int(n_chan), "win": int(win),
        "subjects": args.subjects, "seeds": args.seeds, "fewshot_ns": fewshot_ns,
        "accuracy": {m: ci[m] for m in order}, "best_delta_fit_gain": gain,
    }, indent=2))
    print(f"\nfigure: {fig.name}; data: results/moabb_transfer.json\n")


if __name__ == "__main__":
    main()
