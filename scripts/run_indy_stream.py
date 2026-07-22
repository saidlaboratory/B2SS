#!/usr/bin/env python3
"""Phase 11 headline — CADENCE on the monkey-Indy continual stream.

Freeze a GRU decoder on the earliest Indy sessions, then adapt online across the
remaining sessions in temporal order (with revisits of the earliest streamed ones).
Every method plugs into the same streaming harness and is scored on continual
metrics: cumulative online R2, worst-session R2, collapse-rate, backward-transfer.

The decisive read is the accuracy x stability plane — a method wins by being ABOVE
No-Adapt on cumulative R2 while AT-OR-BELOW it on collapse-rate, which free-TTA
(Tent/CoTTA) provably cannot do over a long recurring stream.

    python scripts/run_indy_stream.py --quick            # go/no-go smoke
    python scripts/run_indy_stream.py --seeds 3          # ~real run (CPU, minutes)
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
from b2ss.tta_baselines import NoAdapt, Tent, CoTTA, RDumb, free_lora
from b2ss.ibci_baselines import MPA, NoMAD, source_input_stats, source_latent_moments
from b2ss.stream import run_stream
from b2ss.continual import cumulative_r2, worst_session_r2, collapse_rate, backward_transfer
from b2ss.stats import mean_ci

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20
METRICS = ["cumulative", "worst", "collapse", "bwt"]


def vel_r2(Yt, Yp):
    return float(r2_score(Yt, Yp, multioutput="variance_weighted"))


def zc(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def load_windows(win: int):
    """Load every downloaded Indy session as windows, sliced to the common (first
    n_chan) electrode set so channels correspond across days."""
    out = []
    for p in list_sessions():
        try:
            d = load_indy_session(p)
        except OSError as e:                                  # partial/truncated download
            print(f"  skip {p.name}: {str(e)[:60]}")
            continue
        X, Y, split, _ = make_windows(d, win)
        out.append([p.stem, X, Y, split])
    if not out:
        sys.exit("No readable Indy sessions in ~/b2ss_data/indy (see b2ss/indy.py).")
    n_chan = min(x[1].shape[1] for x in out)
    for row in out:
        row[1] = row[1][:, :n_chan]                          # first n_chan electrodes
    return out, n_chan


def build_sessions(data, idxs, xmu, xsd, ymu, ysd, adapt_cap, rng, retrain_cap=8000):
    sessions = []
    for i in idxs:
        name, X, Y, split = data[i]
        Xtr, Ytr = X[split == 0], Y[split == 0]
        # adaptation set (small, capped) vs full-recalibration set (large — the honest
        # ceiling; a per-session decoder must see the whole session's training data).
        cap = rng.choice(len(Xtr), min(adapt_cap, len(Xtr)), replace=False)
        rcap = rng.choice(len(Xtr), min(retrain_cap, len(Xtr)), replace=False)
        sessions.append({
            "name": name,
            "Xtr": zc(Xtr[cap], xmu, xsd), "Ytr": zc(Ytr[cap], ymu, ysd),
            "Xtr_full": zc(Xtr[rcap], xmu, xsd), "Ytr_full": zc(Ytr[rcap], ymu, ysd),
            "Xte": zc(X[split == 1], xmu, xsd), "Yte": zc(Y[split == 1], ymu, ysd),
        })
    return sessions


def metrics_of(traj, first_visit, last_visit, floor):
    return {
        "cumulative": cumulative_r2(traj),
        "worst": worst_session_r2(traj),
        "collapse": collapse_rate(traj, floor),
        "bwt": backward_transfer(first_visit, last_visit),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--held-in", type=int, default=3, help="earliest sessions used as frozen source")
    ap.add_argument("--revisits", type=int, default=2, help="earliest streamed sessions revisited at the end")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--src-cap", type=int, default=8000)
    ap.add_argument("--adapt-cap", type=int, default=1500)
    ap.add_argument("--fast-steps", type=int, default=30)
    ap.add_argument("--collapse-floor", type=float, default=0.2)
    ap.add_argument("--reset-every", type=int, default=3)
    ap.add_argument("--methods", default="no-adapt,mpa,tent,cotta,nomad,free-lora,cadence")
    ap.add_argument("--no-retrain", action="store_true", help="skip the full-retrain ceiling")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.held_in, args.epochs = 1, 3, 20
        args.src_cap, args.adapt_cap, args.fast_steps = 5000, 1200, 30
        args.methods = "no-adapt,mpa,tent,nomad,free-lora,cadence"
        args.no_retrain = True
    RESULTS.mkdir(exist_ok=True)

    data, n_chan = load_windows(WIN)
    if len(data) <= args.held_in + 1:
        sys.exit(f"Need > held_in+1 sessions; have {len(data)}, held_in={args.held_in}.")
    methods = args.methods.split(",")
    stream_idx = list(range(args.held_in, len(data)))
    print(f"\nIndy continual stream | {len(data)} sessions, {n_chan} electrodes | "
          f"source={args.held_in} frozen, stream={len(stream_idx)} + {args.revisits} revisits\n")
    print("sessions:", ", ".join(d[0] for d in data), "\n")

    fs = args.fast_steps
    per = {m: {k: [] for k in METRICS} for m in methods}
    if not args.no_retrain:
        per["full-retrain"] = {k: [] for k in METRICS}

    for seed in range(args.seeds):
        import torch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        # frozen source decoder on the earliest sessions
        Xs = np.concatenate([data[i][1][data[i][3] == 0] for i in range(args.held_in)])
        Ys = np.concatenate([data[i][2][data[i][3] == 0] for i in range(args.held_in)])
        if len(Xs) > args.src_cap:
            k = rng.choice(len(Xs), args.src_cap, replace=False)
            Xs, Ys = Xs[k], Ys[k]
        xmu, xsd = Xs.mean((0, 2), keepdims=True), Xs.std((0, 2), keepdims=True) + 1e-6
        ymu, ysd = Ys.mean(0), Ys.std(0) + 1e-6
        Xsz, Ysz = zc(Xs, xmu, xsd), zc(Ys, ymu, ysd)
        dec = GRUDecoder(n_chan)
        fit(dec, Xsz, Ysz, epochs=args.epochs, lr=1e-3, batch_size=256, seed=seed)
        src_stats = source_feature_stats(dec, Xsz)
        src_in = source_input_stats(Xsz)
        src_lat = source_latent_moments(dec, Xsz)

        sessions = build_sessions(data, stream_idx, xmu, xsd, ymu, ysd, args.adapt_cap, rng)
        order = list(range(len(sessions)))
        schedule = order + order[:args.revisits]

        factories = {
            "no-adapt": lambda: NoAdapt(dec),
            "mpa": lambda: MPA(dec, src_in),
            "tent": lambda: Tent(dec, n_chan, src_stats, steps=fs),
            "cotta": lambda: CoTTA(dec, n_chan, src_stats, steps=fs),
            "nomad": lambda: NoMAD(dec, n_chan, src_lat, steps=fs),
            "rdumb": lambda: RDumb(lambda: Tent(dec, n_chan, src_stats, steps=fs), args.reset_every),
            "free-lora": lambda: free_lora(dec, n_chan, src_stats, fast_steps=fs),
            "cadence": lambda: CADENCE(dec, n_chan, src_stats=src_stats, fast_steps=fs),
        }
        for m in methods:
            out = run_stream(factories[m](), sessions, schedule, score_fn=vel_r2)
            traj = [r for _, r in out["visits"]]
            mm = metrics_of(traj, out["first_visit"], out["last_visit"], args.collapse_floor)
            for k in METRICS:
                per[m][k].append(mm[k])
            print(f"  seed {seed} {m:>10}: cum={mm['cumulative']:.3f} worst={mm['worst']:.3f} "
                  f"collapse={mm['collapse']:.2f} bwt={mm['bwt']:+.3f}")

        if not args.no_retrain:
            traj = []
            for s in sessions:
                rt = GRUDecoder(n_chan)                       # per-session full recalibration (ceiling)
                fit(rt, s["Xtr_full"], s["Ytr_full"], epochs=args.epochs, lr=1e-3, batch_size=256, seed=seed)
                traj.append(vel_r2(s["Yte"], predict(rt, s["Xte"])))
            fv = {sessions[i]["name"]: traj[i] for i in range(len(sessions))}
            mm = metrics_of(traj, fv, fv, args.collapse_floor)
            for k in METRICS:
                per["full-retrain"][k].append(mm[k])
            print(f"  seed {seed} full-retrain: cum={mm['cumulative']:.3f} worst={mm['worst']:.3f}")

    # aggregate + verdict
    ci = {m: {k: mean_ci(per[m][k]) for k in METRICS} for m in per}
    print("\nContinual metrics — mean [95% CI] over "
          f"{args.seeds} seeds (collapse floor R2<{args.collapse_floor})\n")
    print(f"  {'method':>12}  {'cumulative':>18}  {'worst':>18}  {'collapse':>16}  {'BWT':>16}")
    for m in per:
        c = ci[m]
        def f(k): return f"{c[k][0]:.3f}[{c[k][1]:.2f},{c[k][2]:.2f}]"
        print(f"  {m:>12}  {f('cumulative'):>18}  {f('worst'):>18}  {f('collapse'):>16}  {f('bwt'):>16}")

    na = ci.get("no-adapt")
    verdict = "n/a"
    if na and "cadence" in ci:
        cad = ci["cadence"]
        acc_gain = cad["cumulative"][0] - na["cumulative"][0]
        unstructured = [m for m in ("free-lora", "nomad") if m in ci]      # the free/high-DOF methods
        worst_free = max((ci[m]["collapse"][0] for m in unstructured), default=0.0)
        cadence_safe = cad["collapse"][0] <= na["collapse"][0] + 1e-9
        free_collapses = any(ci[m]["collapse"][0] > na["collapse"][0] + 1e-9 for m in unstructured)
        verdict = (f"CADENCE cum {acc_gain:+.3f} vs No-Adapt, collapse {cad['collapse'][0]:.2f} "
                   f"vs No-Adapt {na['collapse'][0]:.2f}; unstructured (free-LoRA/NoMAD) collapse "
                   f"up to {worst_free:.2f}. " +
                   ("PARETO WIN (structured adaptation stays accurate + collapse-safe; "
                    "matched-param unstructured collapses)"
                    if acc_gain >= 0 and cadence_safe and free_collapses else
                    "no clean Pareto win yet — inspect trajectory / stream length"))
    print(f"\n  VERDICT: {verdict}\n")

    (RESULTS / "indy_stream.json").write_text(json.dumps({
        "sessions": [d[0] for d in data], "n_chan": int(n_chan),
        "held_in": args.held_in, "revisits": args.revisits, "seeds": args.seeds,
        "collapse_floor": args.collapse_floor,
        "metrics": {m: {k: ci[m][k] for k in METRICS} for m in per},
        "verdict": verdict,
    }, indent=2))
    _pareto_figure(ci, args.collapse_floor)
    print(f"figure: results/indy_stream_pareto.png; data: results/indy_stream.json\n")


def _pareto_figure(ci, floor):
    plt.figure(figsize=(6.5, 5))
    for m in ci:
        x, xl, xh = ci[m]["cumulative"]
        y, yl, yh = ci[m]["collapse"]
        plt.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
                     fmt="o", capsize=3, label=m)
        plt.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("cumulative online velocity R² (higher better)")
    plt.ylabel(f"collapse-rate (R²<{floor}) — lower better")
    plt.title("Indy continual stream: accuracy × stability")
    plt.grid(alpha=0.3)
    plt.gca().invert_yaxis()                                  # up-and-right = best corner
    plt.tight_layout()
    plt.savefig(RESULTS / "indy_stream_pareto.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
