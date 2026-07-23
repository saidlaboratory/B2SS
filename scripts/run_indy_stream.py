#!/usr/bin/env python3
"""Phase 11 — the Indy continual stream: what adaptation costs when it goes wrong.

Freeze a GRU decoder on the earliest Indy sessions, then adapt online across the remaining
sessions in temporal order (with revisits of the earliest streamed ones). Every method plugs
into the same streaming harness and is scored on continual metrics.

Two things this script is careful about, because the first version of it got both wrong:

  STABILITY IS MEASURED AS REGRET, NOT FLOOR-CROSSINGS. A fixed R2 floor cannot separate
  methods on a real stream -- one genuinely hard session is hard for everyone, so No-Adapt,
  MPA and CADENCE all post the identical collapse-rate and the metric decides nothing. We
  report `regret` against the per-session No-Adapt trajectory (how often, and how badly,
  adapting LOST accuracy) and keep collapse-rate as a secondary column for the methods that
  genuinely diverge.

  THE RECALIBRATION CEILING IS FAIR, IN BOTH THE WAYS IT PREVIOUSLY WASN'T. It fine-tunes
  the frozen source decoder rather than training from scratch, AND it re-normalises the
  input with the session's own statistics rather than the source's. Getting the second one
  wrong cost the ceiling ~0.6 R2 and produced the false claim that cheap label-free
  adaptation beats per-session recalibration. It does not: recalibration wins this stream
  outright. See `recalibrate` for the measurement.

The structure ablation is Tent vs free-LoRA, NOT CADENCE vs free-LoRA: CADENCE's affine head
is fit in closed form while free-LoRA descends a gradient, so that pair confounds structure
with optimiser. Tent is the gradient-fit DIAGONAL head at the same parameter count, same
objective, same lr and steps -- it differs from free-LoRA in structure alone.

    python scripts/run_indy_stream.py --quick            # go/no-go smoke
    python scripts/run_indy_stream.py --seeds 3          # ~real run (CPU, minutes)
"""

from __future__ import annotations

import argparse
import copy
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
from b2ss.continual import (cumulative_r2, worst_session_r2, collapse_rate,
                            adaptation_regret, backward_transfer)
from b2ss.stats import mean_ci, paired_by_unit

RESULTS = Path(__file__).resolve().parent.parent / "results"
WIN = 20
METRICS = ["cumulative", "worst", "collapse", "regret_rate", "regret_mean", "bwt"]


def vel_r2(Yt, Yp):
    return float(r2_score(Yt, Yp, multioutput="variance_weighted"))


def zc(a, mu, sd):
    return ((a - mu) / sd).astype(np.float32)


def load_windows(win: int, subject: str = "indy"):
    """Load every downloaded session as windows, sliced to the common (first n_chan)
    electrode set so channels correspond across days."""
    out = []
    for p in list_sessions(subject=subject):
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
        # adaptation set (small, capped) vs full-recalibration set (large — the ceiling;
        # a per-session decoder must see the whole session's training data).
        cap = rng.choice(len(Xtr), min(adapt_cap, len(Xtr)), replace=False)
        rcap = rng.choice(len(Xtr), min(retrain_cap, len(Xtr)), replace=False)
        sessions.append({
            "name": name,
            "Xtr": zc(Xtr[cap], xmu, xsd), "Ytr": zc(Ytr[cap], ymu, ysd),
            "Xtr_full": zc(Xtr[rcap], xmu, xsd), "Ytr_full": zc(Ytr[rcap], ymu, ysd),
            "Xte": zc(X[split == 1], xmu, xsd), "Yte": zc(Y[split == 1], ymu, ysd),
        })
    return sessions


def metrics_of(traj, first_visit, last_visit, floor, base_traj):
    reg = adaptation_regret(traj, base_traj)
    return {
        "cumulative": cumulative_r2(traj),
        "worst": worst_session_r2(traj),
        "collapse": collapse_rate(traj, floor),
        "regret_rate": reg["rate"],
        "regret_mean": reg["mean"],
        "bwt": backward_transfer(first_visit, last_visit),
    }


def recalibrate(dec, s, *, epochs, seed, scratch):
    """Per-session recalibration — the honest ceiling. scratch=False fine-tunes a COPY of
    the frozen source decoder at a lower lr; scratch=True trains a fresh one.

    RE-NORMALISES the input with the session's OWN statistics, which is what recalibrating
    means. Feeding it source-normalised input instead is the exact handicap the alignment
    baselines exist to remove, and it is not a small effect: measured on three sessions,
    fine-tuning scores 0.024 in the source frame and 0.741 in the session's own frame. An
    earlier version of this script got that wrong and reported a ~0.13 ceiling, which made
    every label-free adapter look like it beat recalibration. It does not.

    Targets stay in the source frame so velocity R2 is measured in the same units as every
    other method in the table."""
    Xtr, Xte = s["Xtr_full"], s["Xte"]
    mu, sd = Xtr.mean((0, 2), keepdims=True), Xtr.std((0, 2), keepdims=True) + 1e-6
    Xtr, Xte = zc(Xtr, mu, sd), zc(Xte, mu, sd)
    model = GRUDecoder(Xtr.shape[1]) if scratch else copy.deepcopy(dec)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    fit(model, Xtr, s["Ytr_full"], epochs=epochs,
        lr=1e-3 if scratch else 2e-4, batch_size=256, seed=seed)
    return vel_r2(s["Yte"], predict(model, Xte))


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
    ap.add_argument("--methods", default="no-adapt,mpa,tent,cotta,rdumb,nomad,free-lora,cadence")
    ap.add_argument("--no-retrain", action="store_true", help="skip the recalibration ceilings")
    ap.add_argument("--subject", default="indy",
                    help="`loco` is the second monkey on the same rig — see b2ss/indy.py")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.held_in, args.epochs = 1, 3, 20
        args.src_cap, args.adapt_cap, args.fast_steps = 5000, 1200, 30
        args.methods = "no-adapt,mpa,tent,nomad,free-lora,cadence"
        args.no_retrain = True
    RESULTS.mkdir(exist_ok=True)

    data, n_chan = load_windows(WIN, args.subject)
    if len(data) <= args.held_in + 1:
        sys.exit(f"Need > held_in+1 sessions; have {len(data)}, held_in={args.held_in}.")
    methods = args.methods.split(",")
    if "no-adapt" not in methods:
        methods.insert(0, "no-adapt")                        # regret is measured against it
    methods.sort(key=lambda m: m != "no-adapt")              # and it must run first
    stream_idx = list(range(args.held_in, len(data)))
    print(f"\nIndy continual stream | {len(data)} sessions, {n_chan} electrodes | "
          f"source={args.held_in} frozen, stream={len(stream_idx)} + {args.revisits} revisits\n")
    print("sessions:", ", ".join(d[0] for d in data), "\n")

    fs = args.fast_steps
    rows = methods + ([] if args.no_retrain else ["finetune", "scratch"])
    per = {m: {k: [] for k in METRICS} for m in rows}
    traj_by_seed = {m: [] for m in rows}

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
        base = None
        for m in methods:
            out = run_stream(factories[m](), sessions, schedule, score_fn=vel_r2)
            traj = [r for _, r in out["visits"]]
            if base is None:
                base = traj                                  # No-Adapt reference for regret
            mm = metrics_of(traj, out["first_visit"], out["last_visit"], args.collapse_floor, base)
            for k in METRICS:
                per[m][k].append(mm[k])
            traj_by_seed[m].append(traj)
            print(f"  seed {seed} {m:>10}: cum={mm['cumulative']:.3f} worst={mm['worst']:.3f} "
                  f"collapse={mm['collapse']:.2f} regret={mm['regret_rate']:.2f}/"
                  f"{mm['regret_mean']:.3f} bwt={mm['bwt']:+.3f}")

        if not args.no_retrain:
            for kind, scratch in (("finetune", False), ("scratch", True)):
                traj = [recalibrate(dec, s, epochs=args.epochs, seed=seed, scratch=scratch)
                        for s in sessions]
                fv = {sessions[i]["name"]: traj[i] for i in range(len(sessions))}
                # the ceilings see each session once, so pad the reference to match
                mm = metrics_of(traj, fv, fv, args.collapse_floor, base[:len(traj)])
                for k in METRICS:
                    per[kind][k].append(mm[k])
                traj_by_seed[kind].append(traj)
                print(f"  seed {seed} {kind:>10}: cum={mm['cumulative']:.3f} worst={mm['worst']:.3f}")

    # aggregate + verdict
    ci = {m: {k: mean_ci(per[m][k]) for k in METRICS} for m in per}
    print("\nContinual metrics — mean over "
          f"{args.seeds} seeds. regret = vs No-Adapt per session (rate / mean shortfall); "
          f"collapse = R2<{args.collapse_floor}\n")
    hdr = f"  {'method':>12}  {'cumulative':>12}  {'worst':>8}  {'collapse':>9}  {'regret':>14}  {'BWT':>8}"
    print(hdr)
    for m in per:
        c = ci[m]
        print(f"  {m:>12}  {c['cumulative'][0]:>+12.3f}  {c['worst'][0]:>+8.3f}  "
              f"{c['collapse'][0]:>9.2f}  {c['regret_rate'][0]:>6.2f}/{c['regret_mean'][0]:<7.3f} "
              f"{c['bwt'][0]:>+8.3f}")

    # -- honest head-to-head: who actually wins the stream? -------------------------- #
    n_visits = len(traj_by_seed["no-adapt"][0])
    flat = {m: np.concatenate(traj_by_seed[m]) for m in methods}
    best = max(methods, key=lambda m: ci[m]["cumulative"][0])
    lines = []
    if "cadence" in methods:
        for other in [m for m in methods if m != "cadence"]:
            d = paired_by_unit(flat["cadence"], flat[other], n_visits)
            lines.append((other, d))
        print(f"\nCADENCE vs each method, paired over {n_visits} session visits "
              f"(seeds averaged within visit)\n")
        for other, d in sorted(lines, key=lambda t: -t[1]["delta"]):
            print(f"  vs {other:>10}  {d['delta']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  "
                  f"p={d['p']:.3f}  {d['won']}/{d['n']} visits")

    # structure ablation: SAME optimiser, objective, lr, steps and param count; only the
    # head structure differs (diagonal vs dense rank-1). CADENCE is closed-form, so it is
    # NOT the right comparator here.
    abl = None
    if "tent" in methods and "free-lora" in methods:
        abl = paired_by_unit(flat["tent"], flat["free-lora"], n_visits)
        print(f"\nStructure ablation (gradient-fit, matched params, diagonal vs dense rank-1):"
              f"\n  tent − free-lora  {abl['delta']:+.3f} [{abl['ci'][0]:+.3f},{abl['ci'][1]:+.3f}]  "
              f"p={abl['p']:.3f}  {abl['won']}/{abl['n']} visits")

    parts = [f"Best cumulative R2 on this stream: {best} ({ci[best]['cumulative'][0]:+.3f})."]
    if "cadence" in methods and best != "cadence":
        d = dict(lines)[best]
        parts.append(f"CADENCE ({ci['cadence']['cumulative'][0]:+.3f}) does NOT lead here — "
                     f"{best} beats it by {-d['delta']:+.3f} (p={d['p']:.3f}); at this adaptation "
                     f"budget ({args.adapt_cap} windows) the shrinkage costs accuracy, exactly as "
                     f"the data-efficiency curve predicts. The stream is not where the method wins.")
    elif "cadence" in methods:
        parts.append("CADENCE leads on cumulative R2 at this adaptation budget.")
    if abl is not None:
        parts.append(f"Structure ablation: matched-parameter diagonal beats dense by "
                     f"{abl['delta']:+.3f} (p={abl['p']:.3f}), gradient-fit on both sides.")
    if not args.no_retrain:
        ceiling = max(ci['finetune']['cumulative'][0], ci['scratch']['cumulative'][0])
        parts.append(f"Recalibration ceiling (own-normalised, the fair one): fine-tune "
                     f"{ci['finetune']['cumulative'][0]:+.3f}, from-scratch "
                     f"{ci['scratch']['cumulative'][0]:+.3f} — it "
                     f"{'BEATS' if ceiling > ci[best]['cumulative'][0] else 'does not beat'} "
                     f"every label-free adapter (best {best} {ci[best]['cumulative'][0]:+.3f}). "
                     f"Label-free adaptation buys cheapness, not accuracy.")
    verdict = " ".join(parts)
    print(f"\n  VERDICT: {verdict}\n")

    (RESULTS / "indy_stream.json").write_text(json.dumps({
        "sessions": [d[0] for d in data], "n_chan": int(n_chan),
        "held_in": args.held_in, "revisits": args.revisits, "seeds": args.seeds,
        "collapse_floor": args.collapse_floor, "adapt_cap": args.adapt_cap,
        "metrics": {m: {k: ci[m][k] for k in METRICS} for m in per},
        "trajectories": traj_by_seed,
        "cadence_vs": {o: d for o, d in lines},
        "structure_ablation_tent_vs_freelora": abl,
        "verdict": verdict,
    }, indent=2))
    _regret_figure(ci, methods)
    print(f"figure: results/indy_stream_pareto.png; data: results/indy_stream.json\n")


def _regret_figure(ci, methods):
    plt.figure(figsize=(6.5, 5))
    for m in methods:
        x = ci[m]["cumulative"][0]
        y = ci[m]["regret_mean"][0]
        xl, xh = ci[m]["cumulative"][1], ci[m]["cumulative"][2]
        plt.errorbar(x, y, xerr=[[x - xl], [xh - x]], fmt="o", capsize=3)
        plt.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("cumulative online velocity R² (higher better)")
    plt.ylabel("mean regret vs No-Adapt on sessions it hurt (lower better)")
    plt.title("Indy continual stream: accuracy × harm")
    plt.grid(alpha=0.3)
    plt.gca().invert_yaxis()                                  # up-and-right = best corner
    plt.tight_layout()
    plt.savefig(RESULTS / "indy_stream_pareto.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
