#!/usr/bin/env python3
"""Experiment-3 offline comparison: B2SS vs matched-capacity control decoder.

Trains both decoders per synthetic subject and reports the proposal's metrics
(MSE, Pearson r, effective latency). This exercises the software end to end on
data that *contains* a CV->latency structure; it is not evidence for H1-H6.

    python scripts/run_offline_comparison.py --subjects 6 --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b2ss.data import make_subject, sample_cvs, spread_cvs
from b2ss.eval import mse, pearson_r, xcorr_lag, samples_to_ms
from b2ss.model import DecoderConfig, proposal_config, B2SSDecoder
from b2ss.train import train_decoder, predict, decode_continuous


def run_subject(cv, cfg, epochs, n_train, seed, device):
    sub = make_subject(cv, n_train=n_train, seed=seed)
    out = {}
    for name, gate in (("b2ss", True), ("control", False)):
        model = B2SSDecoder(DecoderConfig(**{**cfg.__dict__, "use_cv_gate": gate}))
        train_decoder(model, sub, epochs=epochs, seed=seed, device=device)
        pred = predict(model, sub.X_test, sub.cv, device)
        cont = decode_continuous(model, sub.cont_eeg, sub.cv, device)
        with torch.no_grad():
            tau = float(model.tau_ms(torch.tensor(sub.cv)))
        out[name] = {
            "mse": mse(pred, sub.Y_test),
            "r": pearson_r(pred, sub.Y_test),
            "lat_ms": samples_to_ms(xcorr_lag(cont, sub.cont_kin)),
            "tau_ms": tau,
        }
    out["cv"] = sub.cv
    out["width"] = sub.width
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=150,
                    help="training windows/subject (fewer surfaces the CV-prior benefit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proposal-size", action="store_true",
                    help="use full proposal dims (d_model 256, 8 heads, 4 layers); slow")
    ap.add_argument("--realistic", action="store_true",
                    help="draw a normal cohort instead of spanning the CV range")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = proposal_config() if args.proposal_size else DecoderConfig()
    cvs = (sample_cvs if args.realistic else spread_cvs)(args.subjects, seed=args.seed)

    print(f"\nB2SS offline comparison — {args.subjects} synthetic subjects, "
          f"{args.epochs} epochs, n_train={args.n_train}\n")
    hdr = f"{'subj':>4} {'CV':>6} {'width':>5} | {'MSE b2ss':>9} {'MSE ctrl':>9} " \
          f"{'r b2ss':>7} {'r ctrl':>7} {'lat b2ss':>8} {'lat ctrl':>8} {'tau b2ss':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for i, cv in enumerate(cvs):
        r = run_subject(cv, cfg, args.epochs, args.n_train, args.seed + i, args.device)
        rows.append(r)
        print(f"{i:>4} {r['cv']:>6.1f} {r['width']:>5d} | "
              f"{r['b2ss']['mse']:>9.4f} {r['control']['mse']:>9.4f} "
              f"{r['b2ss']['r']:>7.3f} {r['control']['r']:>7.3f} "
              f"{r['b2ss']['lat_ms']:>7.1f}m {r['control']['lat_ms']:>7.1f}m "
              f"{r['b2ss']['tau_ms']:>7.1f}m")

    b_mse = np.array([r["b2ss"]["mse"] for r in rows])
    c_mse = np.array([r["control"]["mse"] for r in rows])
    b_lat = np.array([r["b2ss"]["lat_ms"] for r in rows])
    c_lat = np.array([r["control"]["lat_ms"] for r in rows])
    wins = int(np.sum(b_mse < c_mse))
    mse_red = 100 * (c_mse.mean() - b_mse.mean()) / c_mse.mean()
    lat_red = 100 * (c_lat.mean() - b_lat.mean()) / (abs(c_lat.mean()) + 1e-9)

    print("\nSummary")
    print(f"  mean test MSE   : b2ss {b_mse.mean():.4f}  control {c_mse.mean():.4f}  "
          f"({mse_red:+.1f}% vs control)")
    print(f"  mean latency    : b2ss {b_lat.mean():.1f} ms  control {c_lat.mean():.1f} ms  "
          f"({lat_red:+.1f}% vs control)")
    print(f"  B2SS wins (MSE) : {wins}/{len(rows)} subjects")
    if len(rows) >= 3:
        t, p = stats.ttest_rel(b_mse, c_mse)
        print(f"  paired t-test   : t={t:.2f}, p={p:.4f} (B2SS vs control MSE)")
    print("\nNote: synthetic testbed. A win shows the decoder can exploit CV->latency "
          "structure when present; it is not evidence for the scientific hypotheses.\n")


if __name__ == "__main__":
    main()
