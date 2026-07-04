#!/usr/bin/env python3
"""Sensitivity sweep (P5): is the CV-gate benefit robust to hyperparameters, or a
knife-edge artifact of one setting?

On the heterogeneous-CV synthetic regime (where CV is genuine information), sweep
each knob one-at-a-time and report the cv-vs-learned MSE gap (positive = CV helps).
A robust mechanism keeps the gap clearly > 0 across all settings.

    python scripts/run_sensitivity.py            # a few minutes on CPU
    python scripts/run_sensitivity.py --quick
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

from b2ss import model as M
from b2ss.data import make_heterogeneous
from b2ss.eval import mse
from b2ss.model import DecoderConfig, B2SSDecoder
from b2ss.stats import mean_ci
from b2ss.train import fit, predict

RESULTS = Path(__file__).resolve().parent.parent / "results"


def gap(cfg_over, mod_over, seeds, n_train):
    """mean (learned_mse - cv_mse) over seeds on heterogeneous data; >0 => CV helps."""
    saved = {k: getattr(M, k) for k in mod_over}
    for k, v in mod_over.items():
        setattr(M, k, v)
    try:
        gaps, cvs, lrs = [], [], []
        for s in seeds:
            het = make_heterogeneous(n_train=n_train, n_test=400, seed=s)
            res = {}
            for mode in ("cv", "learned"):
                m = B2SSDecoder(DecoderConfig(**{**cfg_over, "gate_mode": mode}))
                fit(m, het.X_train, het.Y_train,
                    cv=(het.cv_train if mode == "cv" else None), epochs=40, seed=s)
                res[mode] = mse(predict(m, het.X_test,
                                        het.cv_test if mode == "cv" else None), het.Y_test)
            gaps.append(res["learned"] - res["cv"]); cvs.append(res["cv"]); lrs.append(res["learned"])
        return mean_ci(gaps), float(np.mean(cvs)), float(np.mean(lrs))
    finally:
        for k, v in saved.items():
            setattr(M, k, v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = list(range(1 if args.quick else args.seeds))
    RESULTS.mkdir(exist_ok=True)

    grid = {
        "MASK_GAMMA":     [("MASK_GAMMA", v) for v in (0.25, 0.5, 1.0)],
        "SPAN_FRAC_MAX":  [("SPAN_FRAC_MAX", v) for v in (0.4, 0.6, 0.8)],
        "ode_steps":      [("ode_steps", v) for v in (10, 20, 40)],
        "patch":          [("patch", v) for v in (1, 2, 5)],
    }
    module_knobs = {"MASK_GAMMA", "SPAN_FRAC_MAX"}

    print(f"\nSensitivity of the cv-vs-learned gap (heterogeneous CV, n_train={args.n_train}, "
          f"{len(seeds)} seeds). Gap>0 => CV helps.\n")
    out = {}
    for knob, settings in grid.items():
        print(f"{knob}:")
        out[knob] = {}
        for name, val in settings:
            cfg_over = {} if name in module_knobs else {name: val}
            mod_over = {name: val} if name in module_knobs else {}
            (g, lo, hi), cv_mse, lr_mse = gap(cfg_over, mod_over, seeds, args.n_train)
            out[knob][str(val)] = {"gap": g, "ci": [lo, hi], "cv_mse": cv_mse, "learned_mse": lr_mse}
            flag = "OK " if lo > 0 else "~"
            print(f"  {name}={val:<5}: gap={g:+.3f} [{lo:+.3f},{hi:+.3f}] {flag}"
                  f"(cv={cv_mse:.3f} learned={lr_mse:.3f})")

    (RESULTS / "sensitivity.json").write_text(json.dumps(out, indent=2))
    all_pos = all(v["ci"][0] > 0 for k in out for v in out[k].values())
    print(f"\nCV-gate benefit {'robust (all lower-CIs > 0)' if all_pos else 'NOT robust everywhere'} "
          f"across swept settings. -> results/sensitivity.json\n")


if __name__ == "__main__":
    main()
