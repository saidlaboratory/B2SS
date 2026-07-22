#!/usr/bin/env python3
"""Phase 11 — the drift-decomposition figure.

Partitions the cross-session transfer gap into a TIMING/conduction component and a
REPRESENTATION component, using the two already-run intracortical brackets (same
velocity-R² units, so the conduction marginal is directly comparable):

  timing-dominated  = injected-latency MC_Maze (results/transfer_modes.json):
                      zero-shot measured-CV normalization vs no-norm  -> large + marginal
  representation-   = real cross-session MC_Maze S/M/L (results/xsession.json):
    dominated         best data-driven delta-fit vs no-norm          -> ~null marginal

The figure is the honest core of the decomposition claim: conduction normalization
helps ONLY where timing is the gap; the real multi-session gap is representation drift.
EEG (Zhou2016) is reported as a secondary null in RESULTS, in accuracy units.

    python scripts/run_decomposition_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b2ss.stats import mean_ci

RESULTS = Path(__file__).resolve().parent.parent / "results"


def paired_marginal(json_path, better_key, base_key="no-norm", field="velocity_r2"):
    """Per-seed paired difference (better − base) → (mean, lo, hi)."""
    d = json.loads(Path(json_path).read_text())[field]
    diff = np.asarray(d[better_key], float) - np.asarray(d[base_key], float)
    return mean_ci(diff)


def main():
    for f in ("transfer_modes.json", "xsession.json"):
        if not (RESULTS / f).exists():
            sys.exit(f"missing {RESULTS/f} — run the corresponding benchmark first.")

    timing = paired_marginal(RESULTS / "transfer_modes.json", "zero-shot")
    repr_ = paired_marginal(RESULTS / "xsession.json", "few-100")
    brackets = [
        ("timing-dominated\n(injected MC_Maze)", timing),
        ("representation-dominated\n(real MC_Maze S/M/L)", repr_),
    ]

    print("\nConduction/timing marginal (Δ velocity R² over no-norm), mean [95% CI]:\n")
    for name, (m, lo, hi) in brackets:
        print(f"  {name.splitlines()[0]:>22}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\n  => conduction normalization helps where timing dominates, is null on the "
          "real representation-drift gap.\n")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs = np.arange(len(brackets))
    ms = [b[1][0] for b in brackets]
    err = [[b[1][0] - b[1][1] for b in brackets], [b[1][2] - b[1][0] for b in brackets]]
    colors = ["#1f6feb", "#8b949e"]
    ax.bar(xs, ms, yerr=err, capsize=5, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in brackets])
    ax.set_ylabel("conduction marginal — Δ velocity R² vs no-norm")
    ax.set_title("Drift decomposition: conduction helps only where timing dominates")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "decomposition.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"figure: results/{out.name}\n")


if __name__ == "__main__":
    main()
