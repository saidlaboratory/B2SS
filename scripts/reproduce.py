#!/usr/bin/env python3
"""One command to regenerate everything (P4).

Runs the test suite and every experiment, writing figures + JSON to results/.
Deterministic: each experiment seeds torch/numpy per run (see b2ss.train.fit and
the per-seed loops in the scripts).

    python scripts/reproduce.py             # full (includes the slow real-EEG benchmark)
    python scripts/reproduce.py --fast      # skip downloads/long runs (tests + synthetic only)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(desc, cmd):
    print(f"\n{'='*70}\n▶ {desc}\n  $ {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    status = "OK" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"  [{status}] {desc} — {dt:.0f}s", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="tests + synthetic only; skip downloads")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    py = sys.executable

    steps = [
        ("Test suite", [py, "-m", "pytest", "tests/", "-q"]),
        ("Statistical harness (power/ICC/mixed-model)", [py, "-m", "b2ss.stats"]),
        ("Real-time latency vs 50 ms budget", [py, "scripts/bench_latency.py", "--proposal-size"]),
        ("Synthetic offline comparison (Experiment-3 shape)",
         [py, "scripts/run_offline_comparison.py", "--subjects", "7"]),
        ("Gate ablation (Study A prior / Study B information)",
         [py, "scripts/run_ablation.py", "--seeds", str(args.seeds)]),
        ("Sensitivity sweep", [py, "scripts/run_sensitivity.py", "--seeds", "3"]),
    ]
    if not args.fast:
        steps.append(("Real-EEG benchmark (PhysioNet; downloads on first run)",
                      [py, "scripts/run_real_benchmark.py", "--seeds", "3"]))
        if (ROOT / "scripts" / "run_intracortical_benchmark.py").exists():
            steps.append(("Intracortical benchmark (DANDI MC_Maze; downloads)",
                          [py, "scripts/run_intracortical_benchmark.py"]))
        # Phase 11 (CADENCE): transfer brackets, the Indy continual stream, and the
        # decomposition figure (which consumes the two brackets' JSON — run it last).
        for desc, script, extra in [
            ("Phase 11: calibration spectrum (injected MC_Maze)", "run_transfer_modes.py", []),
            ("Phase 11: real cross-session transfer (MC_Maze S/M/L)", "run_xsession.py", []),
            ("Phase 11: Indy continual stream (downloads Indy on first run)",
             "run_indy_stream.py", ["--seeds", "3"]),
            ("Phase 11: drift-decomposition figure", "run_decomposition_figure.py", []),
        ]:
            if (ROOT / "scripts" / script).exists():
                steps.append((desc, [py, f"scripts/{script}", *extra]))

    results = [(d, run(d, c)) for d, c in steps]
    print(f"\n{'='*70}\nREPRODUCE SUMMARY\n{'='*70}")
    for d, ok in results:
        print(f"  {'✓' if ok else '✗'} {d}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
