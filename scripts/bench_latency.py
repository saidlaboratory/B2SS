#!/usr/bin/env python3
"""Measure real-time inference latency vs the proposal's <50 ms budget (§4.7).

Times single-window (batch=1) forward passes for the default and proposal-size
decoders, in three configs: eager FP32, dynamically-quantized (qint8 Linear —
the proposal's "FP16 quantization / head pruning" mitigation family), and
TorchScript. Reports p50/p95/p99. CPU here; the proposal targets ONNX+CUDA on an
RTX 4080, which is faster — so CPU numbers are an upper bound.

    python scripts/bench_latency.py
    python scripts/bench_latency.py --proposal-size --iters 300
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b2ss.model import DecoderConfig, proposal_config, B2SSDecoder, count_params

BUDGET_MS = 50.0


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return np.array(ts)


def _report(name, ts):
    p50, p95, p99 = np.percentile(ts, [50, 95, 99])
    ok = "OK " if p95 < BUDGET_MS else "OVER"
    print(f"  {name:<22} p50={p50:6.2f}  p95={p95:6.2f}  p99={p99:6.2f} ms  "
          f"[{ok} vs {BUDGET_MS:.0f} ms budget]")


@torch.no_grad()
def bench(cfg, iters, warmup, threads):
    torch.set_num_threads(threads)
    m = B2SSDecoder(cfg).eval()
    x = torch.randn(1, cfg.n_chan, cfg.win)
    cv = torch.tensor(55.0)
    print(f"\n{cfg.gate_mode} decoder | d_model={cfg.d_model} layers={cfg.num_layers} "
          f"| {count_params(m):,} params | {threads} thread(s)")

    _report("eager fp32", _time(lambda: m(x, cv), iters, warmup))

    try:
        mq = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8).eval()
        _report("dynamic-quant int8", _time(lambda: mq(x, cv), iters, warmup))
    except Exception as e:  # quantization can be unsupported for some ops
        print(f"  dynamic-quant int8     skipped ({type(e).__name__})")

    try:
        ms = torch.jit.trace(m, (x, cv)).eval()
        _report("torchscript", _time(lambda: ms(x, cv), iters, warmup))
    except Exception as e:
        print(f"  torchscript            skipped ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--proposal-size", action="store_true")
    ap.add_argument("--threads", type=int, default=1, help="1 = realistic single-stream latency")
    args = ap.parse_args()

    print(f"Real-time inference latency (batch=1, CPU) — budget {BUDGET_MS:.0f} ms")
    bench(DecoderConfig(gate_mode="cv"), args.iters, args.warmup, args.threads)
    if args.proposal_size:
        bench(proposal_config(gate_mode="cv"), args.iters, args.warmup, args.threads)
    print("\nNote: CPU eager is the slow path; proposal targets ONNX Runtime + CUDA "
          "on an RTX 4080. Quantization/TorchScript shown as portable mitigations.\n")


if __name__ == "__main__":
    main()
