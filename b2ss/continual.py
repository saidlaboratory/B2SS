"""Continual-stream metrics for lifelong test-time adaptation.

All operate on the per-visit velocity R² trajectory a streaming run produces
(one R² per session visit, in stream order). Kept pure (no torch, no data) so the
metric definitions are auditable independently of the harness.
"""

from __future__ import annotations

import numpy as np


def cumulative_r2(r2_by_visit) -> float:
    """Mean online R² over every session visit in the stream."""
    return float(np.mean(r2_by_visit))


def worst_session_r2(r2_by_visit) -> float:
    return float(np.min(r2_by_visit))


def collapse_rate(r2_by_visit, floor: float) -> float:
    """Fraction of visits whose R² fell below `floor` (a-priori collapse threshold)."""
    x = np.asarray(r2_by_visit, float)
    return float((x < floor).mean())


def adaptation_regret(r2_by_visit, r2_no_adapt) -> dict:
    """How often, and how badly, adapting is worse than leaving the decoder alone.

    Replaces collapse-rate as the stability metric. A fixed R² floor cannot tell the
    methods apart on a real stream — a hard session is hard for everyone, so No-Adapt,
    MPA and CADENCE all score the identical floor-crossing rate and the metric decides
    nothing. Regret is measured against the per-session No-Adapt trajectory instead, so
    it isolates the harm the adapter itself causes:

      rate  — fraction of visits where adapting lost accuracy
      mean  — mean shortfall on those visits (0 if none; positive = magnitude of harm)
      worst — largest single-visit shortfall
    """
    a = np.asarray(r2_by_visit, float)
    b = np.asarray(r2_no_adapt, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    loss = np.maximum(b - a, 0.0)
    hit = loss > 0
    return {"rate": float(hit.mean()),
            "mean": float(loss[hit].mean()) if hit.any() else 0.0,
            "worst": float(loss.max())}


def backward_transfer(first_visit: dict, revisit: dict) -> float:
    """Mean R² change on revisited sessions: revisit − first_visit (negative =
    forgetting). Averaged over sessions present in both dicts."""
    keys = [k for k in first_visit if k in revisit]
    if not keys:
        return 0.0
    return float(np.mean([revisit[k] - first_visit[k] for k in keys]))


def _selfcheck() -> None:
    traj = [0.7, 0.6, 0.2, 0.65]
    assert abs(cumulative_r2(traj) - 0.5375) < 1e-9
    assert worst_session_r2(traj) == 0.2
    assert collapse_rate(traj, floor=0.3) == 0.25          # one of four below 0.3
    assert collapse_rate(traj, floor=0.0) == 0.0

    # regret: adapting helped on 2 of 4 visits, hurt on 2 (by 0.1 and 0.3)
    base = [0.6, 0.7, 0.5, 0.6]
    r = adaptation_regret(traj, base)                       # traj = [.7, .6, .2, .65]
    assert r["rate"] == 0.5 and abs(r["mean"] - 0.2) < 1e-9 and abs(r["worst"] - 0.3) < 1e-9
    # a method that never loses to No-Adapt scores zero on every field
    assert adaptation_regret(base, base) == {"rate": 0.0, "mean": 0.0, "worst": 0.0}
    # the metric collapse-rate cannot see: same floor-crossings, very different regret
    assert collapse_rate(traj, 0.3) == collapse_rate([0.7, 0.6, 0.2, 0.6], 0.3)
    assert adaptation_regret(traj, base)["worst"] > adaptation_regret(base, base)["worst"]

    # forgetting: revisit worse than first visit on both -> negative BWT
    bwt = backward_transfer({"a": 0.7, "b": 0.6}, {"a": 0.5, "b": 0.6})
    assert abs(bwt - (-0.1)) < 1e-9, bwt
    assert backward_transfer({"a": 0.5}, {"z": 0.9}) == 0.0  # no overlap
    print("continual.py self-check OK: cumulative/worst/collapse/regret/BWT")


if __name__ == "__main__":
    _selfcheck()
