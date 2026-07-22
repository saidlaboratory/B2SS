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
    # forgetting: revisit worse than first visit on both -> negative BWT
    bwt = backward_transfer({"a": 0.7, "b": 0.6}, {"a": 0.5, "b": 0.6})
    assert abs(bwt - (-0.1)) < 1e-9, bwt
    assert backward_transfer({"a": 0.5}, {"z": 0.9}) == 0.0  # no overlap
    print("continual.py self-check OK: cumulative/worst/collapse/BWT")


if __name__ == "__main__":
    _selfcheck()
