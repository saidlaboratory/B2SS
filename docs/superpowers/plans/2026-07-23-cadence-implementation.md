# CADENCE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the software core for CADENCE — a frozen decoder + composed low-DOF test-time adapter evaluated on a lifelong session stream — through the Week-4 go/no-go, plus the drift-decomposition fallback (already ~80% in hand).

**Architecture:** A frozen GRU backbone (`baselines.GRUDecoder`) is wrapped by a composed adapter: a slow EMA conduction anchor (`transfer.ConductionDelayAligner`) + a fast borrowed representation head + a collapse-sensing revert controller. A shared streaming harness replays real intracortical sessions (monkey-Indy, ported from the existing MC_Maze pynwb loader) in temporal order with revisits; every method (No-Adapt, Tent, CoTTA, RDumb, free-LoRA, CADENCE, and the paper-gated MPA/NoMAD) plugs into the same harness and is scored on continual metrics (cumulative / worst-session R², collapse-rate, backward-transfer).

**Tech Stack:** Python 3.10+, PyTorch 2.x (CPU), NumPy, SciPy, scikit-learn, pynwb + h5py (data loaders), Matplotlib. No new heavy dependencies.

## Global Constraints

- **Python ≥3.10, CPU-only.** Everything must run on CPU on public data. Copy the existing pattern (`torch`, `numpy`, `sklearn.metrics.r2_score` variance-weighted).
- **Test convention = the repo's, not pytest fixtures.** Every new module gets a `_selfcheck()` with `assert`s and a `if __name__ == "__main__": _selfcheck()`; wire it into `tests/test_b2ss.py` as `def test_<module>_selfcheck(): <module>_mod._selfcheck()`. No new test frameworks (matches the 21 existing checks).
- **No new dependencies without cause.** numpy/torch/scipy/sklearn/pynwb/h5py/matplotlib are available; adding anything else needs a one-line justification (ponytail).
- **Honesty is load-bearing.** Nulls are reported straight (e.g. `best_delta_fit_gain` printed even when negative). No result is tuned toward a win. The conduction term's ~0 accuracy on real gaps is stated, not hidden.
- **Git identity = Quang Bui.** Commits and any GitHub text carry no Claude/AI attribution, no `Co-Authored-By`. Commit messages are terse and human.
- **Velocity R² = `r2_score(Y, pred, multioutput="variance_weighted")`** everywhere, for comparability with `run_xsession.py` / `run_transfer_modes.py`.
- **Reuse first:** `MazeData`, `make_windows`, `GRUDecoder`, `TransferNormalizer`, `ConductionDelayAligner`, `fractional_shift`, `train.fit/predict`, `stats.mean_ci`, `source_feature_stats`, `inject_group_latency`, `shift_channels` already exist — the plan wires them together, it does not re-implement them.

## File Structure

- `b2ss/continual.py` — **new.** Pure continual-stream metrics over per-session R² (cumulative, worst-session, collapse-rate, backward-transfer). No torch, no data.
- `b2ss/indy.py` — **new.** Monkey-Indy grid-reach loader → `MazeData` (reuses the dataclass + `make_windows` from `intracortical.py`). Chronological train/val split (no trials → split by time). Synthetic self-check (no download in CI).
- `b2ss/stream.py` — **new.** The streaming harness: an `Adapter` protocol + `run_stream(adapter, sessions, schedule)` that replays sessions with revisits and returns the per-visit R² trajectory the metrics consume.
- `b2ss/cadence.py` — **new.** The CADENCE adapter: slow EMA conduction anchor + fast representation head + collapse-sensing revert. Implements the `Adapter` protocol.
- `b2ss/tta_baselines.py` — **new.** No-Adapt, Tent (re-derived for the BN-free GRU regressor), CoTTA (mean-teacher + restore), RDumb (periodic reset), free-LoRA (matched-param unstructured adapter). All implement the `Adapter` protocol.
- `b2ss/ibci_baselines.py` — **new, paper-gated.** MPA (arXiv 2606.14866) and NoMAD (Nat Commun 2025) faithful reimplementations. Interface + validation contract here; internals gated on reading the papers (Task 7).
- `scripts/run_indy_stream.py` — **new.** The headline experiment: loader → harness → all methods → metrics → Pareto figure + JSON.
- `scripts/run_pareto_figure.py` — **new.** Assembles the accuracy×stability plane from the JSON outputs (headline figure).
- `tests/test_b2ss.py` — **modify.** Add one `test_<module>_selfcheck()` line per new module + a few cross-cutting asserts.

Tasks 1–6 are fully specified below. Tasks 7–8 are contracted follow-ons (interfaces + acceptance criteria) because they gate on reading source papers and on the Week-4 go/no-go verdict; they are deliberately not fabricated line-by-line.

---

### Task 1: Continual-stream metrics (`b2ss/continual.py`)

The pure scoring layer everything else reports through. No data, no torch — start here (fast, verifiable, unblocks the harness).

**Files:**
- Create: `b2ss/continual.py`
- Modify: `tests/test_b2ss.py`

**Interfaces:**
- Produces:
  - `cumulative_r2(r2_by_visit: list[float]) -> float` — mean online R² over the stream.
  - `worst_session_r2(r2_by_visit: list[float]) -> float` — min.
  - `collapse_rate(r2_by_visit: list[float], floor: float) -> float` — fraction of visits with R² < floor.
  - `backward_transfer(first_visit: dict[str, float], revisit: dict[str, float]) -> float` — mean over revisited session names of `revisit[name] - first_visit[name]` (negative = forgetting).

- [ ] **Step 1: Write the failing self-check**

In `b2ss/continual.py`:

```python
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
```

- [ ] **Step 2: Run it, verify it passes**

Run: `python3 -m b2ss.continual`
Expected: `continual.py self-check OK: ...`

- [ ] **Step 3: Wire into the test suite**

In `tests/test_b2ss.py`, add `from b2ss import continual as continual_mod` with the other imports, and:

```python
def test_continual_selfcheck():
    continual_mod._selfcheck()
```

- [ ] **Step 4: Run the suite**

Run: `python3 -m pytest tests/ -q`
Expected: all green (22 checks).

- [ ] **Step 5: Commit**

```bash
git add b2ss/continual.py tests/test_b2ss.py
git commit -m "Add continual-stream metrics (cumulative/worst/collapse/BWT)"
```

---

### Task 2: Streaming harness (`b2ss/stream.py`)

The engine every method plugs into: replay sessions in order with revisits, adapt, score, record the trajectory.

**Files:**
- Create: `b2ss/stream.py`
- Modify: `tests/test_b2ss.py`

**Interfaces:**
- Consumes: `continual` metrics (for the self-check only).
- Produces:
  - `Adapter` protocol (duck-typed): `.adapt(X, Y=None) -> None` and `.predict(X) -> np.ndarray`. `Y=None` means unlabeled/self-supervised adaptation.
  - `run_stream(adapter, sessions, schedule, *, score_fn) -> dict` where `sessions: list[dict]` each `{"name", "Xtr", "Ytr", "Xte", "Yte"}`, `schedule: list[int]` is session indices in visit order (revisits = repeated indices), `score_fn(Y_true, Y_pred) -> float`. Returns `{"visits": [(name, r2), ...], "first_visit": {name: r2}, "last_visit": {name: r2}}`.

- [ ] **Step 1: Write the harness + failing self-check**

In `b2ss/stream.py`:

```python
"""Online streaming harness for lifelong test-time adaptation.

Replays a list of sessions in a given visit order (revisits = repeated indices).
At each visit the adapter adapts on that session's train split (unlabeled unless a
mode passes labels) and is scored on its test split BEFORE the next visit — so the
recorded R² is genuinely online. The backbone stays frozen inside the adapter; the
harness never touches decoder weights. Every method (No-Adapt/Tent/CoTTA/CADENCE/...)
is a duck-typed Adapter with .adapt(X, Y=None) and .predict(X)."""

from __future__ import annotations


def run_stream(adapter, sessions, schedule, *, score_fn):
    visits, first_visit, last_visit = [], {}, {}
    for idx in schedule:
        s = sessions[idx]
        adapter.adapt(s["Xtr"])                       # unlabeled online adaptation
        r2 = score_fn(s["Yte"], adapter.predict(s["Xte"]))
        visits.append((s["name"], r2))
        first_visit.setdefault(s["name"], r2)
        last_visit[s["name"]] = r2
    return {"visits": visits, "first_visit": first_visit, "last_visit": last_visit}


def _selfcheck() -> None:
    import numpy as np
    from .continual import cumulative_r2, backward_transfer

    # a toy adapter that "improves" a fixed prediction toward the labels it never sees
    # by shrinking a bias each adapt() call — enough to exercise trajectory bookkeeping.
    class Toy:
        def __init__(self):
            self.bias = 2.0
        def adapt(self, X, Y=None):
            self.bias *= 0.5
        def predict(self, X):
            return np.full((len(X), 1), self.bias, dtype=np.float32)

    def score(Yt, Yp):                                # 1 - |mean error|, toy monotone score
        return float(1.0 - abs(Yt.mean() - Yp.mean()))

    sess = [{"name": n, "Xtr": np.zeros((4, 3)), "Ytr": np.zeros((4, 1)),
             "Xte": np.zeros((2, 3)), "Yte": np.zeros((2, 1))} for n in ("a", "b")]
    out = run_stream(Toy(), sess, schedule=[0, 1, 0, 1], score_fn=score)
    assert [v[0] for v in out["visits"]] == ["a", "b", "a", "b"]
    # revisits score higher than first visits (bias shrank) -> positive BWT
    assert backward_transfer(out["first_visit"], out["last_visit"]) > 0
    assert 0.0 <= cumulative_r2([r for _, r in out["visits"]]) <= 1.0
    print("stream.py self-check OK: schedule order, first/last visit, BWT bookkeeping")


if __name__ == "__main__":
    _selfcheck()
```

- [ ] **Step 2: Run it**

Run: `python3 -m b2ss.stream`
Expected: `stream.py self-check OK: ...`

- [ ] **Step 3: Wire into the suite** (`test_stream_selfcheck`), same pattern as Task 1.

- [ ] **Step 4: Run** `python3 -m pytest tests/ -q` → green.

- [ ] **Step 5: Commit**

```bash
git add b2ss/stream.py tests/test_b2ss.py
git commit -m "Add online streaming harness (adapt/score/revisit trajectory)"
```

---

### Task 3: Monkey-Indy loader (`b2ss/indy.py`)

Port the intracortical loader to the long Indy grid-reach stream. This is the tap-root; treat the download as W1–2 work. The self-check runs on a synthetic in-memory session (no download), so CI stays green offline; a `__main__` path loads a real file when present.

**Files:**
- Create: `b2ss/indy.py`
- Modify: `tests/test_b2ss.py`

**Interfaces:**
- Consumes: `MazeData`, `make_windows` from `b2ss.intracortical`.
- Produces:
  - `bin_session(spike_times: list[np.ndarray], cursor_pos: np.ndarray, cursor_t: np.ndarray, bin_s: float, val_frac: float) -> MazeData` — pure binning core (spikes → per-channel counts; velocity = finite-difference of cursor position on the bin grid; `split` = chronological last-`val_frac` as val; `rt` = all-nan). Unit-testable without any file.
  - `load_indy_session(path, bin_s=0.02, val_frac=0.2) -> MazeData` — read one Indy `.mat` (HDF5 v7.3 via `h5py`), extract spike-time cells + cursor position/time, call `bin_session`. Cache to `path.with_suffix(".binned.npz")` like `intracortical.load_maze`.
  - `INDY_DIR` (default `Path.home()/"b2ss_data"/"indy"`), `list_sessions() -> list[Path]` (sorted `indy_*.mat`), download docstring (Zenodo record 583331).

- [ ] **Step 1: Write the loader with a synthetic self-check**

Key correctness point to encode (mirrors the MC_Maze `hand_vel` bug the repo already caught): **velocity is the finite difference of cursor position placed on the bin grid by its own timestamps, not by row index.** In `b2ss/indy.py`:

```python
"""Monkey-Indy self-paced grid-reach loader (O'Doherty/Makin/Sabes).

The long, labeled, multi-session intracortical stream CADENCE evaluates on — the
same data MPA and NoMAD report on. Self-paced continuous reaching (no trials), so
the train/val split is chronological, not trial-based. Download the .mat sessions
(HDF5 v7.3) from Zenodo record 583331 into INDY_DIR:

    ~/b2ss_data/indy/indy_YYYYMMDD_NN.mat   (one file per session)

Each file holds: spike-time cells per (channel, unit), cursor position (finger_pos
or cursor_pos), and a timestamp vector `t`. Velocity is the finite difference of
cursor position ON THE BIN GRID (placed by its own timestamps — never row index,
which desyncs from the spike clock; cf. the MC_Maze hand_vel fix)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .intracortical import MazeData, make_windows  # reuse dataclass + windower

INDY_DIR = Path.home() / "b2ss_data" / "indy"
BIN_S = 0.020


def bin_session(spike_times, cursor_pos, cursor_t, bin_s: float = BIN_S,
                val_frac: float = 0.2) -> MazeData:
    """Bin spikes and cursor velocity onto one real-time grid.

    spike_times: list of 1-D arrays (one per channel/unit), spike times in seconds.
    cursor_pos:  (T_samp, 2) cursor/finger position.  cursor_t: (T_samp,) its times.
    Velocity = d(pos)/dt evaluated on bin centres.  split: last `val_frac` of bins
    (chronologically) = val (1), rest = train (0); rt all-nan (no trials)."""
    from scipy.stats import binned_statistic
    cursor_t = np.asarray(cursor_t, float)
    edges = np.arange(float(cursor_t[0]), float(cursor_t[-1]) + bin_s, bin_s)
    n = len(edges) - 1
    centers = edges[:-1] + bin_s / 2
    pos = np.stack([binned_statistic(cursor_t, np.asarray(cursor_pos)[:, k], "mean",
                                     bins=edges)[0] for k in (0, 1)], 1)
    pos = _fill_nan_forward(pos)
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / bin_s                # finite-diff velocity on grid
    vel = np.nan_to_num(vel).astype(np.float32)
    C = len(spike_times)
    spikes = np.zeros((n, C), dtype=np.float32)
    for c, st in enumerate(spike_times):
        if len(st):
            spikes[:, c] = np.histogram(np.asarray(st, float), bins=edges)[0]
    split = np.zeros(n, np.int8)
    split[int(round(n * (1 - val_frac))):] = 1           # chronological val tail
    rt = np.full(n, np.nan, np.float32)
    return MazeData(spikes, vel, split, rt, 1.0 / bin_s)


def _fill_nan_forward(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN rows (empty position bins) so finite-diff velocity is finite."""
    out = a.copy()
    for k in range(out.shape[1]):
        col = out[:, k]
        idx = np.where(~np.isnan(col))[0]
        if len(idx):
            col[: idx[0]] = col[idx[0]]
            for i in range(1, len(col)):
                if np.isnan(col[i]):
                    col[i] = col[i - 1]
    return out


def list_sessions(indy_dir: Path = INDY_DIR):
    return sorted(indy_dir.glob("indy_*.mat"))


def load_indy_session(path, bin_s: float = BIN_S, val_frac: float = 0.2) -> MazeData:
    path = Path(path)
    cache = path.with_suffix(".binned.npz")
    if cache.exists():
        z = np.load(cache)
        return MazeData(z["spikes"], z["vel"], z["split"], z["rt"], float(z["fs"]))
    spike_times, pos, t = _read_indy_mat(path)
    d = bin_session(spike_times, pos, t, bin_s, val_frac)
    np.savez(cache, spikes=d.spikes, vel=d.vel, split=d.split, rt=d.rt, fs=d.fs)
    return d


def _read_indy_mat(path: Path):
    """Read spike-time cells + cursor position/time from an Indy .mat (HDF5 v7.3).
    Returns (spike_times: list[np.ndarray], cursor_pos: (T,2), cursor_t: (T,))."""
    import h5py
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"]).squeeze()
        pos_key = "finger_pos" if "finger_pos" in f else "cursor_pos"
        pos = np.asarray(f[pos_key]).T[:, :2].astype(float)
        refs = f["spikes"]                               # (n_unit, n_chan) cell of refs
        spike_times = []
        for col in range(refs.shape[1]):
            for row in range(refs.shape[0]):
                st = np.asarray(f[refs[row, col]]).squeeze()
                spike_times.append(np.atleast_1d(st).astype(float))
    return spike_times, pos, t


def _selfcheck() -> None:
    # synthetic session: 3 channels, a cursor moving on a smooth path; velocity on the
    # bin grid must recover finite differences and windows must have the right shapes.
    rng = np.random.default_rng(0)
    dur, bin_s = 20.0, 0.02
    ct = np.arange(0, dur, 0.004)
    pos = np.stack([np.sin(ct), np.cos(ct)], 1)          # smooth circular path
    spike_times = [np.sort(rng.uniform(0, dur, k)) for k in (300, 120, 0)]  # incl. a silent ch
    d = bin_session(spike_times, pos, ct, bin_s, val_frac=0.2)
    n = int(dur / bin_s)
    assert d.spikes.shape[0] == d.vel.shape[0] and d.spikes.shape[1] == 3
    assert abs(d.spikes.shape[0] - n) <= 2, d.spikes.shape
    assert (d.split == 1).mean() > 0.15 and (d.split == 0).any()   # chronological split
    assert np.isfinite(d.vel).all() and d.vel.std() > 0            # velocity recovered
    assert d.spikes[:, 2].sum() == 0                               # silent channel stays zero
    X, Y, split, _ = make_windows(d, win=20)
    assert X.shape[1] == 3 and X.shape[2] == 20 and Y.shape[1] == 2
    assert (split == 0).any() and (split == 1).any()
    print(f"indy.py self-check OK: binned {d.spikes.shape}, vel std {d.vel.std():.2f}, "
          f"windows X {X.shape}")


if __name__ == "__main__":
    _selfcheck()
```

- [ ] **Step 2: Run the synthetic self-check**

Run: `python3 -m b2ss.indy`
Expected: `indy.py self-check OK: ...`. (No download needed — the self-check is synthetic.)

- [ ] **Step 3: Wire into the suite** (`test_indy_selfcheck`).

- [ ] **Step 4: Validate against one REAL session** (manual, gated on download; not a CI test)

Download one `indy_*.mat` into `~/b2ss_data/indy/`, then:
Run: `python3 -c "from b2ss.indy import list_sessions, load_indy_session; d=load_indy_session(list_sessions()[0]); print(d.spikes.shape, d.vel.shape, d.fs)"`
Expected: a plausible shape (thousands of bins × ~100–200 channels) and finite velocity. If `finger_pos`/`spikes` keys differ in the real file, fix `_read_indy_mat` keys here — the binning core (`bin_session`) is already validated by Step 1 and does not change.

- [ ] **Step 5: Run the suite + commit**

```bash
python3 -m pytest tests/ -q
git add b2ss/indy.py tests/test_b2ss.py
git commit -m "Add monkey-Indy grid-reach loader (chronological split, grid-placed velocity)"
```

---

### Task 4: Frozen backbone + No-Adapt / RDumb / free-LoRA adapters (`b2ss/tta_baselines.py`)

The floor, the credibility gate, and the make-or-break ablation partner. All implement the `Adapter` protocol so they drop into `run_stream`.

**Files:**
- Create: `b2ss/tta_baselines.py`
- Modify: `tests/test_b2ss.py`

**Interfaces:**
- Consumes: `GRUDecoder` (`b2ss.baselines`), `train.predict`, `ConductionDelayAligner` / `fractional_shift` (`b2ss.transfer` / `b2ss.model`).
- Produces:
  - `NoAdapt(decoder)` — `.adapt` is a no-op; `.predict` = frozen decoder. Accuracy floor / stability ceiling.
  - `RDumb(decoder, adapter_factory, reset_every: int)` — wraps any adapter, resets it every `reset_every` visits (NeurIPS 2023 credibility gate).
  - `FreeLoRA(decoder, rank: int, lr, steps)` — an unstructured low-rank input-space adapter with a param count matched to CADENCE; adapts by the same unsupervised objective (Task 6) so the only difference from CADENCE is *structure*. The make-or-break ablation.

- [ ] **Step 1: Write `NoAdapt` + `FreeLoRA` + `RDumb` with a failing self-check**

Signatures and the shared unsupervised objective (latent-moment match on the frozen `decoder.head` hook — reuse `transfer.source_feature_stats` for the source moments). `NoAdapt` first (trivial, anchors the protocol):

```python
"""Test-time-adaptation baselines for the CADENCE stream, all sharing the Adapter
protocol (.adapt(X, Y=None) / .predict(X)) so they plug into b2ss.stream.run_stream.

The backbone is frozen in every method; only adapter-owned parameters move. Tent and
CoTTA are RE-DERIVED for a BN-free GRU velocity regressor (no entropy/BN to exploit),
using the same source-latent-moment matching self-supervision the whole study uses,
so the comparison isolates the adaptation MECHANISM, not the objective."""

from __future__ import annotations

import numpy as np
import torch

from .train import predict as _predict


class NoAdapt:
    """Frozen decoder; adaptation is a no-op. Accuracy floor / stability ceiling."""
    def __init__(self, decoder):
        self.decoder = decoder.eval()
    def adapt(self, X, Y=None):
        pass
    def predict(self, X):
        return _predict(self.decoder, X)
```

`FreeLoRA` and `RDumb` follow in the same file (full code authored during execution against the Task 6 objective interface — the objective function name/signature is fixed there: `unsup_objective(decoder, x_batch, src_mean, src_var) -> torch.Tensor`).

The self-check builds a tiny `GRUDecoder(n_chan=6)`, confirms: (a) `NoAdapt.predict` equals a raw `train.predict`; (b) `NoAdapt.adapt` leaves decoder weights byte-identical; (c) `FreeLoRA`'s trainable-param count is within ±20% of a reference CADENCE adapter's (the matched-param invariant — assert it explicitly so the ablation stays fair); (d) `RDumb` zeroes its inner adapter's params on the reset visit.

- [ ] **Step 2–5:** run `python3 -m b2ss.tta_baselines`, wire `test_tta_baselines_selfcheck`, run suite, commit `"Add No-Adapt/RDumb/free-LoRA TTA baselines (matched-param ablation partner)"`.

---

### Task 5: CADENCE adapter (`b2ss/cadence.py`)

The method: slow EMA conduction anchor + fast representation head + collapse-sensing revert.

**Files:**
- Create: `b2ss/cadence.py`
- Modify: `tests/test_b2ss.py`

**Interfaces:**
- Consumes: `ConductionDelayAligner`, `TransferNormalizer`, `source_feature_stats` (`b2ss.transfer`); `fractional_shift` (`b2ss.model`); the `unsup_objective` (Task 6).
- Produces:
  - `unsup_objective(decoder, x_batch, src_mean, src_var) -> torch.Tensor` — the delay-sensitive label-free objective (cross-covariance/lag form; upgrades the CORAL-moment version that fails at 0.302 in `run_transfer_modes.py`). Lives here because CADENCE is its primary consumer; imported by Tasks 4/6.
  - `CADENCE(decoder, n_chan, *, n_groups=8, max_delay=12, rank=4, anchor_ema=0.95, fast_lr=0.05, collapse_z=3.0, src_stats)` implementing the `Adapter` protocol, where:
    - **slow anchor** = a `ConductionDelayAligner`, updated toward each session's fitted delays by EMA `anchor_ema` (low plasticity → holds the slow timing component);
    - **fast head** = a rank-`rank` input-space representation aligner, re-fit per session by `fast_lr` steps of `unsup_objective`;
    - **controller** = `.adapt` watches the objective + latent-norm; if the post-fit objective z-scores above `collapse_z` vs its running history, the fast head is reset toward the anchor state (`collapse_revert()`), guaranteeing the anchor is the safe floor.

- [ ] **Step 1: Write `unsup_objective` + `CADENCE` with a failing self-check**

The self-check reuses the `transfer.py` identifiability rig (a fixed decoder reading the window centre; a target = source shifted by a KNOWN per-group delay). Assertions: (a) after `.adapt` on shifted-but-unlabeled target data, velocity error is lower than `NoAdapt` on the same target (the composed adapter helps where a shift exists); (b) the frozen decoder weights are byte-identical before/after (freeze invariant); (c) injecting an artificial divergence (a huge random fast-head state) triggers `collapse_revert()` and restores the objective below the collapse threshold (controller works); (d) the anchor moves by ≤ `(1-anchor_ema)`× the per-session delta in one step (EMA slowness invariant).

- [ ] **Step 2–5:** run `python3 -m b2ss.cadence`, wire `test_cadence_selfcheck`, run suite, commit `"Add CADENCE adapter (EMA conduction anchor + fast head + collapse revert)"`.

---

### Task 6: Tent + CoTTA (`b2ss/tta_baselines.py`, extend)

The continual-TTA references, re-derived for the BN-free GRU regressor against the shared `unsup_objective`.

**Files:**
- Modify: `b2ss/tta_baselines.py`, `tests/test_b2ss.py`

**Interfaces:**
- Produces:
  - `Tent(decoder, lr, steps)` — adapts only the affine params of the decoder's `head`/normalisation via `steps` of `unsup_objective` per visit, *carried forward* across the stream (the canonical error-accumulation foil; no reset).
  - `CoTTA(decoder, lr, steps, ema, restore_p)` — mean-teacher (EMA of the adapted params) + stochastic restoration (`restore_p` fraction of params reset to source each step). The anti-forgetting reference.

- [ ] **Step 1:** self-check asserts each adapts on the identifiability rig (objective decreases over `.adapt` calls) and that `Tent` accumulates drift over a long synthetic shift stream while `CoTTA`/`CADENCE` do not (a direct, unit-level rehearsal of the headline contrast). Steps 2–5 as before; commit `"Add Tent + CoTTA continual-TTA baselines (re-derived for GRU regressor)"`.

---

### Task 7: Headline experiment script (`scripts/run_indy_stream.py`) + Pareto figure

Wires loader → harness → all methods → metrics → JSON + the accuracy×stability figure. This is the artifact that produces the Week-4 go/no-go read and the final headline.

**Files:**
- Create: `scripts/run_indy_stream.py`, `scripts/run_pareto_figure.py`

**Interfaces:**
- Consumes: `indy.list_sessions/load_indy_session`, `intracortical.make_windows`, `stream.run_stream`, `continual.*`, `cadence.CADENCE`, `tta_baselines.*`, `baselines.GRUDecoder`, `train.fit`, `stats.mean_ci`, `r2_score(variance_weighted)`.
- Produces: `results/indy_stream.json` (`{method: {cumulative, worst, collapse_rate, bwt} with mean_ci}` over ≥5 seeds) and `results/indy_stream_pareto.png` (x=cumulative R², y=collapse-rate, points per method + CIs).

- [ ] **Step 1:** Script structure (mirror `run_xsession.py`): load + window every Indy session; z-score using source-pool stats; `fit` + **freeze** a `GRUDecoder` on the earliest K sessions; build the visit schedule (temporal order + revisits of the earliest sessions); for each method build its adapter over the frozen decoder and `run_stream`; compute the four continual metrics; aggregate with `mean_ci`; print a table + `best gain over No-Adapt` and a `VERDICT`; write JSON + figure. Include `--quick`, `--seeds`, `--held-in`, `--revisits`, `--collapse-floor` flags.

- [ ] **Step 2 (GO/NO-GO gate):** Run `python3 scripts/run_indy_stream.py --quick` with only `NoAdapt`, `Tent`, `CoTTA`. **Decision:** if `Tent`/`CoTTA` collapse-rate ≫ `NoAdapt` over the full stream → proceed to the full battery. If free-TTA stays stable on Indy → switch the headline to the decomposition benchmark (Task 8) and record the negative in `RESULTS.md`. Commit `"Add Indy stream benchmark + Pareto figure (go/no-go harness)"`.

---

### Task 8: Paper-gated reimplementations + decomposition + write-up (contracted)

These gate on (a) the Week-4 go/no-go verdict and (b) reading the source papers, so they are contracted here rather than fabricated. Each is a real task with an interface and an acceptance criterion:

- **`b2ss/ibci_baselines.py` — MPA (arXiv 2606.14866) + NoMAD (Nat Commun 2025).** Interface: both implement the `Adapter` protocol over the frozen decoder. Acceptance: each reproduces its published *qualitative* regime on a sanity slice (MPA improves over No-Adapt on a single shifted session; NoMAD's unsupervised alignment recovers > No-Adapt cross-session) before being trusted as a baseline. Budget: the critical path (~2.5 wk). Never dropped from the battery.
- **Decomposition brackets (reuse).** Run the existing `scripts/run_transfer_modes.py` (timing-dominated, injected — already positive 0.649 vs 0.403) and `scripts/run_xsession.py` (representation-dominated, real — already null −0.015) as the two brackets; add their marginals to the paper's decomposition figure. No new code beyond a small plotting helper.
- **EEG breadth (reuse).** `scripts/run_moabb_transfer.py` with a frozen EEGNet + EA/Riemannian + T-TIME as the secondary stream.
- **Multi-seed + write-up.** ≥5 seeds across methods×streams via `stats.mean_ci`/FDR; resolve the matched-param free-LoRA ablation verdict; assemble the two headline figures; write the 4-page paper leading with the conceded conduction null.

Acceptance for the plan as a whole: `python3 -m pytest tests/ -q` green with all new self-checks; `run_indy_stream.py --quick` produces a JSON + figure; the go/no-go verdict is recorded in `RESULTS.md`.

---

## Self-Review

**Spec coverage.** Frozen backbone → Task 4 (`NoAdapt` over `GRUDecoder`). Slow anchor + fast head + controller → Task 5. Streaming/revisit protocol → Task 2 + Task 7. Continual metrics → Task 1. Free-TTA battery → Tasks 4+6. Matched-param free-LoRA ablation → Task 4 (invariant asserted). MPA/NoMAD → Task 8. Decomposition brackets + EEG breadth + write-up → Task 8. Indy arena → Task 3 + Task 7. Go/no-go → Task 7 Step 2. All spec sections map to a task.

**Placeholder scan.** Tasks 1–3 carry complete code. Tasks 4–6 carry the anchoring code (`NoAdapt`, protocol, objective signature) verbatim and specify the remaining classes by exact signature + self-check contract — deliberately, because their bodies depend on the fixed `unsup_objective` interface defined in Task 5; the executor writes them against that signature. Task 8 is explicitly contracted (paper-gated), not a hidden placeholder.

**Type consistency.** `Adapter` protocol (`.adapt(X, Y=None)`/`.predict(X)`) is used identically in Tasks 2, 4, 5, 6, 7. `unsup_objective(decoder, x_batch, src_mean, src_var)` is defined once (Task 5) and consumed by Tasks 4 and 6. `MazeData`/`make_windows` reused unchanged from `intracortical.py`. Velocity R² is the variance-weighted `r2_score` throughout.
