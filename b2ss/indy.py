"""Monkey-Indy self-paced grid-reach loader (O'Doherty/Makin/Sabes).

The long, labeled, multi-session intracortical stream CADENCE evaluates on — the
same data MPA and NoMAD report on. Self-paced continuous reaching (no trials), so
the train/val split is chronological, not trial-based. Download the .mat sessions
(HDF5 v7.3) from Zenodo record 583331 into INDY_DIR:

    ~/b2ss_data/indy/indy_YYYYMMDD_NN.mat   (one file per session)

Each file holds: spike-time cells per (channel, unit), cursor position (finger_pos
or cursor_pos), and a timestamp vector `t`. Velocity is the finite difference of
cursor position ON THE BIN GRID (placed by its own timestamps — never row index,
which desyncs from the spike clock; cf. the MC_Maze hand_vel fix).
"""

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
    (chronologically) = val (1), rest = train (0); rt all-nan (no trials).
    """
    from scipy.stats import binned_statistic
    cursor_t = np.asarray(cursor_t, float)
    edges = np.arange(float(cursor_t[0]), float(cursor_t[-1]) + bin_s, bin_s)
    n = len(edges) - 1
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


def list_sessions(indy_dir: Path = INDY_DIR, subject: str = "indy"):
    """Sessions for one subject, in date order (the filenames sort chronologically).

    The same Zenodo record carries a second monkey, `loco` (10 sessions), on the same rig
    and the same 96-electrode Utah array — so replicating any stream result on a second
    subject is a download plus `--subject loco`, no code. We did not run it here: the loco
    files are ~1.1-1.6 GB each (vs ~120 MB for indy) because they carry a second array, so
    the set is ~12 GB. That is a real gap in the evidence, not a solved problem — a
    single-subject result is a single-subject result.
    """
    return sorted(Path(indy_dir).glob(f"{subject}_*.mat"))


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
    """Read spikes + cursor position/time from an Indy .mat (HDF5 v7.3).

    `spikes` is a (n_units, n_chan) cell of spike-time vectors. We return ONE
    multiunit spike train per CHANNEL (concatenating all units on that electrode) —
    the fixed Utah-array electrode is the identity that corresponds across days,
    whereas unit sorting drifts. Placeholder/empty cells fall outside the cursor time
    window and are dropped by the windowed binning, so no special-casing is needed.
    Returns (spike_times: list[np.ndarray] len n_chan, cursor_pos: (T,2), cursor_t: (T,))."""
    import h5py
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"]).squeeze()
        pos = np.asarray(f["cursor_pos"]).T[:, :2].astype(float)   # 2-D screen cursor
        refs = f["spikes"]                               # (n_units, n_chan)
        n_units, n_chan = refs.shape
        spike_times = []
        for ch in range(n_chan):
            per_unit = [np.atleast_1d(np.asarray(f[refs[u, ch]]).squeeze()).astype(float)
                        for u in range(n_units)]
            spike_times.append(np.concatenate(per_unit) if per_unit else np.empty(0))
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
