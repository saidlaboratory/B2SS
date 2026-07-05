"""Continuous intracortical benchmark data: NLB MC_Maze_Small (monkey maze reach).

Decode 2-D hand velocity from motor-cortex spikes — the *regression* regime the
B2SS decoder is actually built for (unlike the small-trial EEG classification).
Read directly with pynwb (nlb_tools pins pandas <=1.3.4, unusable on py3.12).

Download once (30 MB) with:
    curl -sSL "https://api.dandiarchive.org/api/dandisets/000140/versions/draft/\
assets/7821971e-c6a4-4568-8773-1bfa205c13f8/download/" -o \
      ~/b2ss_data/000140/mc_maze_small_train.nwb
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NWB_PATH = Path.home() / "b2ss_data" / "000140" / "mc_maze_small_train.nwb"
CACHE = NWB_PATH.with_suffix(".binned.npz")
BIN_S = 0.020            # 20 ms bins


@dataclass
class MazeData:
    spikes: np.ndarray       # (n_bins, n_units) counts
    vel: np.ndarray          # (n_bins, 2) hand velocity
    split: np.ndarray        # (n_bins,) 0=train, 1=val, -1=outside any trial
    rt: np.ndarray           # (n_bins,) reaction time (ms) of the containing trial; nan outside
    fs: float                # bins/s


def _bin_nwb(path: Path, bin_s: float) -> MazeData:
    """Bin spikes AND velocity onto one real-time grid. Critical: hand_vel is NOT
    uniformly sampled (inter-trial gaps removed), so velocity must be placed by its
    own timestamps, not by row index — otherwise it desyncs from the spike clock."""
    from pynwb import NWBHDF5IO
    from scipy.stats import binned_statistic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nwb = NWBHDF5IO(str(path), "r", load_namespaces=True).read()
        hv = nwb.processing["behavior"]["hand_vel"]
        vt = np.asarray(hv.timestamps[:])
        vd = np.asarray(hv.data[:], dtype=np.float64)             # (T_samp, 2)
        edges = np.arange(0.0, float(vt[-1]) + bin_s, bin_s)
        n = len(edges) - 1
        vx = binned_statistic(vt, vd[:, 0], "mean", bins=edges)[0]
        vy = binned_statistic(vt, vd[:, 1], "mean", bins=edges)[0]
        vel = np.nan_to_num(np.stack([vx, vy], 1)).astype(np.float32)  # (n_bins, 2)
        units = nwb.units
        spikes = np.zeros((n, len(units.id)), dtype=np.float32)
        for i in range(len(units.id)):
            spikes[:, i] = np.histogram(np.asarray(units["spike_times"][i]), bins=edges)[0]
        tr = nwb.trials
        starts, stops = tr["start_time"][:], tr["stop_time"][:]
        splits = tr["split"][:].astype(str)
        rts = tr["rt"][:].astype(float)
        centers = (np.arange(n) + 0.5) * bin_s
        split = np.full(n, -1, np.int8)
        rt = np.full(n, np.nan, np.float32)
        for s0, s1, sp, r in zip(starts, stops, splits, rts):
            m = (centers >= s0) & (centers < s1)
            split[m] = 0 if sp == "train" else 1
            rt[m] = r
    return MazeData(spikes, vel, split, rt, 1.0 / bin_s)


def load_maze(bin_s: float = BIN_S, path: Path = NWB_PATH) -> MazeData:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — download MC_Maze_Small (see module docstring).")
    if CACHE.exists():
        z = np.load(CACHE)
        return MazeData(z["spikes"], z["vel"], z["split"], z["rt"], float(z["fs"]))
    d = _bin_nwb(path, bin_s)
    np.savez(CACHE, spikes=d.spikes, vel=d.vel, split=d.split, rt=d.rt, fs=d.fs)
    return d


def inject_group_latency(spikes: np.ndarray, n_groups: int = 8,
                         max_delay_bins: int = 8, seed: int = 0):
    """Inject a KNOWN per-group conduction latency into real binned spikes (Phase 8
    bridge). Units are split into `n_groups`; each group's spike train is delayed by
    an integer δ_g bins (output[t]=input[t-δ_g], zero-filled) — scrambling the
    cross-group temporal alignment. Returns (shifted_spikes, align_delays) where
    align_delays[c] = -δ_group(c) are the delays a decoder must apply to UNDO the
    injection (i.e. the ground-truth "measured CV" for the aligner)."""
    rng = np.random.default_rng(seed)
    n, C = spikes.shape
    group = np.arange(C) % n_groups
    deltas = rng.integers(0, max_delay_bins + 1, n_groups)      # δ_g >= 0, bins
    out = np.zeros_like(spikes)
    for c in range(C):
        d = int(deltas[group[c]])
        out[d:, c] = spikes[:n - d, c] if d else spikes[:, c][:n]
        if d == 0:
            out[:, c] = spikes[:, c]
    align_delays = (-deltas[group]).astype(np.float32)          # undo shifts (<=0)
    return out.astype(np.float32), align_delays


def shift_channels(spikes: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Integer per-channel time shift: out[t,c] = spikes[t-delays[c], c] (zero-filled).
    Matches ChannelDelay for integer delays. Used to *align* injected data by its
    measured (undo) delays back to a common conduction frame."""
    n, C = spikes.shape
    out = np.zeros_like(spikes)
    for c in range(C):
        d = int(round(float(delays[c])))
        if d > 0:
            out[d:, c] = spikes[:n - d, c]
        elif d < 0:
            out[:n + d, c] = spikes[-d:, c]
        else:
            out[:, c] = spikes[:, c]
    return out


def make_windows(d: MazeData, win: int, smooth_sigma: float = 3.0):
    """Sliding windows over the continuous stream, assigned to the split of their
    last bin. Spikes are Gaussian-smoothed into firing rates first (standard for
    motor decoding; sigma in bins). X: (m, n_units, win); Y: (m, 2); split/rt."""
    from scipy.ndimage import gaussian_filter1d
    S = gaussian_filter1d(d.spikes, smooth_sigma, axis=0) if smooth_sigma > 0 else d.spikes
    V, n = d.vel, S.shape[0]
    idx = np.arange(win - 1, n)
    idx = idx[d.split[idx] >= 0]                                # window end inside a trial
    X = np.stack([S[t - win + 1:t + 1].T for t in idx]).astype(np.float32)  # (m, units, win)
    Y = V[idx].astype(np.float32)
    return X, Y, d.split[idx], d.rt[idx]


if __name__ == "__main__":
    d = load_maze()
    print(f"binned: spikes {d.spikes.shape}, vel {d.vel.shape}, fs {d.fs} Hz")
    print(f"train/val bins: {(d.split==0).sum()}/{(d.split==1).sum()}")
    X, Y, sp, rt = make_windows(d, win=20)
    print(f"windows: X {X.shape}, Y {Y.shape}, train/val {(sp==0).sum()}/{(sp==1).sum()}")
    print(f"vel range: [{Y.min():.0f}, {Y.max():.0f}], rt range: [{np.nanmin(rt):.0f}, {np.nanmax(rt):.0f}] ms")
