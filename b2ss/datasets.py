"""Real-EEG loader: PhysioNet EEG Motor Movement/Imagery Database (Schalk 2004).

Loaded via MNE's mne.datasets.eegbci (hosted on physionet.org). Default task is
left-fist vs right-fist *motor execution* (runs 3, 7, 11; T1=left, T2=right) —
64-channel, 160 Hz, a real closed-loop-relevant 2-class problem. This removes the
circularity of the synthetic testbed: nobody planted a CV->window law here.

The CV proxy (mu peak frequency) is computed per subject from the sensorimotor
channels of its own trials — no MRI/TMS needed. See proxies.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 160.0
FIST_RUNS = (3, 7, 11)                 # left vs right fist, motor execution
SENSORIMOTOR = ("C3", "Cz", "C4", "C1", "C2", "FC3", "FC4", "CP3", "CP4")
# Commonly flagged as inconsistently sampled/timed in downstream analyses.
BAD_SUBJECTS = (88, 89, 92, 100)


@dataclass
class SubjectEEG:
    subject: int
    X: np.ndarray          # (n_trials, n_chan, n_time) microvolts
    y: np.ndarray          # (n_trials,) 0=left fist, 1=right fist
    fs: float
    ch_names: list
    sm_idx: np.ndarray     # indices of sensorimotor channels (for the mu proxy)


def default_cohort(n: int = 10) -> list[int]:
    return [s for s in range(1, 110) if s not in BAD_SUBJECTS][:n]


def load_subject(subject: int, *, runs=FIST_RUNS, l_freq=7.0, h_freq=30.0,
                 tmin=0.5, tmax=3.5, verbose=False) -> SubjectEEG:
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf, concatenate_raws
    mne.set_log_level("WARNING" if verbose else "ERROR")

    raws = [read_raw_edf(f, preload=True)
            for f in eegbci.load_data(subject, list(runs), update_path=True, verbose=verbose)]
    raw = concatenate_raws(raws)
    eegbci.standardize(raw)                       # fixes 'C3..' -> 'C3' etc.
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.filter(l_freq, h_freq, verbose=verbose)

    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    picks = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    epochs = mne.Epochs(raw, events, event_id=picks, tmin=tmin, tmax=tmax,
                        baseline=None, picks="eeg", preload=True, verbose=verbose)

    X = epochs.get_data(copy=True) * 1e6          # V -> microvolts
    n_time = int(round((tmax - tmin) * FS))
    X = X[:, :, :n_time].astype(np.float32)       # trim to an exact length
    y = (epochs.events[:, 2] == picks["T2"]).astype(np.int64)  # T2=right -> 1

    ch = epochs.ch_names
    sm = np.array([ch.index(c) for c in SENSORIMOTOR if c in ch], dtype=int)
    return SubjectEEG(subject, X, y, FS, ch, sm)


def crop_windows(X: np.ndarray, y: np.ndarray, win: int, stride: int):
    """Cropped-window augmentation (Schirrmeister et al. 2017): split each trial
    into overlapping windows to give deep nets enough training samples on
    small-trial EEG. Returns (Xw (m, C, win), yw (m,), trial_idx (m,))."""
    Xw, yw, tid = [], [], []
    for i, (xt, yt) in enumerate(zip(X, y)):     # xt: (C, T)
        for s in range(0, xt.shape[1] - win + 1, stride):
            Xw.append(xt[:, s:s + win]); yw.append(yt); tid.append(i)
    return (np.asarray(Xw, np.float32), np.asarray(yw), np.asarray(tid))


def trial_vote(logits: np.ndarray, trial_idx: np.ndarray, n_trials: int) -> np.ndarray:
    """Aggregate per-window logits into one prediction per trial (summed softmax)."""
    votes = np.zeros((n_trials, logits.shape[1]))
    np.add.at(votes, trial_idx, logits)
    return votes.argmax(1)


def zscore_train(X_train: np.ndarray, X_test: np.ndarray):
    """Per-channel standardisation using TRAIN stats only (no leakage)."""
    mu = X_train.mean(axis=(0, 2), keepdims=True)
    sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return ((X_train - mu) / sd).astype(np.float32), ((X_test - mu) / sd).astype(np.float32)


if __name__ == "__main__":
    # Smoke test: load one subject, report shapes + class balance + mu proxy.
    from .proxies import mu_peak_frequency
    s = load_subject(1, verbose=True)
    print(f"subject {s.subject}: X={s.X.shape} y={np.bincount(s.y)} "
          f"chans={len(s.ch_names)} sm={len(s.sm_idx)}")
    flat = s.X[:, s.sm_idx].transpose(1, 0, 2).reshape(len(s.sm_idx), -1)
    print(f"mu peak (sensorimotor) ~= {mu_peak_frequency(flat, s.fs):.2f} Hz")
