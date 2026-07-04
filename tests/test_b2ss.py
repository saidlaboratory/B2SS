"""Runnable checks — module self-checks plus cross-cutting asserts.

    python3 -m pytest tests/ -q
"""

import numpy as np
import torch

from b2ss import cv as cv_mod
from b2ss import data as data_mod
from b2ss import model as model_mod
from b2ss import eval as eval_mod
from b2ss import train as train_mod
from b2ss import proxies as proxies_mod
from b2ss import stats as stats_mod
from b2ss import baselines as baselines_mod


def test_cv_selfcheck():
    cv_mod._selfcheck()


def test_data_selfcheck():
    data_mod._selfcheck()


def test_model_selfcheck():
    model_mod._selfcheck()


def test_eval_selfcheck():
    eval_mod._selfcheck()


def test_train_selfcheck():
    train_mod._selfcheck()


def test_proxies_selfcheck():
    proxies_mod._selfcheck()


def test_stats_selfcheck():
    stats_mod._selfcheck()


def test_baselines_selfcheck():
    baselines_mod._selfcheck()


def test_gru_and_ridge_baselines():
    import torch
    from b2ss.baselines import GRUDecoder, ridge_r2
    x = torch.randn(8, 16, 20)
    g = GRUDecoder(16, n_out=2)
    assert g(x).shape == (8, 2)
    rng = np.random.default_rng(0)
    # linearly-decodable, overdetermined (600 samples >> 20 features) -> high R²
    X = rng.standard_normal((600, 4, 5)).astype("float32")
    W = rng.standard_normal((4 * 5, 2))
    Y = (X.reshape(600, -1) @ W).astype("float32")
    assert ridge_r2(X[:450], Y[:450], X[450:], Y[450:], alpha=0.1) > 0.9


def test_maze_windows():
    from b2ss.intracortical import MazeData, make_windows
    n = 200
    d = MazeData(spikes=np.random.default_rng(0).random((n, 8)).astype("float32"),
                 vel=np.random.default_rng(1).random((n, 2)).astype("float32"),
                 split=np.where(np.arange(n) < 150, 0, 1).astype("int8"),
                 rt=np.full(n, 300.0, "float32"), fs=50.0)
    d.split[:10] = -1                       # some bins outside any trial
    X, Y, sp, rt = make_windows(d, win=20)
    assert X.shape[1:] == (8, 20) and Y.shape[1] == 2
    assert (sp >= 0).all() and set(np.unique(sp)).issubset({0, 1})


def test_crop_windows_and_vote():
    from b2ss.datasets import crop_windows, trial_vote
    X = np.random.default_rng(0).standard_normal((5, 8, 100)).astype("float32")
    y = np.array([0, 1, 0, 1, 0])
    Xw, yw, tid = crop_windows(X, y, win=40, stride=20)
    assert Xw.shape[1:] == (8, 40) and len(Xw) == len(yw) == len(tid)
    assert tid.max() == 4 and (yw[tid == 2] == 0).all()
    # trial_vote reduces per-window logits to one prediction per trial
    logits = np.zeros((len(tid), 2)); logits[np.arange(len(tid)), yw] = 1.0
    assert (trial_vote(logits, tid, 5) == y).all()


def test_matched_capacity():
    b2ss, ctrl = model_mod.make_pair()
    pb, pc = model_mod.count_params(b2ss), model_mod.count_params(ctrl)
    assert abs(pb - pc) / pb < 0.01


def test_tau_bounds_and_monotonicity():
    b2ss, _ = model_mod.make_pair()
    with torch.no_grad():
        for cvv in (25.0, 47.5, 70.0):
            t = float(b2ss.tau_ms(torch.tensor(cvv), batch=1)[0])
            assert 20.0 - 1e-4 <= t <= 100.0 + 1e-4
        assert float(b2ss.tau_ms(torch.tensor(65.0), batch=1)) < \
            float(b2ss.tau_ms(torch.tensor(30.0), batch=1))


def test_all_gate_modes_forward():
    x = torch.randn(4, data_mod.N_CHAN, data_mod.WIN)
    cv = torch.full((4,), 55.0)
    for mode, m in model_mod.make_variants().items():
        out = m(x, cv)
        assert out.shape == (4, data_mod.N_KIN), mode


def test_cv_structure_is_exploitable():
    """Oracle with the correct per-subject width beats a fixed width — necessary
    condition for the CV-modulated decoder to have anything to learn."""
    correct, fixed = [], []
    for i, cv in enumerate(data_mod.sample_cvs(8, seed=3)):
        s = data_mod.make_subject(cv, n_train=50, n_test=200, seed=100 + i)
        correct.append(data_mod._mse(
            data_mod.oracle_predict(s.X_test, s.A, s.B, s.width), s.Y_test))
        fixed.append(data_mod._mse(
            data_mod.oracle_predict(s.X_test, s.A, s.B, data_mod.W_MAX), s.Y_test))
    assert np.mean(correct) < np.mean(fixed)


def test_heterogeneous_cv_is_information():
    """When CV varies per trial, per-trial CV beats the BEST single constant window
    -> CV carries information a learned constant cannot capture."""
    het = data_mod.make_heterogeneous(n_train=10, n_test=300, seed=5)
    per_trial = data_mod._mse(
        data_mod.oracle_predict(het.X_test, het.A, het.B,
                                [data_mod.cv_to_width(c) for c in het.cv_test]),
        het.Y_test)
    best_const = min(
        data_mod._mse(data_mod.oracle_predict(het.X_test, het.A, het.B, w), het.Y_test)
        for w in range(data_mod.W_MIN, data_mod.W_MAX + 1))
    assert per_trial < best_const
