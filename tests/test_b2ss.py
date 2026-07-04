"""Runnable checks — each module's self-check plus a couple of cross-cutting asserts.

    python3 -m pytest tests/ -q
"""

import numpy as np
import torch

from b2ss import cv as cv_mod
from b2ss import data as data_mod
from b2ss import model as model_mod
from b2ss import eval as eval_mod
from b2ss import train as train_mod


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


def test_matched_capacity():
    b2ss, ctrl = model_mod.make_pair()
    pb, pc = model_mod.count_params(b2ss), model_mod.count_params(ctrl)
    assert abs(pb - pc) / pb < 0.01


def test_tau_bounds_and_monotonicity():
    b2ss, _ = model_mod.make_pair()
    for cvv in (25.0, 47.5, 70.0):
        t = b2ss.tau_ms(torch.tensor(cvv)).item()
        assert 20.0 - 1e-4 <= t <= 100.0 + 1e-4
    assert b2ss.tau_ms(torch.tensor(65.0)) < b2ss.tau_ms(torch.tensor(30.0))


def test_cv_structure_is_exploitable():
    """Oracle with the correct per-subject width beats a fixed width — the
    necessary condition for the CV-modulated decoder to have anything to learn."""
    correct, fixed = [], []
    for i, cv in enumerate(data_mod.sample_cvs(8, seed=3)):
        s = data_mod.make_subject(cv, n_train=50, n_test=200, seed=100 + i)
        correct.append(data_mod._mse(
            data_mod.oracle_predict(s.X_test, s.A, s.B, s.width), s.Y_test))
        fixed.append(data_mod._mse(
            data_mod.oracle_predict(s.X_test, s.A, s.B, data_mod.W_MAX), s.Y_test))
    assert np.mean(correct) < np.mean(fixed)
