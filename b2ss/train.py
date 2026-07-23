"""Training + inference (proposal 4.5): Adam, MSE/CE + L2, early stopping.

Works on raw arrays (regression or classification), so the same loop drives the
synthetic testbed, the ablations, and the real-EEG benchmark. CV is passed per
sample, so heterogeneous-CV data and per-subject-constant data use one code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .data import SubjectData, WIN
from .model import B2SSDecoder


@dataclass
class TrainResult:
    best_val: float
    epochs_run: int
    history: list[float]


def _t(a, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def _cv_array(cv, n: int, device):
    if cv is None:
        return None
    arr = np.full(n, float(cv)) if np.ndim(cv) == 0 else np.asarray(cv, float)
    return _t(arr, device)


def fit(model: B2SSDecoder, X, Y, *, cv=None, cv_sd=None, delays=None, epochs: int = 40,
        lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 128,
        patience: int = 12, val_frac: float = 0.18, device: str = "cpu",
        seed: int = 0) -> TrainResult:
    """Train in place; early-stop on a held-out val split. lr default 1e-3 for the
    small CPU models here (the proposal's 1e-4 is for the full A100 model)."""
    torch.manual_seed(seed)
    model.to(device)
    clf = model.cfg.task == "classification"

    n = len(X)
    Xt = _t(X, device)
    Yt = _t(Y, device, torch.long if clf else torch.float32)
    cvt = _cv_array(cv, n, device)
    sdt = _cv_array(cv_sd, n, device)
    dlt = None if delays is None else _t(delays, device)   # per-channel, batch-invariant

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(round(n * val_frac)))
    vi, ti = perm[:n_val], perm[n_val:]

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss() if clf else nn.MSELoss()

    def run(idx):
        c = cvt[idx] if cvt is not None else None
        s = sdt[idx] if sdt is not None else None
        return model(Xt[idx], c, s, dlt)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, since, history = float("inf"), 0, []
    g = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        model.train()
        for b in ti[torch.randperm(len(ti), generator=g)].split(batch_size):
            opt.zero_grad()
            loss_fn(run(b), Yt[b]).backward()
            opt.step()
            if hasattr(model, "constrain_"):      # e.g. EEGNet max-norm constraints
                model.constrain_()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(run(vi), Yt[vi]))
        history.append(val)
        if val < best_val - 1e-6:
            best_val, since = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(best_val=best_val, epochs_run=len(history), history=history)


def own_normalize(Xfit, *others):
    """Re-standardise per channel using Xfit's OWN statistics. Returns (Xfit, *others).

    Every "retrain / recalibrate on the target's own data" baseline in this repo needs
    this. Those baselines are handed inputs already z-scored by the SOURCE session's
    per-channel statistics — which is exactly the distribution shift the alignment methods
    exist to correct — and then judged as an upper bound. They are not an upper bound in
    that frame: on three Indy sessions the same fine-tune scores 0.024 source-normalised
    and 0.741 own-normalised. Applying this to already-source-normalised input is
    equivalent to standardising the raw input (a composition of per-channel affines), so
    call sites do not need the raw arrays.

    Targets are deliberately NOT touched: leaving them in the shared frame keeps R2
    comparable across every method in the same table.
    """
    Xfit = np.asarray(Xfit, np.float32)
    mu = Xfit.mean((0, 2), keepdims=True)
    sd = Xfit.std((0, 2), keepdims=True) + 1e-6
    out = [((Xfit - mu) / sd).astype(np.float32)]
    out += [((np.asarray(o, np.float32) - mu) / sd).astype(np.float32) for o in others]
    return out[0] if not others else tuple(out)


@torch.no_grad()
def predict(model: B2SSDecoder, X, cv=None, cv_sd=None, delays=None, device: str = "cpu") -> np.ndarray:
    """Regression -> (N, out); classification -> logits (N, n_classes)."""
    model.eval()
    n = len(X)
    dlt = None if delays is None else _t(delays, device)
    out = model(_t(X, device), _cv_array(cv, n, device), _cv_array(cv_sd, n, device), dlt)
    return out.cpu().numpy()


# -- SubjectData convenience wrappers (synthetic offline comparison) --------- #
def train_decoder(model: B2SSDecoder, sub: SubjectData, *, epochs: int = 40,
                  device: str = "cpu", seed: int = 0, **kw) -> TrainResult:
    return fit(model, sub.X_train, sub.Y_train, cv=sub.cv, epochs=epochs,
               device=device, seed=seed, **kw)


@torch.no_grad()
def decode_continuous(model: B2SSDecoder, eeg: np.ndarray, cv: float,
                      device: str = "cpu") -> np.ndarray:
    """Slide the window over a continuous segment -> (N_KIN, L) decode for latency."""
    from numpy.lib.stride_tricks import sliding_window_view
    xs = np.transpose(sliding_window_view(eeg, WIN, axis=1), (1, 0, 2))
    preds = predict(model, xs.astype(np.float32), cv, device=device)
    L = eeg.shape[1]
    out = np.empty((preds.shape[1], L), dtype=np.float32)
    out[:, WIN - 1:] = preds.T
    out[:, :WIN - 1] = preds[0][:, None]
    return out


def _selfcheck_own_normalize() -> None:
    """own_normalize must (a) standardise the fit array per channel, (b) apply the SAME
    transform to the others, and (c) be idempotent w.r.t. a prior per-channel affine --
    that last property is why call sites can pass already-source-normalised input."""
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((200, 5, 20)) * np.array([0.01, 1, 5, 0.5, 2])[None, :, None]
         + np.arange(5)[None, :, None]).astype("float32")
    Xte = (rng.standard_normal((50, 5, 20)) * np.array([0.01, 1, 5, 0.5, 2])[None, :, None]
           + np.arange(5)[None, :, None]).astype("float32")
    a, b = own_normalize(X, Xte)
    assert np.allclose(a.mean((0, 2)), 0, atol=1e-5) and np.allclose(a.std((0, 2)), 1, atol=1e-3)
    assert abs(float(b.mean())) < 0.5                      # same transform, not refit
    # Idempotent through any prior per-channel affine (that prior affine is the source
    # z-scoring). Exact in real arithmetic; the tolerance is the +1e-6 scale epsilon, which
    # matters only for the deliberately near-silent channel 0 (sd 0.01) — the same channel
    # that makes an unregularised standardiser explode.
    src = ((X - X.mean((0, 2), keepdims=True) * 0.3) / 7.0).astype("float32")
    src_te = ((Xte - X.mean((0, 2), keepdims=True) * 0.3) / 7.0).astype("float32")
    c, d = own_normalize(src, src_te)
    assert np.allclose(a[:, 1:], c[:, 1:], atol=1e-4) and np.allclose(b[:, 1:], d[:, 1:], atol=1e-4)
    assert np.allclose(a, c, atol=5e-3) and np.allclose(b, d, atol=5e-3)
    print("train.py own_normalize OK: standardises, shares the transform, affine-idempotent")


def _selfcheck() -> None:
    from .data import make_subject
    from .model import DecoderConfig
    from .eval import mse
    sub = make_subject(cv=55.0, n_train=200, n_test=100, seed=7)
    m = B2SSDecoder(DecoderConfig(gate_mode="cv"))
    res = train_decoder(m, sub, epochs=15, seed=7)
    assert np.isfinite(res.best_val)
    assert res.history[-1] <= res.history[0] + 1e-6, "val loss should not blow up"
    assert np.isfinite(mse(predict(m, sub.X_test, sub.cv), sub.Y_test))

    # classification path
    Xc = np.random.default_rng(0).standard_normal((60, sub.X_train.shape[1], WIN)).astype("float32")
    yc = np.random.default_rng(1).integers(0, 2, 60)
    clf = B2SSDecoder(DecoderConfig(task="classification", n_classes=2, gate_mode="none"))
    rc = fit(clf, Xc, yc, epochs=5, seed=0)
    assert predict(clf, Xc).shape == (60, 2) and np.isfinite(rc.best_val)
    print(f"train.py self-check OK: val {res.history[0]:.3f}->{res.best_val:.3f} "
          f"in {res.epochs_run} ep; clf path OK")


if __name__ == "__main__":
    _selfcheck()
