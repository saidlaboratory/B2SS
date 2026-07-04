"""Per-subject training loop (proposal 4.5): Adam, MSE + L2, early stopping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .data import SubjectData
from .model import B2SSDecoder


@dataclass
class TrainResult:
    best_val_mse: float
    epochs_run: int
    history: list[float]


def _to_tensor(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).float().to(device)


def train_decoder(model: B2SSDecoder, sub: SubjectData, *, epochs: int = 40,
                  lr: float = 1e-4, weight_decay: float = 1e-5, batch_size: int = 128,
                  patience: int = 10, val_frac: float = 0.18, device: str = "cpu",
                  seed: int = 0) -> TrainResult:
    """Train in place on sub.X_train/Y_train; early-stop on a held-out val split.

    val_frac of X_train (already the 85% non-test portion) ~ 0.15 of all data,
    matching the proposal's 70/15/15 split.
    """
    torch.manual_seed(seed)
    model.to(device)
    uses_cv = model.cfg.use_cv_gate
    cv = torch.tensor(sub.cv, device=device)

    X = _to_tensor(sub.X_train, device)
    Y = _to_tensor(sub.Y_train, device)
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(1, int(round(n * val_frac)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xval, Yval = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, since_best, history = float("inf"), 0, []
    g = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        for b in torch.randperm(Xtr.shape[0], generator=g).split(batch_size):
            opt.zero_grad()
            pred = model(Xtr[b], cv) if uses_cv else model(Xtr[b])
            loss_fn(pred, Ytr[b]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xval, cv) if uses_cv else model(Xval)
            val = float(loss_fn(vp, Yval))
        history.append(val)

        if val < best_val - 1e-6:
            best_val, since_best = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    return TrainResult(best_val_mse=best_val, epochs_run=len(history), history=history)


@torch.no_grad()
def predict(model: B2SSDecoder, X: np.ndarray, cv: float, device: str = "cpu") -> np.ndarray:
    model.eval()
    xt = _to_tensor(X, device)
    out = model(xt, torch.tensor(cv, device=device)) if model.cfg.use_cv_gate else model(xt)
    return out.cpu().numpy()


@torch.no_grad()
def decode_continuous(model: B2SSDecoder, eeg: np.ndarray, cv: float,
                      device: str = "cpu") -> np.ndarray:
    """Slide the window over a continuous segment -> (N_KIN, L) decode for latency.

    eeg: (C, L). Output aligned so column t is the decode of the window ending at t
    (first WIN-1 columns are NaN-free copies of the first valid decode).
    """
    from .data import WIN
    from numpy.lib.stride_tricks import sliding_window_view
    xs = np.transpose(sliding_window_view(eeg, WIN, axis=1), (1, 0, 2))  # (n, C, WIN)
    preds = predict(model, xs.astype(np.float32), cv, device)            # (n, N_KIN)
    L = eeg.shape[1]
    out = np.empty((preds.shape[1], L), dtype=np.float32)
    out[:, WIN - 1:] = preds.T
    out[:, :WIN - 1] = preds[0][:, None]
    return out


def _selfcheck() -> None:
    from .data import make_subject
    from .model import DecoderConfig
    from .eval import mse
    sub = make_subject(cv=55.0, n_train=200, n_test=100, seed=7)
    model = B2SSDecoder(DecoderConfig(use_cv_gate=True))
    res = train_decoder(model, sub, epochs=15, seed=7)
    assert np.isfinite(res.best_val_mse)
    assert res.history[-1] <= res.history[0] + 1e-6, "val loss should not blow up"
    test_mse = mse(predict(model, sub.X_test, sub.cv), sub.Y_test)
    assert np.isfinite(test_mse)
    print(f"train.py self-check OK: val_mse {res.history[0]:.3f}->{res.best_val_mse:.3f} "
          f"in {res.epochs_run} ep; test MSE {test_mse:.3f}")


if __name__ == "__main__":
    _selfcheck()
