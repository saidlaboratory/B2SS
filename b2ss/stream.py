"""Online streaming harness for lifelong test-time adaptation.

Replays a list of sessions in a given visit order (revisits = repeated indices).
At each visit the adapter adapts on that session's train split (unlabeled unless a
mode passes labels) and is scored on its test split BEFORE the next visit — so the
recorded R² is genuinely online. The backbone stays frozen inside the adapter; the
harness never touches decoder weights. Every method (No-Adapt/Tent/CoTTA/CADENCE/...)
is a duck-typed Adapter with .adapt(X, Y=None) and .predict(X).
"""

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
