#!/usr/bin/env python3
"""Real-EEG benchmark on PhysioNet (left vs right fist), removing two threats:

  * "can scalp EEG even carry this / does the architecture work on real data?"
    -> compares the B2SS decoder against published baselines (EEGNet, CSP+LDA) on
    real 64-ch EEG, within-subject cross-validation. No planted structure.
  * "is CV information or just a prior?" -> on real data CV is a per-subject
    constant (the mu-frequency proxy), so the gate/learned/none ablation here
    tests the *practical* value of the proxy; the information case is shown on
    synthetic heterogeneous data (run_ablation.py Study B).

    python scripts/run_real_benchmark.py                 # ~10-20 min CPU
    python scripts/run_real_benchmark.py --quick         # smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from b2ss.datasets import (load_subject, default_cohort, zscore_train,
                           crop_windows, trial_vote)
from b2ss.proxies import mu_peak_frequency, frequencies_to_pseudo_cv
from b2ss.model import DecoderConfig, B2SSDecoder, count_params
from b2ss.baselines import EEGNet, csp_lda_accuracy
from b2ss.stats import mean_ci
from b2ss.train import fit, predict

RESULTS = Path(__file__).resolve().parent.parent / "results"
B2SS_MODES = ("cv", "learned", "none")     # gate ablation on real data
CROP_WIN, CROP_STRIDE = 240, 40            # 1.5 s windows, 0.25 s stride (~7/trial)


def eeg_cfg(n_chan, mode):
    return DecoderConfig(n_chan=n_chan, win=CROP_WIN, fs=160, patch=8,
                         d_model=64, nhead=4, num_layers=2, dropout=0.4,
                         task="classification", n_classes=2, gate_mode=mode)


def run_fold(Xtr, ytr, Xte, yte, pcv, n_chan, epochs, seed):
    """Deep nets: cropped-window training + trial-level vote. CSP+LDA: full trials."""
    Xtr, Xte = zscore_train(Xtr, Xte)
    Xtrw, ytrw, _ = crop_windows(Xtr, ytr, CROP_WIN, CROP_STRIDE)
    Xtew, _, tid = crop_windows(Xte, yte, CROP_WIN, CROP_STRIDE)
    acc = {}
    for mode in B2SS_MODES:
        m = B2SSDecoder(eeg_cfg(n_chan, mode))
        cv = pcv if mode == "cv" else None
        fit(m, Xtrw, ytrw, cv=cv, epochs=epochs, lr=1e-3, batch_size=64, seed=seed)
        pred = trial_vote(predict(m, Xtew, cv=cv), tid, len(yte))
        acc[f"b2ss-{mode}"] = float((pred == yte).mean())
    eeg = EEGNet(n_chan, CROP_WIN, 2)
    fit(eeg, Xtrw, ytrw, epochs=epochs, lr=1e-3, batch_size=64, seed=seed)
    acc["eegnet"] = float((trial_vote(predict(eeg, Xtew), tid, len(yte)) == yte).mean())
    acc["csp+lda"] = csp_lda_accuracy(Xtr, ytr, Xte, yte)
    return acc


def run_seed(cohort, data, pcv, models, folds, epochs, seed):
    """One seed: per-subject k-fold. Returns {model: {subject: accuracy}}."""
    acc = {m: {} for m in models}
    for s in cohort:
        se = data[s]
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        fold = {m: [] for m in models}
        for k, (tr, te) in enumerate(skf.split(se.X, se.y)):
            a = run_fold(se.X[tr], se.y[tr], se.X[te], se.y[te], pcv[s],
                         se.X.shape[1], epochs, seed * 100 + k)
            for m in models:
                fold[m].append(a[m])
        for m in models:
            acc[m][s] = float(np.mean(fold[m]))
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.subjects, args.folds, args.epochs, args.seeds = 3, 2, 40, 2
    RESULTS.mkdir(exist_ok=True)

    cohort = default_cohort(args.subjects)
    print(f"\nLoading {len(cohort)} PhysioNet subjects (cached after first run)...")
    data, peaks = {}, {}
    for s in cohort:
        se = load_subject(s)
        data[s] = se
        sm = se.X[:, se.sm_idx].transpose(1, 0, 2).reshape(len(se.sm_idx), -1)
        peaks[s] = mu_peak_frequency(sm, se.fs)
    pcv = dict(zip(cohort, frequencies_to_pseudo_cv([peaks[s] for s in cohort])))

    models = [f"b2ss-{m}" for m in B2SS_MODES] + ["eegnet", "csp+lda"]
    seeds = list(range(args.seeds))
    per_seed = []                                   # list of {model:{subj:acc}}
    for seed in seeds:
        acc = run_seed(cohort, data, pcv, models, args.folds, args.epochs, seed)
        per_seed.append(acc)
        print(f"seed {seed}: " + "  ".join(
            f"{m}={np.mean(list(acc[m].values())):.3f}" for m in models))

    # subject accuracy averaged over seeds (for paired tests across subjects)
    subj_acc = {m: [float(np.mean([ps[m][s] for ps in per_seed])) for s in cohort]
                for m in models}
    # model mean +/- 95% CI across seeds (of the per-seed cohort mean)
    seed_means = {m: [float(np.mean(list(ps[m].values()))) for ps in per_seed] for m in models}
    ci = {m: mean_ci(seed_means[m]) for m in models}

    print(f"\nSummary — within-subject accuracy, mean [95% CI] across {len(seeds)} seeds")
    for m in models:
        mn, lo, hi = ci[m]
        print(f"  {m:>10}: {mn:.3f} [{lo:.3f}, {hi:.3f}]")

    def paired(a, b):
        da, db = np.array(subj_acc[a]), np.array(subj_acc[b])
        try:
            _, p = stats.wilcoxon(da, db)
        except ValueError:
            p = float("nan")
        return float(np.mean(da - db)), float(p)

    print("\nPaired comparisons (b2ss-cv vs …, per-subject, seed-averaged)")
    comps = {}
    for ref in ("b2ss-learned", "b2ss-none", "eegnet", "csp+lda"):
        d, p = paired("b2ss-cv", ref)
        comps[ref] = {"mean_acc_diff": d, "wilcoxon_p": p}
        print(f"  vs {ref:>12}: Δacc={d:+.3f}  p={p:.3f}")

    plt.figure(figsize=(7, 4))
    xs = np.arange(len(models))
    means = [ci[m][0] for m in models]
    err = [[ci[m][0] - ci[m][1] for m in models], [ci[m][2] - ci[m][0] for m in models]]
    plt.bar(xs, means, yerr=err, capsize=4)
    plt.axhline(0.5, ls="--", c="gray", label="chance")
    plt.xticks(xs, models, rotation=20); plt.ylabel("within-subject accuracy")
    plt.title(f"PhysioNet L/R fist ({len(cohort)} subj, {args.folds}-fold, {len(seeds)} seeds)")
    plt.legend(); plt.tight_layout()
    fig = RESULTS / "real_benchmark.png"; plt.savefig(fig, dpi=120); plt.close()

    (RESULTS / "real_benchmark.json").write_text(json.dumps({
        "cohort": cohort, "folds": args.folds, "epochs": args.epochs, "seeds": len(seeds),
        "mu_peaks": {int(s): peaks[s] for s in cohort},
        "subject_acc_seedavg": {m: subj_acc[m] for m in models},
        "mean_ci": {m: ci[m] for m in models}, "paired_vs_b2ss_cv": comps,
        "b2ss_params": count_params(B2SSDecoder(eeg_cfg(64, "cv"))),
    }, indent=2))
    print(f"\nContext (published, honest splits): EEGNet ~0.67-0.77 & CSP/FBCSP ~0.68 "
          f"on BCI IV-2a 4-class; 2-class within-subject typically high-70s-80s.")
    print(f"figure: {fig.name}; data: results/real_benchmark.json\n")


if __name__ == "__main__":
    main()
