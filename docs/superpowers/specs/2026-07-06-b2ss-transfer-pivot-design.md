# Design Spec — B2SS Transfer-Normalization Pivot

_Date: 2026-07-06. Status: approved (brainstorming). Next: implementation plan._
_Project-facing narrative: [`PIVOT.md`](../../../PIVOT.md)._

## Context

Multi-seed experiments (RESULTS.md §1–§7) established that within-subject
conduction velocity does not improve decoding — a data-trained decoder learns the
conduction delays — but that measured conduction **does** improve *cross-subject
zero-shot transfer* (the decoder cannot learn a never-seen subject's delays). This
spec pivots B2SS from a CV-modulated decoder into a **conduction-aware transfer
normaliser** that cuts the per-subject/per-session calibration burden. North star:
practical transfer. Alignment source: the full spectrum (zero-shot measured CV /
few-shot labeled / unsupervised). Benchmarks: staged intracortical → EEG.

## Architecture

Invert the design: a decoder is trained once on a **source pool** and **frozen**;
only a small **`TransferNormalizer`** adapts per target.

```
target ─▶ TransferNormalizer ─▶ frozen decoder ─▶ output
             ├─ SpatialAligner (optional, borrowed; EEG only)
             └─ ConductionDelayAligner  δ ∈ R^K   (novel core)
                   δ from: measured CV | few-shot fit | unsupervised fit
```

### Components (each independently testable)

1. **`ConductionDelayAligner`** (`b2ss/transfer.py`) — applies per-tract-group
   temporal delays `δ ∈ R^K` (K≈8–16) to the input, generalising
   `b2ss.model.ChannelDelay` from per-channel to grouped/low-dim. Interface:
   `forward(x, delta) -> x_aligned`. Dependency: `ChannelDelay` shift primitive.
2. **`DelayFitter`** (`b2ss/transfer.py`) — produces `δ` for a target in one of
   three modes. Interface: `fit(target_data, mode, decoder=None, measured_cv=None,
   labels=None) -> delta`. Depends on a frozen decoder (few-shot/unsup) and
   `train.fit`-style loops.
3. **`TransferNormalizer`** (`b2ss/transfer.py`) — orchestrates optional spatial
   alignment + conduction alignment around a frozen backbone. Interface:
   `adapt(target_data, mode, ...) -> normalizer_state`; `predict(x) -> yhat`.
4. **`SpatialAligner`** (optional; EEG stage) — off-the-shelf Euclidean/Riemannian
   alignment. Clearly labeled borrowed. Interface: `fit(target)`, `transform(x)`.
5. **Backbone** — existing GRU / B2SS transformer, trained once, frozen. Behind a
   `predict(x)` interface; swappable without touching the normaliser.

### Reuse (do not reinvent)
`b2ss.model.ChannelDelay` (shift primitive), `b2ss.train.fit/predict`, GRU/transformer
backbones, `b2ss.stats.mean_ci`, `b2ss.intracortical.load_maze/make_windows/inject_group_latency`,
`b2ss.baselines`.

## Data flow — three deploy modes

Source pool → train decoder → freeze. Target arrives → select mode → `DelayFitter`
produces `δ` → `TransferNormalizer` normalises → frozen decoder predicts.

- **Zero-shot:** `δ = g(measured_cv)`; no target neural data. (Validated: RESULTS §7.)
- **Few-shot:** fit `δ` (K params) minimising decoding loss on a few labeled target
  trials, decoder frozen.
- **Unsupervised:** fit `δ` by self-supervised objective (match target encoder-latent
  statistics to source distribution / entropy-minimisation), no labels.

Primary output: an **accuracy-vs-target-information** curve (zero → unlabeled → few
labeled) against (a) no-normalisation and (b) full per-target retraining.

## Benchmarks

- **Stage 1 (intracortical):** controlled injected-delay transfer (done) + a real
  multi-session reach set (confirm availability). Metrics: transfer R² vs target-info;
  trials-to-target-accuracy. Baselines: raw frozen, full retraining, learned-delay
  (no conduction structure), a standard TTA method.
- **Stage 2 (EEG/MOABB):** cross-session & cross-subject; `TransferNormalizer` =
  spatial align + conduction module; **ablate the conduction module's marginal delta**
  on top of spatial alignment. Baselines: MOABB standards, spatial-align-only.

Pre-declared success: Stage 1 conduction-norm beats no-norm zero/few-shot and nears
full-retraining with less data; Stage 2 significant marginal conduction delta OR an
honest "intracortical-only" bound.

## Error handling & guardrails

- Unsupervised `δ`-fit instability → low-dim `δ` regularises; fall back to zero/few-shot.
- Scarce real multi-session intracortical → lean on controlled proof; label real vs
  controlled explicitly.
- Injected ≠ real conduction differences → always labelled; marginal-delta framing.
- New dep (MOABB) staged, subsettable.

## Testing

Unit: normaliser forward/shapes; `δ`-fit recovers a known injected `δ` (oracle);
few-shot fit reduces loss; unsupervised objective moves `δ` toward truth on controlled
data; **frozen-decoder invariance** (backbone params unchanged during adaptation).
E2E: run the transfer scripts; read accuracy-vs-target-info curves; ≥3–5 seeds with
CIs; JSON + error-bar figures in `results/`.

## Deliverables

`b2ss/transfer.py` (+ tests); `scripts/run_transfer_modes.py` (spectrum curve),
`scripts/run_moabb_transfer.py` (EEG stage); `PIVOT.md`; updated README/BRIEF/ROADMAP/
PAPER_OUTLINE; `results/transfer_modes.*`, `results/moabb_transfer.*`.

## Phased plan (for writing-plans)

1. `ConductionDelayAligner` + `DelayFitter` (3 modes) + `TransferNormalizer` + tests.
2. Intracortical spectrum benchmark (controlled) — `run_transfer_modes.py`, curves.
3. Real multi-session intracortical (confirm dataset; loader; run).
4. EEG/MOABB stage — spatial-align + conduction ablation.
5. Docs + honest reporting of whatever the numbers say.

## Out of scope
Solving non-temporal transfer axes (borrow spatial alignment); wet-lab acquisition of
paired neural+CV data (justified by BACKGROUND §9 but not built here).
