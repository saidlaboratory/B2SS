# B2SS — Pivot: Conduction-Aware Transfer Normalization

_From "a CV-modulated decoder" (which the evidence killed) to "a conduction
normalizer that makes decoders transfer across subjects/sessions with little
calibration" (which the evidence supports)._

Design spec: [`docs/superpowers/specs/2026-07-06-b2ss-transfer-pivot-design.md`](docs/superpowers/specs/2026-07-06-b2ss-transfer-pivot-design.md).
Full results behind the pivot: [RESULTS.md](RESULTS.md); honest paper framing:
[PAPER_OUTLINE.md](PAPER_OUTLINE.md).

## 1. Why we're pivoting (what the runs proved)

Across synthetic, real EEG (PhysioNet), and real intracortical (MC_Maze) data, at
multi-seed rigor:

- **The original claim is false within-subject.** Conduction velocity does not
  improve a decoder trained on that subject's own data — because a data-trained
  decoder simply *learns* the conduction delays. The gate gave no benefit (EEG,
  even after the mask fix), hurt continuous decoding (intracortical), and the
  Phase-8 delay-alignment redesign added at most a small, not-significant low-data
  prior. See RESULTS §1–§6.
- **One regime works: cross-subject zero-shot transfer.** When a decoder is applied
  to a *new* subject it never trained on, it *cannot* learn that subject's delays —
  so aligning the target by its measured conduction improved zero-shot transfer
  (B2SS +0.076, GRU +0.035 R², consistent across seeds; RESULTS §7).

The lesson: **CV is not decoding information; it is a cross-subject conduction
normaliser.** The pivot builds B2SS around that, aimed at the real BCI pain point —
the per-subject/per-session **recalibration burden**.

## 2. The new thesis

> Neural decoders fail to transfer across subjects/sessions partly because of
> **individual conduction-timing differences**. A decoder trained on a source pool
> and **frozen**, wrapped by a small **conduction-delay normaliser** that aligns a
> new target into a common conduction frame, transfers with little or no
> calibration. The normaliser's delays come from a measured CV when available, or
> are **estimated from the target's own data** when not.

Novelty vs prior transfer/adaptation work: the adaptation space is **low-dimensional
and conduction-structured** (per-tract temporal delays), not a free network
fine-tune — making it data-efficient, interpretable, and biophysically grounded.

## 3. What B2SS becomes (architecture)

Invert the design: **train the decoder once on a source pool, freeze it, adapt only
a tiny normaliser per target.**

```
target neural data ──▶ [ TransferNormalizer ] ──▶ [ FROZEN decoder ] ──▶ output
                          │  spatial align (optional, borrowed — EEG only)
                          │  conduction-delay align  δ  (NOVEL, low-dim: K tracts)
                          └─ δ from: measured CV | few labeled trials | unlabeled data
```

- **Conduction-delay aligner** (novel core) — per-tract-group temporal shifts `δ`
  (K≈8–16 groups, not per-channel-free → data-efficient, regularised). Generalises
  the existing `b2ss.model.ChannelDelay`.
- **Delay-fitting engine** — produces `δ` per target via the three modes below.
- **Optional spatial aligner** — off-the-shelf Euclidean/Riemannian alignment,
  stacked *before* the delay aligner, only for EEG (where the gap is more than
  temporal). Explicitly borrowed, not our contribution.
- The backbone (GRU or transformer) is a black box behind a clean interface —
  swappable without touching the normaliser.

New module: `b2ss/transfer.py`. Reuses `ChannelDelay`, `train.fit/predict`, the
GRU/transformer backbones, `stats.mean_ci`, and the MC_Maze loader.

## 4. Three deploy modes (the "calibration cost" spectrum)

Source pool → train → **freeze**. A new target arrives → pick the mode by what's
available → fit the low-dim `δ` → normalise → frozen decoder predicts. All modes fit
the *same* `δ`-space, so we can plot one **accuracy-vs-target-information** curve.

| Mode | Target info needed | How `δ` is obtained |
| --- | --- | --- |
| **Zero-shot** | a measured CV (no neural data) | map measured/known conduction → `δ` (validated, RESULTS §7) |
| **Few-shot** | a handful of *labeled* calibration trials | fit `δ` by minimising decoding loss, decoder frozen (K≈8–16 params → few trials suffice) |
| **Unsupervised** | *unlabeled* target data | fit `δ` by a self-supervised objective (match target latent stats to source / entropy-min) |

Headline comparison: this curve vs (a) no-normalisation and (b) full per-target
retraining — i.e. *how much calibration does conduction-normalisation save?*

## 5. Benchmarks & how we prove it honestly

**Stage 1 — intracortical (faithful mechanism).**
- Controlled (done): injected conduction differences on held-out pseudo-subjects.
- Real: a genuine multi-session/-subject reach set (MC_RTT / Makin–Flint multi-day —
  availability to confirm) for a non-injected test.
- Baselines: raw frozen decoder, **full per-target retraining** (the cost we beat),
  learned-delay (no conduction structure), a standard test-time-adaptation method.
- Metric: transfer R² vs target-info; trials-to-target-accuracy (calibration saved).

**Stage 2 — EEG breadth (MOABB cross-session & cross-subject).**
- `TransferNormalizer` = borrowed spatial alignment + our conduction module;
  **ablate the conduction module's marginal delta on top of spatial alignment** —
  that isolated delta is the actual claim.
- Baselines: MOABB standards (Riemannian TS/MDM, EEGNet), spatial-align-only.

**Pre-declared success criteria:** Stage 1 — conduction-normalisation beats no-norm
in zero/few-shot and approaches full-retraining with far less data. Stage 2 — the
conduction module adds a *significant marginal* delta on top of spatial alignment,
**or we report plainly that it is intracortical-only.**

## 6. Honest scope & guardrails

- Our novelty is the **temporal/conduction axis** only; real transfer gaps also
  involve spatial/tuning/non-stationarity — we borrow spatial alignment and isolate
  our marginal contribution rather than claiming to solve all of transfer.
- Controlled (injected) vs real results are always labelled as such; injected
  conduction differences are a mechanism proof / upper bound, not proof that real
  inter-subject gaps are conduction-dominated.
- If the conduction module doesn't help EEG, that **bounds** the claim
  (intracortical-only), it doesn't sink it. No result will be tuned toward a win.

## 7b. Validation outcome (Phase 10 built + tested)

The pivot is implemented (`b2ss/transfer.py` + four benchmark scripts, 21 tests green)
and run. Full numbers in [RESULTS.md](RESULTS.md) §8.

- **✅ Works where conduction is the gap (controlled, real spikes).** On the
  calibration-cost spectrum, **zero-shot measured-CV alignment transfers to a new
  subject better than retraining from scratch** — 0.649 vs 0.403 R² full-retrain,
  +0.250 over naive transfer with *zero* target data (CIs separated). Few-shot recovers
  most of it with a handful of trials; the low-dim conduction structure beats free
  delays; unsupervised (moment-matching) honestly fails.
- **⚠️ Bounded on real multi-session data.** On real cross-session intracortical
  (MC_Maze S/M/L, per-electrode) and EEG (Zhou2016), conduction alignment gives **no**
  transfer benefit and full-retrain dominates — the real gap is unit turnover / tuning
  drift / non-conduction factors, not timing.

**Net:** the reframed thesis is *validated and scoped* — conduction normalisation
enables calibration-free transfer **where the cross-subject gap is conduction-dominated**
(proven on real spikes), and real multi-session gaps are not. The honest next step is
combining conduction normalisation with representation alignment, or targeting settings
where conduction dominates.

## 7. What we keep vs change

**Keep:** the `ChannelDelay` mechanism, the decoder backbones, the training/eval/stats
harness, the MC_Maze pipeline, the transfer experiment, and all the honest negative
results (they motivate the pivot).
**Change:** the headline claim (CV-modulated decoding → conduction normalisation for
transfer); the product (decoder → normaliser wrapping a frozen decoder); the
benchmarks (within-subject accuracy → cross-subject/session transfer & calibration
cost). See [ROADMAP.md](ROADMAP.md) Phase 10 for the to-do list.
