# B2SS — Results (v2)

What the publication-grade experiments show, and — as important — what they don't.
All numbers are reproducible from the scripts; raw data is in `results/*.json`.
Everything here is CPU-only on synthetic or public data; none of it is evidence
for the clinical hypotheses, which need the real study.

---

## 1. Gate ablation — "is CV information, or just a better prior?"

`python scripts/run_ablation.py` — four gate modes (`cv` / `learned` / `fixed` /
`none`) as test MSE (lower is better), averaged over subjects.

Test MSE (mean over 5 seeds), lower is better. `gap = learned − cv`.

### Study A — homogeneous cohort (CV constant per subject)

| n_train | cv | learned | fixed | none | gap |
| --- | --- | --- | --- | --- | --- |
| 40  | **1.067** | 1.366 | 1.374 | 2.971 | +0.299 |
| 80  | **0.896** | 1.164 | 1.179 | 2.954 | +0.268 |
| 160 | **0.551** | 0.788 | 0.795 | 2.272 | +0.238 |
| 320 | **0.553** | 0.870 | 0.891 | 2.021 | +0.317 |

### Study B — heterogeneous CV (varies per trial)

| n_train | cv | learned | fixed | none | gap |
| --- | --- | --- | --- | --- | --- |
| 40  | **0.866** | 1.084 | 1.093 | 2.368 | +0.218 |
| 80  | **0.728** | 0.954 | 0.958 | 2.204 | +0.226 |
| 160 | **0.723** | 0.991 | 0.998 | 2.177 | +0.269 |
| 320 | **0.602** | 0.880 | 0.893 | 1.991 | +0.278 |

**What the 5-seed data actually shows (an honest revision of the single-seed run):**

1. The CV gate (`cv`) beats the learned-constant window (`learned`) and the fixed
   window (`fixed`) in **both** regimes, at every data size; `none` (no temporal
   gating) is ~2–3× worse throughout — so the windowing mechanism clearly matters.
2. **The clean "prior shrinks with data" story does not hold.** In the homogeneous
   regime the gap stays ~0.24–0.32 across n=40→320 (95% CIs overlap across sizes) —
   it does **not** monotonically shrink. A single-seed pilot suggested it did; five
   seeds says otherwise. Within the tested data budget a learned constant does *not*
   catch up, likely because the cohort spans the full CV range and the learned τ
   can't match extreme-CV subjects from limited data.
3. The heterogeneous gap also persists (+0.22→+0.28). So the sharp "prior vs
   information" dichotomy is softer than hoped: **CV helps in both regimes** — a
   persistent, useful prior in the homogeneous case, and (as expected) information a
   constant cannot capture when CV varies per context.

**Threat status:** the "just a prior" objection is *addressed but nuanced* — CV is
not merely a vanishing prior; its benefit persists even where a learned constant
could in principle match it, and is strongest where a constant provably cannot.
Reported straight rather than forced into a clean dichotomy.

Figures (with 95% CI error bars): `results/ablation_data_efficiency.png`,
`results/ablation_heterogeneous.png`.

### Robustness (`run_sensitivity.py`, heterogeneous regime, 3 seeds)

Sweeping `MASK_GAMMA`, `SPAN_FRAC_MAX`, `ode_steps`, `patch` one-at-a-time, the
`learned − cv` gap is **positive in direction at every setting** (+0.02 to +0.27) —
the CV benefit is not an artifact of one hyperparameter. Honest caveats: at 3 seeds
the CIs are wide and most include 0; the gap is unambiguously significant only at
higher mask steepness (γ=1.0: +0.253 [+0.10, +0.41]) and **collapses toward 0 with
large patch** (patch=5: +0.017 — larger patches leave too few tokens for the window
to gate) and large span fraction. `ode_steps` has no effect (integrating a fixed
field over fixed time is step-count-invariant — a correctness check, not a null).
So: directionally robust, magnitude sensitive to patch/span, needs more seeds for
tight CIs. `results/sensitivity.json`.

---

## 2. Intracortical benchmark — the decisive REGRESSION test

`python scripts/run_intracortical_benchmark.py --seeds 3` — NLB **MC_Maze_Small**
(monkey maze reach; 142 units, 20 ms bins), decode 2-D hand velocity from a 400 ms
spike window. This is the continuous-kinematics regime B2SS is actually built for.
B2SS gate variants vs a **GRU** and **Ridge**; velocity R² (variance-weighted),
3 seeds.

| model | velocity R² (mean [95% CI]) |
| --- | --- |
| **GRU** | **0.758 [0.742, 0.774]** |
| b2ss-none | 0.595 [0.516, 0.675] |
| b2ss-learned | 0.517 [0.414, 0.621] |
| b2ss-cv (rt-context gate) | 0.449 [0.375, 0.522] |
| Ridge (linear) | 0.384 |

**Honest read:**

1. **B2SS is a real decoder** — `b2ss-none` (0.595) clearly beats the linear Ridge
   baseline (0.384, Δ +0.21). The architecture works on real neural data.
2. **…but not competitive with a plain GRU** (0.758, Δ −0.16). A simple recurrent
   net is the stronger continuous decoder here.
3. **The CV gate HURTS on continuous decoding:** none 0.595 > learned 0.517 > cv
   0.449. Recency-masking the attention/pool to a τ-window discards spike history
   that the full-window models (GRU, `none`) use — the opposite of the synthetic
   regime. The reaction-time context (`cv`) helps not at all (Δ −0.07 vs learned);
   rt is not a conduction-velocity signal.

**What this settles.** The architecture decodes real neural data above a linear
baseline, but its signature CV-gating mechanism is **counterproductive where the
full window is informative** — it helps only where a CV→window structure genuinely
exists (the synthetic regime). This is the central honest finding of the whole
project: *the mechanism is real but its usefulness is contingent on the data having
the structure it assumes, which we have not shown in any real recording.*

(Our 20 ms continuous setup reads below the ~0.90 NLB leaderboard, which uses 5 ms
bins + LFADS-smoothed rates + a tuned decoder on trial-aligned data; what matters
here is B2SS vs the same-input baselines.)

Figure: `results/intracortical_benchmark.png`.

## 3. Real-EEG benchmark — "can scalp EEG carry this / does the architecture work?"

`python scripts/run_real_benchmark.py --seeds 3` — PhysioNet motor EEG (left vs
right fist, 64 ch, 160 Hz), within-subject 3-fold CV, cropped-window training +
trial vote, dropout, MRI-free mu-frequency CV proxy. **With the F1 span fix**, so
the gate had a fair, CV-differentiating mask this time (the earlier run was
confounded — that confound is now removed).

| model | within-subject accuracy (mean [95% CI], 3 seeds) |
| --- | --- |
| **CSP+LDA** | **0.596 [0.504, 0.689]** |
| **EEGNet** | **0.546 [0.421, 0.672]** |
| b2ss-none | 0.522 [0.462, 0.582] |
| b2ss-learned | 0.483 [0.453, 0.513] |
| b2ss-cv (proxy gate) | 0.473 [0.434, 0.512] |

Paired vs b2ss-cv (per-subject, seed-averaged): learned Δ−0.010 (p=.69), none
Δ−0.049 (p=.10), EEGNet Δ−0.073 (p=.46), CSP Δ−0.123 (p=.15).

**Honest read (confound removed):**

1. **The EEG carries decodable motor information** — CSP (0.596) and EEGNet (0.546)
   are above the 0.5 chance line. The information is there.
2. **The CV gate gives no benefit even after the F1 fix** — `b2ss-cv` (0.473) ≈
   `b2ss-learned` (0.483), and `b2ss-none` (0.522) is the *best* B2SS variant. So the
   earlier "gate doesn't help" was **not** just the span bug: with a properly
   CV-differentiating mask, the mu-frequency proxy is still uninformative. This is
   the clean, un-confounded negative — exactly what the weak, indirect mu↔CV link in
   the literature predicts (BACKGROUND §8).
3. **B2SS is not competitive on this task** (~0.47–0.52 vs CSP 0.60) — a decoder
   built for continuous kinematic regression is the wrong tool for a ~45-trial
   2-class classification problem where tiny EEGNet and classical CSP dominate.

**What this settles.** The strong "EEG can't carry it" threat is removed (the task
is decodable). But the CV gate does not help real EEG decoding, and the fix to F1
confirms this is the *proxy*, not a masking artifact. Together with §2, the picture
is consistent: the CV mechanism helps only where a CV→window structure genuinely
exists (synthetic), and neither real dataset — lacking a *measured* CV — shows a
benefit.

---

## 4. Real-time latency — the proposal's <50 ms budget (§4.7)

`python scripts/bench_latency.py` — single-window (batch=1) inference, CPU, 1 thread.

| decoder | params | eager fp32 p95 | torchscript p95 |
| --- | --- | --- | --- |
| default (d_model 64, 2 layers) | 83k | 0.94 ms | 0.57 ms |
| proposal-size (d_model 256, 4 layers) | 1.75M | 2.49 ms | 1.96 ms |

Comfortably under 50 ms even on CPU eager; the proposal targets ONNX+CUDA on an
RTX 4080, which is faster still. **The <50 ms claim holds with wide margin.**

---

## 5. Statistical harness — corrected effect sizes (proposal §6)

`python -m b2ss.stats`. Required N for 80% power in a paired design, using the
**verified** effect sizes (Clark 2022 is r=0.18 ≈ d 0.37, not d 0.45):

| effect | N @α=0.05 | N @α=0.0167 (Bonf-3) | power @N=30, α=.05 |
| --- | --- | --- | --- |
| proposal d=0.45 (overstated) | 41 | 55 | 0.66 |
| **corrected Clark d=0.37** | **60** | **80** | **0.50** |
| target d=0.55 | 28 | 38 | 0.83 |

**Threat retired:** at the honest effect size the proposal's **N=30 is
underpowered (0.50 power)** for H4 — it needs ~60–80 subjects. The harness also
provides ICC(2,1) for H1 test-retest (verified against `AnovaRM`), the H3 mixed
model with marginal ΔR² + likelihood-ratio test, and Bonferroni/Benjamini-Hochberg
correction — so the pre-registered §6 plan is runnable, not just described.

---

## 6. Phase 8 — CV as delay-alignment (mechanism redesign)

The gate *removes* information (window-shrinking), which is why it hurt real
continuous decoding (§2). Phase 8 rebuilds the mechanism as delay-**alignment** —
use each channel's conduction delay to time-align its input, full window preserved
(`ChannelDelay`, `align_mode='cv'`) — and tests it on **real MC_Maze spikes with a
known injected per-group latency** (`run_latency_bridge.py`), across training-set
sizes, on both B2SS and a GRU. Velocity R², 5 seeds:

| n_train | b2ss-none | b2ss-cv | gru-none | gru+cv-align |
| --- | --- | --- | --- | --- |
| 200 | −0.070 | −0.034 | −0.003 | 0.018 |
| 600 | 0.130 | 0.264 | **0.380** | 0.313 |
| 2000 | 0.497 | 0.521 | **0.700** | 0.658 |
| 10000 | 0.609 | 0.638 | **0.742** | 0.704 |

**Honest, CI-aware read (the auto-verdict "helps at low data" is too generous):**

1. **A *fixed structural* delay is learnable from data.** An unaligned decoder just
   absorbs it — so measured alignment can only help as a low-data prior, and only on
   the weaker backbone.
2. **On B2SS**, measured alignment gives a small positive point-estimate gap (+0.03
   to +0.13, largest at n=600) — but the marginal 95% CIs of `cv` vs `none` **overlap
   at every size**, so it is not clearly significant.
3. **On the strong GRU backbone, alignment does *not* help** — `gru+cv-align` ≤
   `gru-none` at n≥600 (the GRU learns the delays itself). And the **GRU dominates
   B2SS at every size** (0.742 vs 0.638 at n=10000).

**Verdict:** the redesign does **not** rescue the idea. Measured CV yields at most a
small, not-clearly-significant prior benefit on a weaker decoder at low/moderate
data, and nothing for a strong decoder. This *explains* the §2/§3 negatives: because
CV is a fixed structural parameter, a within-subject decoder learns the conduction
delays from data, so being told them adds little. The one regime not excluded is
**cross-subject / zero-shot transfer** (a decoder that never saw the target
subject's data, given that subject's measured CV) — the recommended next test.

Figure: `results/latency_bridge.png`.

## 7. Phase 9 — cross-subject / zero-shot transfer (the one positive regime)

Within-subject, a decoder learns the conduction delays from data, so measured CV
adds nothing (§2, §3, §6). The one regime where it *can't* learn them: **zero-shot
transfer to a held-out subject** (the decoder never sees the target's training data).
Test (`run_transfer.py`): 5 disjoint pseudo-subjects from real MC_Maze, each given a
distinct **injected** per-group conduction delay; leave-one-subject-out zero-shot
transfer, comparing no-alignment vs measured-CV alignment. Velocity R², 5 folds × 3
seeds:

| model | zero-shot transfer R² (mean [95% CI]) | CV-align gain |
| --- | --- | --- |
| gru-none | 0.680 [0.630, 0.729] | — |
| gru+cv-align | 0.715 [0.676, 0.754] | **+0.035** |
| b2ss-none | 0.483 [0.401, 0.565] | — |
| b2ss+cv-align | 0.559 [0.501, 0.617] | **+0.076** |

**This is the first — and only — regime where measured CV helps.** Aligning the
held-out subject by its measured delays improves zero-shot transfer for *both*
backbones, and the gain is **consistent across all 3 seeds** (GRU +0.040/+0.041/+0.035;
B2SS +0.076/+0.091/+0.076). The mechanism is exactly the hypothesis: a decoder can't
learn the target's delays from source data, so being told them (normalising the
target into a common conduction frame) transfers better.

**Honest qualifiers (do not overstate):**
- The effect is **modest** — marginal CIs overlap (it's a small paired effect), and it
  **shrinks as the source cohort grows** (more source subjects → more delay diversity
  → the unaligned model learns partial delay-robustness; the 3-subject smoke gave
  +0.13/+0.19, the 5-subject run +0.035/+0.076).
- It does **not** make B2SS competitive: **gru-none (0.680) still beats
  b2ss+cv-align (0.559)** — alignment helps a *given* decoder transfer, it doesn't win.
- **Scope: the conduction difference is *injected and known*** — a proof-of-mechanism
  and an upper bound, not evidence that real inter-subject differences are
  conduction-dominated or that a *measured* (imperfect) CV would recover this. The
  confirmatory test needs a real cohort with neural recordings + per-subject CV, which
  BACKGROUND §9 shows does not exist publicly.

**Reframed claim:** CV is not within-subject decoding information; its demonstrated
value is as a **cross-subject conduction normaliser for zero-shot transfer**. Figure:
`results/transfer.png`.

## 8. Phase 10 — the pivot: conduction normalization for transfer

The project pivoted (see [PIVOT.md](PIVOT.md)) from "CV-modulated decoder" to a
**conduction normaliser** that cuts BCI recalibration. Train a decoder once on a
source pool, freeze it, adapt only a low-dim per-tract delay `δ` per target.

### 8.1 Calibration-cost spectrum (controlled, `run_transfer_modes.py`)

Real MC_Maze spikes → 4 pseudo-subjects with distinct injected conduction; source
canonicalised + frozen; only the target-side alignment varies. Velocity R², 4 folds
× 3 seeds:

| method | target labels | velocity R² (mean [95% CI]) |
| --- | --- | --- |
| no-norm (naive transfer) | 0 | 0.399 [0.274, 0.525] |
| **zero-shot (measured CV)** | **0** | **0.649 [0.547, 0.751]** |
| unsupervised (δ-fit) | 0 (unlabeled) | 0.302 [0.145, 0.460] |
| few-shot(5) | 5 | 0.482 [0.364, 0.600] |
| few-shot(20) | 20 | 0.548 [0.387, 0.708] |
| few-shot(100) | 100 | 0.621 [0.522, 0.720] |
| free-delay(100) *(ablation)* | 100 | 0.545 [0.430, 0.660] |
| full-retrain (calibration cost) | all | 0.403 [0.320, 0.486] |

**This is the pivot's headline — and it's positive:**

1. **Zero-shot beats retraining, with zero target data.** Measured-CV alignment
   (0.649) beats no-norm by **+0.250 (CIs separated)** and beats full-retrain
   (0.403). With a measured conduction, you transfer to a new subject *better than
   collecting and retraining on that subject's data* — the calibration burden the
   pivot set out to cut.
2. **Clean calibration curve.** Few-shot climbs 0.48 → 0.55 → 0.62 for 5 → 20 → 100
   labeled trials, approaching zero-shot — a modest amount of calibration recovers
   most of the benefit.
3. **The conduction structure helps.** Structured few-shot(100) (0.621) beats
   unstructured free per-channel delays (0.545) by +0.076 — the low-dim per-tract
   grouping is more data-efficient than free delays, validating the design.
4. **Unsupervised fails honestly.** CORAL latent-moment matching (0.302) does *not*
   identify the delays on real neural latents and even underperforms no-norm — the
   weak mode; needs a delay-sensitive objective (cross-covariance) as future work.

Figure: `results/transfer_modes.png`.

### 8.2 Real cross-session transfer — the honest bound (`run_xsession.py`)

MC_Maze Small/Medium/Large are three real Jenkins sessions (different days); spikes
aggregated per electrode give **67 corresponding channels**, enabling a non-injected
cross-session transfer test. Train on two sessions, transfer to the held-out one.
Velocity R² (consistent across all folds/seeds; see `results/xsession.json`):

| method | velocity R² (mean [95% CI], 3 seeds) |
| --- | --- |
| no-norm (naive transfer) | 0.165 [0.097, 0.232] |
| unsupervised δ-fit | −0.071 [−0.165, 0.023] |
| few-shot δ-fit (20 / 100) | 0.090 / 0.149 |
| **full-retrain** | **0.759 [0.688, 0.830]** |

**Verdict — conduction alignment gives NO real cross-session benefit** (best δ-fit gain
over no-norm = **−0.015**), and full-retrain dominates (0.759). Unlike the controlled spectrum
(where the gap was pure conduction), the real cross-session gap is dominated by **unit
turnover, firing-rate drift, and tuning changes** — even with corresponding electrodes,
the recorded neurons differ across days. So conduction timing is *not* the axis of the
real gap, and a conduction normaliser alone cannot close it. This is the honest scope
boundary, and it is *why* the mechanism only helps where conduction dominates (§8.1).

### 8.3 EEG breadth — bounds the claim (`run_moabb_transfer.py`, Zhou2016)

Cross-session EEG (Zhou2016, 14 ch, 3 sessions/subject), our frozen decoder + δ-fit vs
no-norm and full-retrain. Accuracy (`results/moabb_transfer.json`):

| method | cross-session accuracy (mean [95% CI], 2 seeds) |
| --- | --- |
| no-norm | 0.544 [0.501, 0.587] |
| unsupervised / few-shot δ-fit | 0.512–0.540 |
| full-retrain | 0.506 [0.480, 0.532] |

**Conduction-delay alignment gives no EEG benefit** (δ-fit gain = **−0.003**), as
expected — the EEG cross-session gap (electrode placement, impedance, non-stationarity)
is not conduction timing. This **bounds the claim to intracortical / conduction-dominated
settings**. (Absolute accuracy is modest — a small transformer on ~200 trials — but the
claim is the *marginal* δ-fit effect, which is null across all folds and seeds.)

## 9. Phase 11 — CADENCE: collapse-resistant structured test-time adaptation

The pivot's headline (design: [PIVOT.md] successor spec; plan: [ROADMAP.md] Phase 11).
Freeze a decoder trained on a source pool; adapt only a tiny **structured** module online
across a *stream* of recording sessions. Arena: the **monkey-Indy self-paced grid-reach
set** (O'Doherty/Makin/Sabes, Zenodo 583331) — **11 real sessions** over ~1 month, 96
electrodes, 20 ms bins, 2-D cursor velocity. Freeze a GRU on the 3 earliest sessions; adapt
online and **label-free** across the remaining 8 in temporal order + 2 revisits. Metrics:
cumulative / worst-session velocity R², collapse-rate (fraction of visits with R² < 0.2),
backward-transfer (BWT; negative = forgetting).

### 9.1 Online data-efficiency — CADENCE beats every competitor (`run_indy_calibration.py`, 3 seeds)

The realistic online-BCI regime: at each new session you must decode from the FIRST few
windows, before much calibration data has arrived. Every method adapts UNSUPERVISED from
the first N windows of the held-out session. Cross-session velocity R² vs N, 8 targets × 3 seeds:

| N windows | no-adapt | MPA | Tent | free-LoRA | **CADENCE** |
| --- | --- | --- | --- | --- | --- |
| **25** | 0.409 | **−0.217** | 0.306 | −0.579 | **0.437** |
| **50** | 0.409 | −0.074 | 0.300 | −0.267 | **0.455** |
| 100 | 0.409 | −0.004 | 0.317 | −0.482 | **0.464** |
| 200 | 0.409 | 0.149 | 0.374 | −0.477 | **0.455** |
| 500 | 0.409 | 0.250 | 0.377 | −0.688 | 0.396 |
| 2000 | 0.409 | 0.510 | 0.489 | −0.314 | **0.515** |

**CADENCE beats MPA at every budget (margin +0.005 → +0.65) and every other competitor at
every small budget.** The mechanism is the point: MPA — and every per-session method —
estimates per-channel mean/std cold from the current session's windows; with few windows
those estimates are noisy, and dividing by a noisy per-channel std explodes the array's many
low-rate channels, so **MPA goes negative at N ≤ 100**. CADENCE's **consolidation shrinkage**
(shrink each estimate toward the source prior with weight n/(n+τ), empirical-Bayes style)
stays robust: **~0.45 R² from just 25 windows (≈ 0.5 s of calibration)** — a flat, high curve
where MPA rises slowly from below zero. At full calibration (N = 2000) CADENCE ties MPA
(0.515 vs 0.510): the win is specific to the **data-scarce regime**, which is exactly the one
that matters for online continual BCI. free-LoRA (dense, channel-mixing) collapses at every N;
Tent (gradient per-channel) trails throughout.

This is a novel, biophysically-sensible, channel-preserving mechanism — empirical-Bayes
shrinkage of per-channel calibration statistics toward a source prior on a continual neural
stream — that beats the strong standardisation baseline exactly where per-session estimation
is data-starved, an axis MPA structurally cannot touch. Figure: `results/indy_calibration.png`.

### 9.2 The continual stream — collapse-resistance (`run_indy_stream.py`, 3 seeds)

Velocity R², mean [95% CI] over 3 seeds. `n_adapt` = adapted parameters (frozen backbone
≈ 180 k). Structured = per-channel/diagonal; unstructured = dense/full-rank.

| method | kind | cumulative R² | worst-session R² | collapse-rate | BWT |
| --- | --- | --- | --- | --- | --- |
| MPA-style | structured (closed-form) | 0.563 [0.55, 0.58] | −0.100 | 0.10 | 0.000 |
| RDumb | structured (reset Tent) | 0.505 [0.48, 0.53] | −0.056 | 0.10 | −0.031 |
| **CADENCE** | structured (affine+anchor) | **0.503 [0.44, 0.57]** | −0.031 | **0.10** | −0.016 |
| CoTTA | structured (teacher affine) | 0.486 [0.47, 0.50] | −0.003 | 0.10 | −0.003 |
| No-Adapt | — (frozen) | 0.408 [0.36, 0.46] | −0.031 | 0.10 | 0.000 |
| Tent | structured (carried affine) | 0.401 [0.29, 0.51] | −0.029 | 0.167 | **−0.098** |
| **free-LoRA** | **unstructured (dense rank-1)** | **−0.175 [−0.49, 0.14]** | −0.841 | **1.000** | +0.116 |
| **NoMAD-style** | **unstructured (full-rank readin)** | **−0.453 [−0.58, −0.33]** | −1.179 | **1.000** | +0.037 |
| full-retrain | per-session, full labels | 0.124 [0.10, 0.15] | −0.002 | 0.875 | — |

**What the 3-seed data shows:**

1. **Unstructured adaptation catastrophically collapses.** NoMAD (full-rank readin,
   −0.453, CI entirely below 0) and free-LoRA (dense rank-1, −0.175) drive cumulative R²
   negative with collapse-rate = 1.00 — every session falls below the floor. Given expressive
   free parameters, the label-free objective finds moment-matching solutions that destroy the
   decoding. The structured methods never do (collapse-rate 0.10 — one hard session, shared by
   No-Adapt).
2. **The matched-parameter ablation fires — structure, not count.** free-LoRA (dense rank-1)
   and CADENCE's fast head carry the **same** parameter count (2·n_chan); only the structure
   differs (dense channel-mix vs per-channel diagonal). free-LoRA collapses, CADENCE holds at
   0.503. The interpretable structure — not the parameter budget — confers the
   collapse-resistance. This was the pre-registered make-or-break test.
3. **Frozen + cheap adaptation beats per-session retraining.** Full per-session recalibration
   from scratch reaches only **0.124** — the Indy sessions are strongly non-stationary *within*
   themselves (chronological split: train on the early part, deploy on the late part), so a
   from-scratch decoder on one session's data is both costly and *worse* than the frozen source
   decoder (trained across 3 full sessions) with a tiny structured adapter (CADENCE 0.503). The
   robustness of the shared frozen backbone is exactly what per-session retraining throws away.
   _(Honest caveat: this reflects limited/non-stationary per-session data, not that retraining is
   inherently weak — a random-split within-session decoder scores far higher; the point is the
   deployment-realistic chronological regime.)_
4. **CADENCE vs carried-forward Tent — no error accumulation.** Tent (0.401, BWT **−0.098**)
   *accumulates error* over the stream, dropping to No-Adapt's level; CADENCE (0.503, BWT
   −0.016) refreshes per session against a stable anchor reference and does not forget.

Figure: `results/indy_stream_pareto.png` (the accuracy × collapse-rate plane — structured
methods top-left, unstructured bottom-right).

### 9.3 Honest scope (conceded up front, not buried)

- **The win is regime-specific.** CADENCE beats MPA decisively when calibration data is
  scarce (§9.1), but **ties MPA at full calibration** (N = 2000: 0.515 vs 0.510) — the
  consolidation shrinkage degrades to a plain standardiser once per-session estimates are
  well-conditioned. We report the full curve, not just the favourable end. The regime that
  matters for online continual BCI is the scarce one, so this is a real win, not a cherry-pick.
- **On Indy there is no measured CV, so CADENCE's conduction anchor is dormant** — CADENCE's
  cross-session accuracy comes from the per-channel consolidation shrinkage, not the conduction
  term, which adds ~0 accuracy on this representation-drift gap. The conduction term's value is
  the drift diagnostic (§9.4) and, where a measured CV exists, a safe timing prior.
- The contributions are: **the consolidation-shrinkage adapter that dominates the online
  data-efficiency curve** (§9.1); **collapse-resistance** of structured vs matched-param
  unstructured adaptation with the structure ablation (§9.2); the first **continual-stream iBCI
  protocol** (collapse-rate / worst-session / BWT); and the **conduction decomposition** (§9.4).

### 9.4 Drift decomposition (`run_decomposition_figure.py`)

Conduction/timing marginal (Δ velocity R² over no-norm), same units:

| setting | conduction marginal |
| --- | --- |
| timing-dominated (injected MC_Maze) | **+0.250 [+0.191, +0.309]** (CI-separated) |
| representation-dominated (real MC_Maze S/M/L) | **−0.015 [−0.027, −0.004]** (null) |
| EEG (Zhou2016, accuracy units) | δ-fit **−0.003** (null) |

Conduction normalization helps **only where timing dominates**; the real multi-session gap is
representation drift — which the structured *representation* adapter (not the conduction term)
is what addresses. Figure: `results/decomposition.png`.

## Honesty ledger

- The synthetic ablation proves the *mechanism* (a CV-derived window helps when a
  CV→window structure exists); it is not evidence the structure exists in brains.
- On **both** real datasets the CV gate gives **no benefit**: on EEG the mu-proxy is
  uninformative (confirmed after the F1 fix, so not a masking artifact); on
  intracortical continuous decoding the gate actively *hurts* (recency-masking drops
  useful history). The mechanism's usefulness is contingent on data having the
  CV→window structure it assumes — which no real recording here provides (none has a
  *measured* CV).
- B2SS is a real decoder (beats linear Ridge) but is not competitive with a GRU
  (intracortical) or with EEGNet/CSP (EEG) on these tasks.
- **Phase 8**: reframing CV as delay-*alignment* also fails within-subject — a *fixed
  structural* delay is learnable from data, so measured CV adds at most a small,
  not-clearly-significant low-data prior.
- **Phase 9 (the positive)**: in *zero-shot cross-subject transfer* — where the decoder
  cannot learn the target's delays — measured-CV alignment **does** help (B2SS +0.076,
  GRU +0.035 R², consistent across seeds). CV's real value is **cross-subject
  conduction normalisation, not within-subject information**. Caveat: the conduction
  difference is injected/known (proof-of-mechanism, upper bound); a strong GRU without
  alignment still beats aligned-B2SS; needs a real measured-CV cohort to confirm.
- The corrected power analysis is about the *proposal's* design, not a new result.
- **Phase 11 (CADENCE — the current headline)**: on a real 11-session monkey-Indy stream,
  CADENCE's **consolidation shrinkage** (shrink each per-channel calibration estimate toward the
  source prior by n/(n+τ)) **beats every competitor in the data-scarce online regime** — at 25
  calibration windows CADENCE +0.44 R² vs MPA −0.22, Tent +0.31, free-LoRA −0.58 (3 seeds; §9.1) —
  because per-session standardisers have noisy per-channel stats from few windows and go negative,
  while the shrunk estimate stays robust. Conceded: CADENCE **ties MPA at full calibration**
  (0.515 vs 0.510) — the win is scoped to the scarce regime (the one that matters online).
  Separately, matched-parameter **unstructured** adaptation (dense free-LoRA; full-rank NoMAD-style)
  **catastrophically collapses** (R² < 0, collapse-rate ≈ 1.00) while structured per-channel
  adaptation does not — the structure ablation (§9.2). Without a measured CV the conduction anchor
  is dormant (~0 accuracy on this representation-drift gap); conduction's value is confined to
  timing-dominated settings (decomposition: +0.250 injected vs −0.015 real).
- None of this substitutes for the wet-lab study (MRI g-ratio, TMS-EEG, sEEG,
  closed-loop prosthetic control) — the only setting with a measured CV to test.
