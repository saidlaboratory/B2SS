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
- None of this substitutes for the wet-lab study (MRI g-ratio, TMS-EEG, sEEG,
  closed-loop prosthetic control) — the only setting with a measured CV to test.
