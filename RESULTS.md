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

`python scripts/run_real_benchmark.py --subjects 8 --folds 3 --epochs 120` —
PhysioNet motor EEG (left vs right fist, 64 ch, 160 Hz), within-subject 3-fold CV,
cropped-window training + trial vote, dropout 0.4, MRI-free mu-frequency CV proxy.

| model | within-subject accuracy |
| --- | --- |
| b2ss-cv (proxy gate) | 0.481 ± 0.085 |
| b2ss-learned | 0.517 ± 0.075 |
| b2ss-none | 0.506 ± 0.076 |
| **EEGNet** | **0.606 ± 0.081** |
| **CSP+LDA** | **0.639 ± 0.132** |

Paired vs b2ss-cv (Wilcoxon): learned Δ−0.036 (p=.16), none Δ−0.025 (p=.81),
EEGNet Δ−0.125 (p=.055), CSP Δ−0.158 (p=.078). ~110k params for B2SS vs ~2.7k for EEGNet.

**Honest read — this is a partial, nuanced result, not a clean win:**

1. **The EEG carries decodable motor information.** CSP+LDA (0.639) and EEGNet
   (0.606) are clearly above the 0.5 chance line — so "scalp EEG can't carry
   motor intent" is false. The information is there.
2. **The B2SS architecture is *not* competitive here** (~0.48–0.52), trailing both
   baselines (marginally significant vs EEGNet/CSP). This is the expected reality
   of small-trial EEG: a 110k-param Transformer+Neural-ODE built for *continuous
   kinematic regression with a CV prior* is the wrong tool for a ~45-trial 2-class
   classification task, where the tiny EEGNet and classical CSP dominate. Dropout,
   cropped-window augmentation, and 120 epochs did not close the gap.
3. **The EEG-only CV proxy gives no benefit** — `b2ss-cv` (0.481) is if anything
   slightly *below* `learned`/`none`. This matches the literature: the mu-frequency
   ↔ conduction-velocity link is weak and indirect (BACKGROUND §8). The proxy is a
   convenience path, not a substitute for measured CV.

**What this does and doesn't settle.** It removes the strong form of the "EEG can't
carry it" threat (the task is decodable) and proves the pipeline runs end-to-end on
real data with no planted structure. It does **not** show the B2SS decoder beating
baselines — and honestly it shouldn't be expected to on this mismatched task. The
architecture's home regime is continuous kinematic decoding; the appropriate real-
data validation is a **continuous intracortical reach dataset** (e.g. Neural Latents
MC_Maze), which is the recommended next step. The CV *mechanism* itself is
demonstrated where it applies — the heterogeneous-CV ablation (Study B).

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

## Honesty ledger

- The synthetic ablation proves the *mechanism* (a CV-derived window helps when a
  CV→window structure exists); it is not evidence the structure exists in brains.
- The real-EEG benchmark tests the *architecture* on real data with a *weak,
  indirect* CV proxy; it does not validate CV-modulated decoding clinically.
- The corrected power analysis is about the *proposal's* design, not a new result.
- None of this substitutes for the wet-lab study (MRI g-ratio, TMS-EEG, sEEG,
  closed-loop prosthetic control).
