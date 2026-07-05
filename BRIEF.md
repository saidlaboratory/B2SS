# B2SS — Project Brief

*Everything, in one place: the idea, the science, the experiments, and the
software this repo actually builds.* Sourced from **B2SS Research Proposal
v1.1** (June 25, 2026). For the literature and citations see
[BACKGROUND.md](BACKGROUND.md); for build status see [ROADMAP.md](ROADMAP.md).

---

## 1. The one-sentence idea

Give a BCI decoder the user's **conduction velocity** — measured, not learned —
and let it set how long the decoder waits before committing to a movement:
fast-conducting brains get a short, low-latency window; slow-conducting brains
get a longer, more accurate one.

## 2. Why it might matter

Prosthetic control today is jerky and delayed. Dexterous movement — catching a
ball, not crushing an egg — depends on millisecond-scale conduction timing. Those
delays aren't noise: they're structured by each person's white-matter
microstructure. A spatial-only decoder literally cannot see this, because the
information isn't in *which* neuron fires but in *how fast* its signal
propagates. B2SS is the first decoder to treat CV as a usable prior.

## 3. Core concepts

- **Conduction velocity (CV):** speed of action potentials along myelinated
  axons. Set by the **g-ratio** = inner axon diameter / total fiber diameter
  (Rushton, 1951). CV rises with axon diameter and is maximized near g ≈ 0.6.
- **CV is structural, not trial-level.** It varies across *tracts*, across
  *individuals*, and over *learning* (hours–days) — but not moment-to-moment.
  So B2SS uses it as a **fixed prior per subject/tract**, never as a live signal.
  This is the proposal's key methodological guard against confounding CV with its
  own slow timescale.
- **CV → integration window.** A tract with faster CV can use a shorter temporal
  integration window (lower latency); a slower tract imposes a **structural
  latency floor** no amount of spatial decoding can beat.

## 4. Research question & hypotheses

> Can tract-specific CV — from MR g-ratio + TMS-EEG — used as a structural
> constraint on a decoder's temporal integration window, improve the speed,
> accuracy, and naturalness of prosthetic control vs. a matched-capacity
> spatial-only decoder?

| ID | Hypothesis | Bar |
| --- | --- | --- |
| H1 | CV is **measurable** & reliable | ≥15% CoV across N=30; test-retest ICC ≥ 0.80 |
| H2 | CV is **modifiable** by ccPAS | MEP latency ↓ ≥ 2 ms at 24 h vs sham |
| H3 | CV carries **information** for decoding | ΔR² ≥ 0.15 for optimal window ~ CV |
| H4 | B2SS **decodes better** | ≥15% lower MSE, ≥20% lower latency vs control |
| H5 | B2SS **feels better** | SUS + fluidity, Cohen's d ≥ 0.5 |
| H6 | B2SS **tracks plasticity** | maintains perf after ccPAS; control degrades ≥10% |

## 5. The five experiments (proposal)

1. **Corticospinal g-ratio atlas** (N=30) — first tract-specific CV reference map
   of the human motor system. Multi-shell dMRI + MT imaging, probabilistic CST
   tractography, voxel-wise g-ratio.
2. **CV pipeline validation** — TMS-EEG MEP latency vs dMRI g-ratio CV (N=30);
   both vs sEEG CCEP ground truth (N=5).
3. **Offline decoder training** — 200 center-out reach+grasp trials; B2SS vs
   control on held-out data. **← this repo simulates this offline stage.**
4. **Blinded closed-loop prosthetic control** — NeuroPix RPN-1 arm, B2SS vs
   control, participant + assessor blinded, counterbalanced.
5. **Plasticity challenge** — ccPAS to M1–PMd, re-measure CV at 24 h, test whether
   B2SS's online window re-adaptation preserves performance.

## 6. The CV estimation pipeline

Three estimators combined into a per-subject, per-tract CV with uncertainty:

1. **g-ratio → CV.** Voxel-wise g from MTsat + neurite density (Stikov et al.,
   2015), then CV = k·v(g) with **v(g) = √(1 − g²)/g**. Deriving CV from g-ratio
   follows Rushton (1951) and Berman, Filo & Mezer (2019, *conduction delays*);
   `k` is a calibration constant. See [BACKGROUND.md](BACKGROUND.md) — the
   proposal's reference list cites the wrong Berman 2019 paper for this formula.
2. **TMS-EEG.** Single-pulse TMS over M1; MEP onset latency (mean/SD over 50
   trials).
3. **Combined.** CV = CST path length / (MEP latency − cortical delay − NMJ
   delay), with NMJ fixed at 0.8 ms and cortical delay from the TMS-EEG evoked
   peak (~15–30 ms). Uncertainty via **Monte-Carlo bootstrap (500 resamples)** →
   per-subject 95% CI.
4. **Validation (N=5 sEEG):** compare against CCEP N1-peak latency.

This math — the g-ratio→CV law and the combined path-length estimate with
bootstrap CIs — is implemented in [`b2ss/cv.py`](b2ss/cv.py).

## 7. The decoder architecture

**B2SS = Transformer encoder → CV modulation gate → Neural-ODE readout.**

- **Encoder.** 4-layer Transformer over a 200 ms EEG window (64 ch @ 250 Hz = 50
  timepoints). Each channel is a token; 8 heads, d=64, FFN=256, LayerNorm.
  Output latent z_t ∈ ℝ²⁵⁶.
- **CV gate.** Learns the integration-window width from CV:
  **τ = σ(W_cv·CV + b_cv)·τ_max + τ_min**, with τ_min = 20 ms, τ_max = 100 ms.
  τ modulates the encoder's temporal attention span (masking) and sets the total
  ODE integration time.
- **Neural-ODE readout.** dz/dt = f_θ(z, t), integrated with **fixed-step Euler
  (Δt = 5 ms, 40 steps)** for real-time inference; readout → 6-DOF wrist
  kinematics (x,y,z position + velocities).
- **Training.** Adam (lr 1e-4), loss = MSE + 1e-5·L2, early stopping, per-subject
  (no cross-subject transfer for the main comparison). ~8.2M params.

**Control decoder:** identical architecture and parameter count, but the CV gate
is removed and **τ is a learned constant per subject** (the freed CV params are
reallocated). This isolates the *information content of CV* from raw capacity —
the single most important design choice for a fair test of H4.

Implemented in [`b2ss/model.py`](b2ss/model.py). We use hand-rolled fixed-step
Euler (the proposal's real-time path) instead of `torchdiffeq`, dropping a
dependency; swap in `torchdiffeq` for adjoint-based training if memory ever bites.

## 8. What this repo builds

The proposal is 36 months of wet-lab + hardware; the *software* is fully
specified and buildable now. This repo delivers:

- **`cv.py`** — the biophysics: g-ratio→CV, combined estimate, bootstrap CIs.
- **`data.py`** — a synthetic generator that bakes in the hypothesized
  **CV→latency law**: a subject's CV sets how far in the past the "intent"
  signal predicts current kinematics. This gives the CV-modulated decoder
  something real to exploit, so the comparison is meaningful in-silico.
- **`model.py`** — B2SS + matched control decoder.
- **`train.py` / `eval.py`** — training loop and the Experiment-3 metrics (MSE,
  Pearson r, effective latency via cross-correlation peak lag).
- **`scripts/run_offline_comparison.py`** — the offline B2SS-vs-control run.
- **`tests/`** — runnable checks: CV monotonicity, gate bounds, forward-pass
  shapes, and "B2SS ≥ control on CV-structured synthetic data."

**Important honesty note:** a win on synthetic data only proves the software
*can* exploit a CV→latency structure when one exists. It is a correctness and
plausibility testbed — **not** evidence for H1–H6, which require the real data.

## 9. Key numbers (proposal)

- Cohort: N=30 healthy (recruit 35) + N=5 sEEG. 6 visits over 5 weeks.
- Compute: A100 for training (~3 h/subject), RTX 4080 for real-time; inference
  budget < 50 ms.
- Timeline: 36 months, 8 phases. Budget: **$789,759** total request.
- Pre-registered on OSF; data → OpenNeuro; code → MIT on GitHub.

## 9b. Publication-grade upgrade (v2): removing the reviewer threats

The first build was a synthetic testbed. v2 adds the machinery to answer the three
threats a reviewer would raise, plus the promised add-ons. Nothing here fakes a
positive result — each is a *fair test*, reported honestly in [RESULTS.md](RESULTS.md).

- **Threat "information vs. prior."** The control already learns its own window τ,
  so a B2SS win could be just a better prior. Two answers: (1) the **gate ablation**
  (`scripts/run_ablation.py`) sweeps training-set size — in the *homogeneous* case
  the CV advantage is honestly characterised as a prior/data-efficiency benefit
  that shrinks with data; (2) the **heterogeneous-CV regime** (`data.make_heterogeneous`),
  where CV varies per trial, is a case a single learned constant *provably cannot*
  fit — there the CV gate carries genuine information and the gap persists. Backed
  by real biology: CV varies ~2× between tracts and ~20× within a tract, and
  learnable-per-input-delay models beat global constants (see BACKGROUND §8).
- **Threat "can scalp EEG carry this?"** The **real-EEG benchmark**
  (`scripts/run_real_benchmark.py`) runs on PhysioNet motor EEG (left vs right
  fist) against **EEGNet** and **CSP+LDA**, within-subject, cropped-window
  training, no planted structure. Honest outcome (see [RESULTS.md](RESULTS.md)):
  the EEG *is* decodable (CSP 0.64, EEGNet 0.61 ≫ chance), so the information is
  there — but the B2SS decoder (~0.48–0.52) does **not** beat these baselines on
  small-trial 2-class EEG, and the weak mu-proxy gate doesn't help. This is the
  wrong regime for the architecture (built for continuous kinematics); the right
  real-data test is a continuous intracortical reach dataset. Reported straight,
  not spun.
- **Threat "effect sizes are optimistic."** The **stats harness** (`b2ss/stats.py`)
  recomputes power with the *verified* Clark effect (d≈0.37, not 0.45): ~60–79
  subjects needed, so the proposal's N=30 is underpowered for it. Also implements
  ICC(2,1) for H1, the H3 mixed model with marginal ΔR², and FDR/Bonferroni.

**Add-ons.** (1) **CV proxy from EEG alone** (`b2ss/proxies.py`) — mu peak frequency,
so the method needs no MRI/TMS (an indirect, honestly-caveated surrogate).
(2) **Uncertainty-aware gate** — a noisy CV estimate shrinks τ toward the
population window (trust the prior only where reliable). (3) **Measured** real-time
latency (`scripts/bench_latency.py`) — comfortably under the 50 ms budget even on
CPU. (4) Four-way gate ablation (`cv`/`learned`/`fixed`/`none`).

## 9c. Phase 8: CV as delay-alignment (mechanism redesign)

The gate *shrinks* the window (removes history), which hurt real continuous decoding.
Phase 8 rebuilds CV usage as delay-**alignment** — time-align each channel by its
conduction delay, full window kept (`ChannelDelay`, `align_mode='cv'`; also added to
the GRU) — and tests it on **real MC_Maze spikes with a known injected latency**
(`scripts/run_latency_bridge.py`), across data sizes. Honest result
([RESULTS.md](RESULTS.md) §6): a *fixed structural* delay is learnable from data, so
measured alignment gives at most a small, not-clearly-significant low-data prior on
the weaker B2SS backbone and **no** benefit on a strong GRU (which learns delays
itself); the GRU dominates B2SS throughout. This *explains* the earlier negatives and
sharpens the thesis-level conclusion: **because CV is a fixed structural parameter, a
within-subject decoder learns the conduction delays anyway** — the only regime where
measured CV could still earn its keep is **cross-subject / zero-shot transfer**.

## 9d. Phase 9: cross-subject transfer — where CV finally helps

Within-subject, a decoder learns the conduction delays from data, so measured CV is
useless (§9b/§9c). The exception is **zero-shot transfer**: a decoder trained on
source subjects and applied to a held-out target *cannot* learn the target's delays.
Test (`scripts/run_transfer.py`): pseudo-subjects from real MC_Maze spikes with
distinct *injected* conduction delays; leave-one-subject-out. Result
([RESULTS.md](RESULTS.md) §7): aligning the target by its measured CV **improves
zero-shot transfer** for both backbones (B2SS +0.076, GRU +0.035 R², consistent
across seeds) — the **first regime where measured CV helps**. Honest scope: the
conduction difference is injected/known (proof-of-mechanism, upper bound), the effect
is modest, and a strong GRU without alignment still beats aligned-B2SS. **Reframed
thesis: CV is a cross-subject conduction normaliser for transfer, not within-subject
decoding information** — the confirmatory test needs a real cohort with a *measured*
CV (none public — see [BACKGROUND.md](BACKGROUND.md) §9).

## 10. Risks the proposal flags

- CV signal too weak to help (null H4) → dataset still yields the g-ratio atlas.
- TMS-EEG CV too noisy → fall back to g-ratio-only CV; sEEG calibrates.
- ccPAS doesn't move CV → higher intensity / theta-burst; measure at 24 & 48 h.
- Inference > 50 ms → FP16, head pruning, fixed-step Euler, precomputed ODE
  trajectory. *(This repo already uses fixed-step Euler.)*
