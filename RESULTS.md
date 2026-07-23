# B2SS — Results

What the experiments show, and — as important — what they don't. Every number is
reproducible from the scripts; raw data is in `results/*.json`. Everything is CPU-only on
public or synthetic data. Nothing here is evidence for a clinical hypothesis.

**Read this first.** The project tested one idea four times and falsified it four times:
that a measured conduction velocity helps a decoder. Part II records those falsifications
in detail, because the mechanism of each failure is the interesting part. Part I is what
came out of them — a calibrated measurement of *where* the cross-session gap actually
lives, and the practical consequence of the answer.

This document was rewritten against [BETTER.md](BETTER.md), an adversarial review that
found the previous version's headline unsupported by its own JSON. Claims that did not
survive that review are gone; the ones that replaced them are marked with the analysis
that backs them.

---

# Part I — the result

## I.1. The decomposition: where the cross-session gap lives

Conduction alignment is used here as an **instrument**, not an adapter. Give a frozen
decoder a per-channel temporal realignment fit to the target session, measure the change in
velocity R² over no alignment, and you have the share of the gap attributable to timing —
in the same units across settings.

An instrument that reads zero everywhere measures nothing, so the first requirement is a
setting where it must read positive. Real MC_Maze spikes with a **known per-group latency
injected** provide it: there, timing is the entire gap by construction.

| setting | conduction/timing marginal (Δ velocity R² over no-norm) |
| --- | --- |
| **positive control** — injected latency, real MC_Maze spikes | **+0.249 [+0.174, +0.324]** (CI-separated) |
| real multi-session intracortical — MC_Maze S/M/L, per-electrode | **−0.015 [−0.027, −0.004]** (null) |
| real cross-session EEG — Zhou2016 (accuracy units) | **−0.003** (null) |

**The reading.** The instrument responds strongly where timing dominates and reads flat —
slightly negative — on both real cross-session gaps. Conduction timing is **not** where the
real multi-session gap lives. What is left is representation drift: unit turnover,
firing-rate change, tuning change. Between two days of the same monkey on the same array,
the recorded neurons themselves differ.

**Scope, stated plainly.** The positive control is an oracle in *two* ways, not one: the
conduction difference is injected and known, **and** the aligner is handed the same
channel→group map (`arange(C) % 8`) used to generate it. That is correct for a control — it
establishes the instrument's ceiling — and would be dishonest quoted as a result. Nothing
here says a *measured* CV in a real cohort would read zero; it says timing does not explain
the drift between sessions of one subject, which is a narrower claim.

`python scripts/run_decomposition_figure.py` → `results/decomposition.png`.

## I.2. The consequence: per-session calibration is data-starved, and the standard fix diverges

If the gap is per-channel gain and offset, the obvious remedy is to standardise each
session's channels — the MPA / AdaBN / Euclidean-alignment family. In the online BCI regime
you must do that from the *first few windows* of a new session, before calibration data has
accumulated. That is where it breaks.

On the 96-electrode Indy array, **14.5% of channels have sd < 0.1 over a 25-window slice**
(measured, `results/indy_calibration.json`). Estimating a per-channel scale from that slice
and dividing by it does not degrade gracefully — it **diverges**. CADENCE shrinks each
estimate toward the source prior by `w = n/(n+τ)`, so the adapter starts near the identity
and earns its way to a full standardiser as evidence arrives. The estimator is standard —
BACKGROUND §10.

> **The reason this works is not the reason we first gave.** See §I.2b: the small-budget
> failure is *chronological bias*, not sampling noise, and shrinkage helps because it
> declines to commit to a biased estimate rather than because it denoises a noisy one. The
> distinction is not cosmetic — it predicts which fixes work, and the principled
> noise-based fix (empirical Bayes) fails.

Frozen GRU on the 3 earliest sessions, adapt label-free from the first N windows of each of
8 held-out sessions, 3 seeds. `all` = every training window in the session (~15k, ~300 s) —
the true full-calibration point, not the 2000-window slice a previous version of this
document called "full calibration."

| method | N=25 | N=50 | N=100 | N=200 | N=500 | N=2000 | N=all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no-adapt | +0.409 | +0.409 | +0.409 | +0.409 | +0.409 | +0.409 | +0.409 |
| mpa-nofloor | -0.217 | -0.074 | -0.004 | +0.149 | +0.250 | +0.510 | +0.530 |
| mpa | +0.027 | +0.049 | +0.123 | +0.233 | +0.258 | +0.518 | +0.543 |
| tent | +0.306 | +0.300 | +0.317 | +0.374 | +0.377 | +0.489 | +0.489 |
| free-lora | -0.534 | -0.419 | -0.368 | -0.560 | -0.316 | -0.332 | -0.277 |
| cadence | +0.437 | +0.455 | +0.464 | +0.455 | +0.396 | +0.515 | +0.544 |


**CADENCE − no-adapt**

| N | Δ R² | 95% CI | p | sessions won | survives BH |
| --- | --- | --- | --- | --- | --- |
| 25 | +0.029 | [+0.016, +0.041] | 0.001 | 8/8 | yes |
| 50 | +0.047 | [+0.016, +0.077] | 0.008 | 7/8 | yes |
| 100 | +0.056 | [+0.004, +0.107] | 0.038 | 7/8 | no |
| 200 | +0.047 | [-0.020, +0.113] | 0.140 | 7/8 | no |
| 500 | -0.013 | [-0.120, +0.094] | 0.777 | 5/8 | no |
| 2000 | +0.106 | [+0.046, +0.166] | 0.004 | 8/8 | yes |
| all | +0.135 | [+0.062, +0.209] | 0.003 | 7/8 | yes |

**CADENCE − MPA (floored)**

| N | Δ R² | 95% CI | p | sessions won | survives BH |
| --- | --- | --- | --- | --- | --- |
| 25 | +0.410 | [+0.173, +0.647] | 0.005 | 8/8 | yes |
| 50 | +0.406 | [+0.067, +0.745] | 0.025 | 8/8 | no |
| 100 | +0.342 | [+0.031, +0.653] | 0.035 | 8/8 | no |
| 200 | +0.222 | [-0.022, +0.466] | 0.068 | 8/8 | no |
| 500 | +0.138 | [-0.030, +0.305] | 0.094 | 8/8 | no |
| 2000 | -0.003 | [-0.016, +0.009] | 0.537 | 4/8 | no |
| all | +0.001 | [-0.001, +0.003] | 0.183 | 6/8 | no |

**MPA (floored) − no-adapt**

| N | Δ R² | 95% CI | p | sessions won | survives BH |
| --- | --- | --- | --- | --- | --- |
| 25 | -0.382 | [-0.618, -0.146] | 0.006 | 0/8 | yes |
| 50 | -0.359 | [-0.695, -0.024] | 0.039 | 0/8 | no |
| 100 | -0.286 | [-0.585, +0.012] | 0.058 | 2/8 | no |
| 200 | -0.176 | [-0.434, +0.082] | 0.152 | 3/8 | no |
| 500 | -0.151 | [-0.416, +0.114] | 0.221 | 3/8 | no |
| 2000 | +0.109 | [+0.048, +0.170] | 0.004 | 7/8 | yes |
| all | +0.134 | [+0.059, +0.209] | 0.004 | 7/8 | yes |

**What the curve shows.**

1. **The unfloored standardiser diverges, it does not degrade.** −0.217 at N=25, still
   negative at N=100. That is not "MPA is weak in the scarce regime" — it is a numerical
   failure caused by dividing by the scale of a channel that happened to be quiet during
   calibration. A previous version of this document reported a "+0.65 margin over MPA"
   built almost entirely on this. The floor is a one-line fix and it belongs in the
   baseline, not in the contribution.

2. **Even a correctly-floored standardiser is worse than not adapting, below ~200 windows.**
   `mpa − no-adapt` is **−0.382 (p=0.006, 0/8 sessions)** at N=25 and still −0.286 at
   N=100. This is the finding that survives, and it is stronger than the one it replaced:
   per-session standardisation is not merely unhelpful when calibration data is scarce, it
   is *actively harmful*, and the floor does not fix that — it only stops the divergence.

3. **Shrinkage is what makes adaptation safe at small N — by a small margin over doing
   nothing.** `cadence − no-adapt` is **+0.029 (p=0.001, 8/8 sessions)** at N=25, rising to
   +0.056 at N=100. Consistent across every session and every seed, and small. Quote it as
   +0.03–0.06 R², never as the margin over a diverging baseline.

4. **At true full calibration the two converge exactly.** With every training window in the
   session (~17,800 on average), CADENCE 0.544 vs MPA 0.543 — **+0.001, p=0.183**. The
   shrinkage weight is `w = n/(n+τ) ≈ 0.99` there, so CADENCE *is* MPA at that budget. The
   old claim "ties MPA at full calibration" turns out to be true; it had simply never been
   tested at full calibration, only at N=2000 (13% of the session).

5. **The N=500 dip is real and explained.** CADENCE falls to 0.396, slightly *below*
   no-adapt. §I.3 shows why: at that budget τ=200 shrinks too little. This is a limitation of
   a single fixed τ, not noise.

**So the honest summary of this experiment:** below ~200 calibration windows, adapting with
a per-session standardiser is worse than not adapting; shrinking the estimate toward the
source prior makes adaptation safe and buys a small consistent gain; above ~2000 windows
the shrinkage is inert and the two methods are the same estimator. `tent` is a distant
third throughout and `free-lora` never works.

## I.2b. Why it actually fails: the calibration window is biased, not noisy

§I.2 gives the standard explanation — few windows, noisy per-channel estimates. It is the
explanation everyone reaches for, and testing it takes one change to the experimental
design: hold the number of calibration windows fixed and change only *how they are drawn*.

`run_calibration_bias.py`, 8 target sessions × 3 seeds:

| | N=25 | N=50 | N=100 | N=200 | N=500 |
| --- | --- | --- | --- | --- | --- |
| MPA — **first** N windows (what an online BCI gets) | +0.027 | +0.049 | +0.123 | +0.233 | +0.258 |
| MPA — **random** N windows (same N, no time bias) | **+0.520** | **+0.535** | **+0.537** | +0.543 | +0.544 |
| CADENCE — first N | +0.437 | +0.455 | +0.464 | +0.455 | +0.396 |
| CADENCE — random N | +0.451 | +0.474 | +0.498 | +0.521 | +0.539 |

Paired over sessions, **random − first**:

| | N=25 | N=100 | N=500 |
| --- | --- | --- | --- |
| MPA | **+0.493** (p=0.003, 7/8) | +0.414 (p=0.020, 7/8) | +0.286 (p=0.057, 7/8) |
| CADENCE | +0.013 (p=0.176, 5/8) | +0.034 (p=0.082, 6/8) | +0.143 (p=0.038, 7/8) |

**Twenty-five randomly-drawn windows are worth more than two thousand consecutive ones.**
The estimator, the sample size, and the data are identical; only the draw changed. So the
online failure is **chronological bias** — the opening minute of a session is systematically
unrepresentative of the rest of it — and not sampling noise.

Three consequences, in increasing order of how much they cost us:

1. **§I.2's mechanism was wrong.** Shrinkage does not denoise a noisy estimate. It helps
   because it *declines to commit* to a biased one, which is the right behaviour for a
   different reason than the one we gave. CADENCE is nearly indifferent to the draw
   (+0.013 at N=25) precisely because it barely uses it.
2. **It predicts the empirical-Bayes failure exactly.** EB models sampling variance and is
   structurally blind to bias, so it measures the large genuine drift, concludes "trust the
   data", and inherits the bias — collapsing onto MPA (§I.3). A principled fix to the wrong
   problem is still wrong.
3. **Shrinkage is not the best available fix, and we should not present it as one.** An
   unbiased sample of *the same size* beats it: MPA-random at N=25 (0.520) beats CADENCE on
   the first 2000 windows (0.515). Where a deployment can spread its calibration windows
   across a session — interleaved, or a brief revisit — that dominates any estimator-side
   correction. CADENCE's honest niche is the genuine cold start, the first seconds of a
   session, when no spread is available yet.

**The general form of this**, which is the part worth taking to a test-time-adaptation
audience rather than a BCI one: *online calibration data is not a random sample of the
distribution it calibrates for.* Any TTA method whose statistics assume exchangeability
inherits the bias, and the standard diagnostic — vary N and watch the curve — cannot see it,
because it confounds sample size with sample position. Varying the draw at fixed N separates
them.

Figure: `results/calibration_bias.png`.

### Second subject (loco) — a directional check, underpowered

The obvious question is whether the bias effect is an Indy artifact. The same Zenodo record
carries a second monkey, **loco**, and `run_calibration_bias.py --subject loco --max-chan 96`
runs the identical diagnostic on it (M1-only, to match Indy's 96 channels). Honest caveats up
front: we downloaded **6 of loco's 10 sessions** (the set is ~12 GB and this connection could
not finish it), so with `--held-in 2` there are only **4 target sessions**, and loco's
cross-session decoding from two source sessions is weak — absolute R² hovers near zero.

| N | mpa-first | mpa-random | random − first (paired over 4 sessions) |
| --- | --- | --- | --- |
| 25 | −0.344 | −0.059 | **+0.285**, p=0.291, 4/4 sessions |
| 50 | −0.074 | −0.025 | +0.050, p=0.437, 3/4 |
| 100 | −0.006 | −0.015 | −0.008, p=0.900, 2/4 |

**The direction agrees at the smallest budget** — a random draw beats the first-N draw on
all four sessions at N=25, the same sign as Indy — **but it is not significant on four
sessions** (p=0.291), and it does not persist at larger N where the base decoder is anyway
barely above zero. loco's native 192-channel (M1+S1) configuration gives the same picture
(+0.181 at N=25, p=0.434). So this is a **directional replication, not a demonstration**:
consistent with the Indy finding, far too underpowered to stand on its own. Settling it needs
the full 10-session loco set, which the download did not finish. We report it here rather
than omit it, and we do not count it as confirmation.

`results/calibration_bias_loco.json`.

## I.3. Is τ tuned on the evaluation sessions?

`shrink_tau = 200` was a default, and it sits in the middle of the budget grid it is
reported on — the exact shape of a hyperparameter fitted to the test set. `run_tau_sweep.py`
answers it two ways: sweep one-at-a-time around the defaults, and select τ by
**leave-one-session-out** (for each held-out session take the τ that was best on the other
seven, then score it on the held-out one).

| N | τ=50 | τ=100 | τ=200 | τ=400 | τ=800 | LOSO-selected | LOSO − default | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | +0.438 | +0.447 | +0.437 | +0.426 | +0.419 | +0.437 | −0.0006 | 0.971 |
| 100 | +0.370 | +0.437 | **+0.464** | +0.459 | +0.443 | +0.464 | +0.0000 | 1.000 |
| 500 | +0.300 | +0.338 | +0.396 | +0.450 | **+0.474** | +0.474 | +0.0784 | 0.134 |
| 2000 | +0.519 | +0.519 | +0.515 | +0.516 | +0.517 | +0.514 | −0.0010 | 0.776 |

**LOSO selection never significantly beats the fixed default** (largest gap +0.078,
p = 0.134), so the reported numbers do not depend on having picked τ = 200 with hindsight.

**But the surface is not flat, and it explains a real wrinkle.** At N = 500 the best τ is
800, not 200 — and N = 500 is exactly the budget where CADENCE dips (§I.2). The dip is not
noise: at that budget τ = 200 shrinks too little, trusting a per-session estimate that is
still noisy. LOSO picks τ = 800 there and recovers most of it. An honest reading is that a
*single* τ is a compromise across budgets, and a schedule (τ growing with the array's
sparsity, or selected online) is the obvious improvement we did not make.

### Removing τ entirely — and why that makes things worse

If a single τ is a compromise, the principled move is to stop guessing it. `shrink="eb"`
estimates the shrinkage from the data (Efron–Morris, with the sampling noise subtracted from
the observed dispersion; separate weights for the mean and the scale, because the relative
error of a sample std is 1/√(2n) for every channel regardless of how loud it is). No
hyperparameter at all.

The comparison is run by `run_tau_sweep.py` (`shrink="eb"` column) at its budget grid; the
full-grid numbers below are from the same code over all budgets:

| N | 25 | 50 | 100 | 500 | 2000 | all |
| --- | --- | --- | --- | --- | --- | --- |
| MPA | +0.027 | +0.049 | +0.123 | +0.258 | +0.518 | +0.543 |
| CADENCE, fixed τ=200 | **+0.437** | **+0.455** | **+0.464** | +0.396 | +0.515 | +0.544 |
| CADENCE, empirical Bayes | +0.037 | +0.054 | +0.125 | +0.259 | +0.518 | +0.543 |

`eb − fixed` at N=25 is **−0.400 [−0.635, −0.166], p=0.005, 0/8 sessions**. EB does not
merely underperform — it **collapses onto the plain standardiser** it was supposed to
improve (`eb − mpa` = +0.010, n.s.).

§I.2b says why, and this is the cleanest evidence for it: EB accounts for variance and is
blind to bias. Given genuinely large between-session drift it correctly concludes the
session estimate is informative, sets `w → 1`, and inherits the full chronological bias. The
fixed τ wins by being *unprincipled* — it hedges against a failure mode it was never
designed to address. We keep `shrink="eb"` in the code as a runnable negative result rather
than deleting it, because "the textbook version of our own estimator fails here" is the kind
of thing a reader should be able to check.

The **`std_floor` sweep is flat to three decimals** across 0.02–0.4 at every budget. That is
not a null result, it is a structural fact: the shrunk scale `w·s_t + (1−w)·1` is already
bounded below by `1−w`, so the floor almost never binds *inside CADENCE*. The floor matters
for the **unshrunk** standardiser, which has no such protection — which is precisely §I.2's
finding, arrived at from the other direction.

`results/tau_sweep.json`, `results/tau_sweep.png`.

## I.4. The continual stream

Freeze the source decoder; replay the 8 remaining sessions in temporal order with 2
revisits; every method adapts label-free at each visit and is scored before the next.

Two things changed here after the review, and both changed conclusions.

**Stability is now measured as regret, not floor-crossings.** The previous version reported
a collapse-rate: the fraction of visits below R² = 0.2. That metric assigned **the identical
value (0.10) to No-Adapt, MPA, CoTTA, RDumb and CADENCE** across all three seeds — one
genuinely hard session in ten visits, hard for everyone. A metric that cannot distinguish
the method from doing nothing is not evidence, and the auto-verdict that consumed it
("PARETO WIN", conditioned on `cadence_collapse <= noadapt_collapse`) was satisfiable by any
method that changed nothing. **Regret** is measured against the per-session No-Adapt
trajectory: how often adapting *lost* accuracy, and by how much.

**The recalibration ceiling is now a fine-tune.** The previous ceiling trained a fresh
decoder per session from scratch, scored 0.124, and was used to claim that cheap adaptation
"beats per-session retraining." Nobody recalibrates a BCI that way. `finetune` continues the
frozen source decoder on the session at a lower learning rate — what a deployment actually
does — and `scratch` is kept as a secondary row.

| method | kind | cumulative R² | worst | regret rate / mean | collapse | BWT |
| --- | --- | --- | --- | --- | --- | --- |
| scratch | recalibration ceiling | +0.757 | +0.718 | 0.00 / 0.000 | 0.00 | +0.000 |
| finetune | recalibration ceiling | +0.741 | +0.662 | 0.00 / 0.000 | 0.00 | +0.000 |
| mpa | label-free adapter | +0.575 | -0.093 | 0.10 / 0.062 | 0.10 | +0.000 |
| cadence | label-free adapter | +0.540 | -0.031 | 0.00 / 0.000 | 0.10 | +0.000 |
| rdumb | label-free adapter | +0.505 | -0.056 | 0.13 / 0.036 | 0.10 | -0.031 |
| cotta | label-free adapter | +0.486 | -0.003 | 0.10 / 0.016 | 0.10 | -0.003 |
| no-adapt | frozen (reference) | +0.408 | -0.031 | 0.00 / 0.000 | 0.10 | +0.000 |
| tent | label-free adapter | +0.401 | -0.029 | 0.50 / 0.137 | 0.17 | -0.098 |
| free-lora | unstructured adapter | -0.155 | -0.684 | 0.97 / 0.587 | 1.00 | +0.031 |
| nomad | unstructured adapter | -0.453 | -1.179 | 1.00 / 0.862 | 1.00 | +0.037 |

| CADENCE vs | Δ cumulative R² | 95% CI | p | visits won |
| --- | --- | --- | --- | --- |
| nomad | +0.993 | [+0.750, +1.236] | 0.000 | 10/10 |
| free-lora | +0.695 | [+0.494, +0.897] | 0.000 | 10/10 |
| tent | +0.139 | [+0.022, +0.256] | 0.025 | 8/10 |
| no-adapt | +0.132 | [+0.059, +0.204] | 0.003 | 8/10 |
| cotta | +0.054 | [+0.006, +0.103] | 0.032 | 7/10 |
| rdumb | +0.035 | [-0.028, +0.098] | 0.238 | 6/10 |
| mpa | -0.035 | [-0.099, +0.029] | 0.252 | 4/10 |

**Recalibration wins, and it is not close.** Given the session's own input normalisation, a
per-session decoder reaches **0.757** (from scratch) or **0.741** (fine-tuned from the frozen
source) against the best label-free adapter's 0.575. It also never falls below the collapse
floor and never loses to No-Adapt on any visit. **What label-free adaptation buys is
cheapness, not accuracy** — 2·n_chan adapted parameters, no labels, no gradient step at
deployment — and that is the honest framing of every adapter in this table.

This corrects a claim the previous version of this document made. That version reported a
ceiling of 0.124 and concluded that "frozen + cheap adaptation beats per-session
retraining." Two things were wrong with it: the ceiling trained from scratch rather than
fine-tuning (the smaller error), and it was fed inputs z-scored with the **source** session's
statistics rather than the session's own — the exact handicap the alignment baselines exist
to remove. Isolating that on three sessions, identical data and schedule:

| | frozen | fine-tune, source-normalised | fine-tune, own-normalised | scratch, own-normalised |
| --- | --- | --- | --- | --- |
| velocity R² | 0.511 | **0.024** | **0.741** | **0.764** |

**Among the label-free methods, MPA leads — not CADENCE.** 0.575 vs 0.540. The difference is
not significant (−0.035, p = 0.252, 4/10 visits), so the fair statement is that they are
indistinguishable here with MPA ahead on the point estimate. This is exactly what §I.2
predicts: the stream's adaptation budget is 1500 windows, well past the point where
shrinkage stops helping and starts costing.

**CADENCE's real property on this stream is that it never hurts.** It is the only adapter
with **regret 0.00 / 0.000** — across 3 seeds and 10 visits it did not lose to the frozen
decoder once. MPA loses on 10% of visits (mean shortfall 0.062), Tent on 50% (0.137). If the
deployment question is "can I turn this on without risking a session," CADENCE and No-Adapt
are the only two answers in the table, and CADENCE is +0.132 better than No-Adapt
(p = 0.003, 8/10 visits).

**But that property is the controller, not the shrinkage — and we had this wrong.** The
collapse sensor reverts the fast head to identity when the unsupervised objective spikes,
and on this stream it **fires on 3, 3, 2 visits** across the three seeds (0 for every other
method — `results/indy_stream.json:collapse_reverts`). Those are exactly the visits where the
adapter would otherwise have lost, so the revert converts a would-be loss into an exact tie
with No-Adapt — which is *why* the regret is 0.000 rather than merely small, and why two
visits in `results/indy_stream_regret.png` sit precisely on the zero line. An earlier version
of `cadence.py` asserted the controller "does not fire in normal use" for the closed-form
head; that was false, and it mis-credited to the shrinkage the one property of the method
that survives scrutiny. The honest statement: **shrinkage keeps the adapter close to safe,
and the controller catches the visits where close is not enough.** Figure:
`results/indy_stream_regret.png`.

**Collapse-rate is degenerate here and regret is not.** Note the collapse column: 0.10 for
No-Adapt, MPA, CoTTA, RDumb *and* CADENCE — identically, across all three seeds. One hard
session in ten visits, hard for everyone. That metric cannot distinguish the method from
doing nothing, which is why the previous version's "Pareto win" was vacuous. Regret
separates the same methods cleanly.

**The structure ablation, unconfounded.** Tent (gradient-fit, per-channel **diagonal**) vs
free-LoRA (gradient-fit, dense **rank-1**) — same objective, same optimiser, same lr, same
steps, same 2·n_chan parameters, differing only in the head's structure:

> **tent − free-lora = +0.556 [+0.337, +0.776], p < 0.001, 10/10 visits.**

Interpretable structure, not parameter count, is what makes label-free adaptation survivable.
The previous version claimed this from CADENCE vs free-LoRA, which confounds structure with
optimiser (CADENCE's head is closed-form, free-LoRA's is gradient-fit) — the comparison above
is the one that isolates it.

**Unstructured adaptation collapses outright.** NoMAD (full-rank readin, −0.453) and
free-LoRA (dense rank-1, −0.155) go negative with regret ≈ 1.00 — they lose to the frozen
decoder on essentially every visit. Given expressive free parameters, the label-free
objective finds moment-matching solutions that destroy decoding.

**Forgetting.** Tent carries state forward and accumulates error (BWT −0.098, dropping to
No-Adapt's level by the end). CADENCE and MPA both score BWT = 0.000 — but that is a
*property, not an achievement*: both refit from scratch each session and are memoryless by
construction. Reporting it as a CADENCE advantage over Tent is fair; reporting it as a
CADENCE advantage over MPA would not be.

---

# Part II — what we ruled out to get here

Four falsifications, in the order they happened. They are kept in full because the
*mechanism* of each failure is what produced Part I: once you understand why a decoder does
not need to be told its conduction delays, the only remaining question is whether conduction
explains any of the cross-session gap — which is the measurement Part I makes.

The chain, in one line each:

1. **CV as a temporal-window prior** helps on synthetic data with a planted CV→window law
   (§II.1), and **hurts** on real continuous decoding (§II.2) — recency-masking discards history
   the decoder was using.
2. **CV via an EEG mu-frequency proxy** gives no benefit even after the masking confound was
   removed (§II.3) — the proxy is uninformative, as the weak literature link predicts.
3. **CV as delay-alignment** rather than window-shrinking fails within-subject too (§II.6): a
   *fixed structural* delay is learnable from data, so being told it adds little.
4. **CV as a cross-subject conduction normaliser** works where the conduction difference is
   injected and known (§II.7, §II.8.1) and is **null on every real cross-session gap** (§II.8.2,
   §II.8.3). That pair of results is the instrument and its calibration — §I.1.

## II.1. Gate ablation — "is CV information, or just a better prior?"

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

## II.2. Intracortical benchmark — the decisive REGRESSION test

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

## II.3. Real-EEG benchmark — "can scalp EEG carry this / does the architecture work?"

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
confirms this is the *proxy*, not a masking artifact. Together with §II.2, the picture
is consistent: the CV mechanism helps only where a CV→window structure genuinely
exists (synthetic), and neither real dataset — lacking a *measured* CV — shows a
benefit.

---

## II.4. Real-time latency — the proposal's <50 ms budget (§4.7)

`python scripts/bench_latency.py` — single-window (batch=1) inference, CPU, 1 thread.

| decoder | params | eager fp32 p95 | torchscript p95 |
| --- | --- | --- | --- |
| default (d_model 64, 2 layers) | 83k | 0.94 ms | 0.57 ms |
| proposal-size (d_model 256, 4 layers) | 1.75M | 2.49 ms | 1.96 ms |

Comfortably under 50 ms even on CPU eager; the proposal targets ONNX+CUDA on an
RTX 4080, which is faster still. **The <50 ms claim holds with wide margin.**

---

## II.5. Statistical harness — corrected effect sizes (proposal §6)

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
correction — so the pre-registered proposal §6 plan is runnable, not just described.

---

## II.6. Phase 8 — CV as delay-alignment (mechanism redesign)

The gate *removes* information (window-shrinking), which is why it hurt real
continuous decoding (§II.2). Phase 8 rebuilds the mechanism as delay-**alignment** —
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
data, and nothing for a strong decoder. This *explains* the §II.2/§II.3 negatives: because
CV is a fixed structural parameter, a within-subject decoder learns the conduction
delays from data, so being told them adds little. The one regime not excluded is
**cross-subject / zero-shot transfer** (a decoder that never saw the target
subject's data, given that subject's measured CV) — the recommended next test.

Figure: `results/latency_bridge.png`.

## II.7. Phase 9 — cross-subject / zero-shot transfer (the one positive regime)

Within-subject, a decoder learns the conduction delays from data, so measured CV
adds nothing (§II.2, §II.3, §II.6). The one regime where it *can't* learn them: **zero-shot
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

## II.8. Phase 10 — the pivot: conduction normalization for transfer

The project pivoted (see [PIVOT.md](PIVOT.md)) from "CV-modulated decoder" to a
**conduction normaliser** that cuts BCI recalibration. Train a decoder once on a
source pool, freeze it, adapt only a low-dim per-tract delay `δ` per target.

### 8.1 Calibration-cost spectrum (controlled, `run_transfer_modes.py`)

Real MC_Maze spikes → 4 pseudo-subjects with distinct injected conduction; source
canonicalised + frozen; only the target-side alignment varies. Velocity R², 4 folds
× 3 seeds:

| method | target labels | velocity R² (mean [95% CI]) |
| --- | --- | --- |
| no-norm (naive transfer) | 0 | 0.406 [0.276, 0.536] |
| **zero-shot (measured CV)** | **0** | **0.655 [0.555, 0.755]** |
| unsupervised (δ-fit) | 0 (unlabeled) | 0.311 [0.155, 0.467] |
| few-shot(5) | 5 | 0.388 [0.252, 0.524] |
| few-shot(20) | 20 | 0.589 [0.456, 0.722] |
| few-shot(100) | 100 | 0.637 [0.546, 0.728] |
| free-delay(100) *(ablation)* | 100 | 0.553 [0.437, 0.669] |
| full-retrain (own-normalised ceiling) | all | 0.439 [0.371, 0.507] |

**This is the pivot's headline — and it's positive:**

1. **Zero-shot beats retraining, with zero target data.** Measured-CV alignment
   (0.655) beats no-norm by **+0.249 (CIs separated)** and beats the
   retraining ceiling (0.439). This claim survived the ceiling fix that
   killed its counterpart on the Indy stream — and the contrast is informative. Here the
   pseudo-subjects are one MC_Maze session with *injected delays*, so their per-channel
   statistics are nearly identical to the source and the source normalisation costs the
   ceiling almost nothing (0.439 own-normalised vs 0.403 before). On
   Indy, where sessions are genuinely different days with real channel drift, the same fix
   moved the ceiling from 0.124 to 0.757. **The handicap scales with how much the channel
   statistics actually differ**, which is the thing this whole document is about.
2. **Clean calibration curve.** Few-shot climbs 0.39 → 0.59 →
   0.64 for 5 → 20 → 100 labeled trials, approaching zero-shot — a modest
   amount of calibration recovers most of the benefit.
3. **The conduction structure helps.** Structured few-shot(100) (0.637) beats
   unstructured free per-channel delays (0.553) by +0.084 — the low-dim
   per-tract grouping is more data-efficient than free delays. Caveat that belongs here: the
   aligner is handed the *same* `arange(C) % 8` grouping used to inject the delays, so this
   compares a correctly-specified structure against an unstructured one, not structure
   discovery.
4. **Unsupervised fails honestly.** CORAL latent-moment matching (0.311) does *not*
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

_(The retrain here is not own-normalised, unlike the Indy stream ceiling — MC_Maze uses
random trial splits, not a chronological one, so there is no opening-window bias to correct
and the source vs own frame give nearly the same answer. The 0.759 is if anything a slight
underestimate of the true ceiling, which only strengthens the verdict below.)_

**Verdict — conduction alignment gives NO real cross-session benefit** (best δ-fit gain
over no-norm = **−0.015**), and full-retrain dominates (0.759). Unlike the controlled spectrum
(where the gap was pure conduction), the real cross-session gap is dominated by **unit
turnover, firing-rate drift, and tuning changes** — even with corresponding electrodes,
the recorded neurons differ across days. So conduction timing is *not* the axis of the
real gap, and a conduction normaliser alone cannot close it. This is the honest scope
boundary, and it is *why* the mechanism only helps where conduction dominates (§II.8.1).

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

---

# Honesty ledger

Everything a reviewer would otherwise have to find in the JSON themselves.

**On the original hypothesis (Part II)**

- The synthetic ablation proves a *mechanism* — a CV-derived integration window helps when a
  CV→window structure exists — on data where we planted that structure. It is not evidence
  the structure exists in brains.
- On **both** real datasets the CV gate gives no benefit: on EEG the mu-frequency proxy is
  uninformative (confirmed after the F1 masking confound was removed, so it is the proxy and
  not an artifact); on intracortical continuous decoding the gate actively *hurts*, because
  recency-masking discards history the decoder was using.
- B2SS is a real decoder (it beats linear Ridge) but is not competitive with a plain GRU
  (intracortical) or with EEGNet/CSP (EEG). The GRU is the backbone everywhere downstream.
- Reframing CV as delay-*alignment* also fails within-subject: a fixed structural delay is
  learnable from data, so being told it adds at most a small, not-clearly-significant
  low-data prior.
- The corrected power analysis concerns the *proposal's* design, not a result here.

**On the decomposition (§I.1)**

- The positive control is an oracle **twice over**: the conduction difference is injected and
  known, *and* the aligner is handed the same `arange(C) % 8` channel→group map used to
  generate it. It establishes the instrument's ceiling. It is not a result.
- The nulls are within-subject-across-days and cross-session EEG. "Timing does not explain
  the drift between two days of the same monkey" is **not** the same claim as "conduction
  velocity is useless as a decoder prior." Only a cohort with measured per-subject CV
  separates them, and BACKGROUND §9 documents that no public dataset provides one.

**On CADENCE (§I.2–§I.4)**

- The estimator resembles textbook empirical Bayes, and shrinking normalization statistics
  toward source statistics is prior art in TTA (BACKGROUND §10) — but note §I.3: the
  *actual* textbook version fails here, so we cannot claim the principled pedigree either.
  A fixed τ works because it hedges against bias, which is not what the derivation says.
- **The mechanism we first published for this was wrong.** We said the small-budget failure
  was noisy estimates. It is chronological bias (§I.2b), which the noise story predicts
  incorrectly in at least two checkable ways (EB should have helped; a random draw of the
  same size should not have mattered). Both checks came out against the noise story.
- **Shrinkage is not the best fix available.** 25 randomly-drawn calibration windows beat
  CADENCE on 2000 consecutive ones. Where a deployment can spread calibration across a
  session, that dominates any estimator-side correction; CADENCE's niche is the cold start.
- The gain over **not adapting** is +0.029 to +0.056 R² at N ≤ 100 — small, consistent
  (7–8/8 sessions), and the number to quote. The large margins in earlier drafts were
  measured against a baseline that was diverging for want of a one-line scale floor.
- **The zero-regret property is not the shrinkage alone.** The collapse controller fires on
  ~3 of 10 stream visits, and those are the visits where the adapter would otherwise have
  lost — it reverts to identity, converting a loss into an exact tie. An earlier docstring
  claimed the controller never fired for the closed-form head; that was wrong and it
  mis-attributed the one property of the method that survives. See §I.4.
- CADENCE **does not lead the continual stream**. See §I.4.
- At full calibration the shrinkage is inert (`w ≈ 0.99`) and CADENCE *is* MPA.
- A single fixed τ is a compromise: at N=500 it under-shrinks and CADENCE drops below
  no-adapt. A τ schedule is the obvious fix we did not build.
- `BWT = 0.000` is a property, not an achievement — the adapter is memoryless by
  construction, and MPA scores the same for the same reason.

**On what beats it**

- **Per-session recalibration wins outright**, once given the session's own input
  normalisation. An earlier version of this harness fed the ceiling source-normalised
  inputs, scored it at ~0.13, and used that to claim cheap adaptation beats retraining.
  Measured properly the ceiling is ~0.75. Label-free adaptation buys **cheapness, not
  accuracy** — 2·n_chan adapted parameters and no labels, at a real accuracy cost.

**On scope**

- **One subject.** Every stream number is monkey Indy, one 96-electrode array, 11 sessions
  over ~1 month. `--subject loco` is wired through and unrun (~12 GB).
- Small N elsewhere: EEG 8 subjects, 2–3 seeds; single MC_Maze sessions in the brackets.
- None of this substitutes for the wet-lab study (MRI g-ratio, TMS-EEG, sEEG, closed-loop
  control) — the only setting with a measured CV to test.
