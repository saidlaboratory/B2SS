# BETTER — what to fix before submitting CADENCE

> **STATUS: all P0, P1 and P2 items below are done.** See §0 for what each one turned into.
> Two of the review's own conclusions were wrong and are corrected there — one in the
> project's favour, one against it.

---

## 0. Outcome of every item

| item | status | what actually happened |
| --- | --- | --- |
| **P0.1** floor the MPA baseline | done | `MPA(std_floor=0.1)`; both variants are separate rows in §2. The floor recovers 0.24 of the 0.65 margin; the rest was real. |
| **P0.2** true full-calibration budget | done | Budget grid extends to `all` (~17.8k windows). **The review's prediction was wrong**: MPA does *not* win there — CADENCE 0.544 vs MPA 0.543, p=0.183. "Ties at full calibration" was true, just untested. |
| **P0.3** re-frame the headline | done | Headline is now the paired result vs no-adapt (+0.029, p=0.001, 8/8) and the finding that even *floored* MPA is −0.382 below no-adapt at N=25. Stronger than the claim it replaced. |
| **P0.4** state the MPA-beats-CADENCE result | done | Stated in RESULTS §4, in the script's verdict string, and in the README banner. Verdict now requires a strict accuracy win. |
| **P0.5** unconfounded structure ablation | done | Tent vs free-LoRA (gradient-fit both sides): +0.556, p<0.001, 10/10 visits. |
| **P1.1** τ selection protocol | done | `run_tau_sweep.py`. LOSO never significantly beats the default (max +0.078, p=0.134) — τ=200 is not tuned on the eval set. Also **explains the N=500 dip**: τ=800 is better there. |
| **P1.2** fair recalibration ceiling | done, **twice** | See §0.1 — the first fix was insufficient and the review's original objection was right for a reason neither the review nor the first fix identified. |
| **P1.3** fix the statistics | done | `stats.paired_by_unit`; sessions are the unit, seeds averaged within, BH across the 21-test grid. 3-seed t-CIs dropped from tables. |
| **P1.4** resolve the conduction anchor | done | Removed from CADENCE (it was an exact identity op on every run). Conduction survives as the decomposition instrument. |
| **P1.5** discriminating stability metric | done | `adaptation_regret` replaces collapse-rate. It separates cleanly where collapse-rate could not. |
| **P2.11** cite the prior art | done | BACKGROUND §10 concedes the estimator to empirical Bayes / AdaBN / Euclidean alignment and states the narrow claim. |
| **P2.12** second stream | **not done, documented** | `--subject loco` is wired through every Indy script, but the loco set is ~12 GB and this connection sustains 0.26 MB/s (~6 h). Every stream result remains single-subject and says so. |
| **P2.13** repo narrative | done | README, PAPER_OUTLINE, ROADMAP rebuilt around the decomposition spine. |

### 0.1 The review got P1.2 half-right, and the first fix was not enough

The review said the ceiling was a strawman because it trained **from scratch**. Replacing it
with a fine-tune barely moved it (0.125 → 0.129), which looked like vindication of the
original claim. It was not. The real handicap was that the ceiling was fed inputs z-scored
with the **source** session's statistics — the exact handicap the alignment baselines exist
to remove. Measured on three sessions, same data and same schedule:

| | frozen | fine-tune, source-normalised | fine-tune, own-normalised | scratch, own-normalised |
| --- | --- | --- | --- | --- |
| mean velocity R² | 0.511 | **0.024** | **0.741** | **0.764** |

So **per-session recalibration wins this stream outright**, and the claim "frozen + cheap
adaptation beats per-session retraining" is dead in both its forms. Label-free adaptation
buys cheapness, not accuracy. `recalibrate()` now re-normalises with the session's own
statistics and the docstring records the measurement.

A related number worth keeping: a fresh per-session decoder scores **0.765** on a
chronological split and **0.952** on a random one. Within-session non-stationarity is real
and worth reporting — but it is a 0.19 effect, not the 0.6 the broken ceiling implied.

### 0.2 What the follow-up work found (after the review's items were closed)

Acting on the three "worth your attention" items turned up four more things, two of which
change claims the review itself had accepted.

1. **The same normalisation bug was in two more ceilings.** `run_transfer_modes.py` and
   `run_xsession.py` both trained their `full-retrain` row on source-normalised inputs.
   Fixed via a shared `train.own_normalize`. The decomposition spine is unaffected (it uses
   `zero-shot − no-norm`, never a retrain row), but §II.8.1's "zero-shot beats retraining"
   is downstream of it and was re-measured.

2. **The collapse controller is not dormant, and it is what produces the zero-regret
   result.** `cadence.py` claimed the controller never fires for the closed-form head. It
   fires on ~3 of 10 real stream visits, precisely on the visits where the adapter would
   otherwise have lost — reverting to identity turns a loss into an exact tie. CADENCE's
   only surviving advantage was therefore mis-attributed to the shrinkage.

3. **Hyperparameter-free empirical Bayes fails.** Replacing the hand-set τ with a proper
   Efron–Morris estimator makes things dramatically worse (−0.400 R² at N=25, 0/8 sessions)
   and collapses the method onto the plain standardiser.

4. **…because the mechanism in the paper was wrong.** Same estimator, same N, different
   draw: MPA on 25 *random* windows scores **0.520**; on the first 25 windows, **0.027**
   (+0.493, p=0.003, 7/8 sessions). The online failure is **chronological bias**, not
   sampling noise. This explains (3) — EB models variance and is blind to bias — and it
   demotes the shrinkage from "principled denoiser" to "conservative hedge that happens to
   work." It also shows shrinkage is not the best fix: 25 well-spread windows beat CADENCE
   on 2000 consecutive ones.

Finding 4 is a better contribution than the one it replaces, and it generalises past BCI:
*online calibration data is not a random sample of the distribution it calibrates for*, and
the standard "vary N and watch the curve" diagnostic cannot detect that, because it
confounds sample size with sample position.

---

## The original review follows unchanged.


Adversarial review of the Phase-11 submission (NeurIPS 2026 Workshop, *Towards Test-Time
Continual Learning Agents*), turned into a work plan. Every finding below was verified
against the repo's own code and `results/*.json`; the two headline checks were re-run
from scratch on the real Indy data.

**Verdict as it stands: reject.** Not for sloppiness — the artifact is above workshop
median — but because the headline claim is contradicted by `results/indy_stream.json`,
and the margin producing it is largely a missing scale floor in the baseline.

**Verdict after P0: weak accept.** After the reframe in §5: a good workshop paper.

_Last updated: 2026-07-23._

---

## 1. The one-paragraph summary

CADENCE, as evaluated, is `MPA + w = n/(n+τ) shrinkage + a std floor`. The conduction
anchor never runs in the headline experiments and the collapse controller never fires.
That residual mechanism does produce a real, reproducible effect — but it is **+0.03 to
+0.06 R² over not adapting**, not the +0.65 the abstract implies, and in the paper's own
stream table the nearest baseline (MPA) beats it outright. The path forward is not to
defend the current claim; it is to make the drift decomposition the spine and the
shrinkage result the practical payoff. That paper is defensible end-to-end and we already
have the data for it.

---

## 2. Verified findings

Everything here is reproducible. The two re-runs are described in §6.

### 2.1 MPA strictly dominates CADENCE in our own stream table

From `results/indy_stream.json`, all three seeds:

| | cumulative R² | collapse-rate | BWT | worst-session |
| --- | --- | --- | --- | --- |
| MPA | **0.563 [0.55, 0.58]** | 0.100 | 0.000 | −0.100 |
| CADENCE | 0.540 [0.52, 0.56] | 0.100 | 0.000 | −0.031 |

Identical collapse-rate to 15 decimal places, identical BWT, lower accuracy. There is no
axis on which CADENCE wins §9.2, yet `RESULTS.md:402` bolds CADENCE and the prose never
says this. The auto-verdict prints "PARETO WIN" because its condition is
`cadence_collapse <= noadapt_collapse` (`scripts/run_indy_stream.py:215`) — satisfied by
any method that does nothing.

### 2.2 A third of the §9.1 margin is a missing scale floor

`MPA` computes scale as `X.std((0,2)) + 1e-6` (`b2ss/ibci_baselines.py:38`) — no floor.
CADENCE gets `std_floor=0.1` (`b2ss/cadence.py:128`). At N=25 windows on the 96-electrode
array, **14.5% of channels have sd < 0.1**. MPA divides by ~0 and diverges.

Re-run of the §9.1 curve with one line changed (same floor CADENCE already has), 3 seeds
× 8 targets:

| N windows | 25 | 50 | 100 | 200 | 500 | 2000 |
| --- | --- | --- | --- | --- | --- | --- |
| no-adapt | +0.409 | +0.409 | +0.409 | +0.409 | +0.409 | +0.409 |
| mpa (as shipped) | −0.217 | −0.074 | −0.004 | +0.149 | +0.250 | +0.510 |
| **mpa-floor** | **+0.027** | **+0.049** | **+0.123** | **+0.233** | **+0.258** | **+0.518** |
| cadence | +0.437 | +0.455 | +0.464 | +0.455 | +0.396 | +0.515 |

The floor alone recovers 0.244 of the 0.654 N=25 margin. **The remaining +0.41 is real** —
the shrinkage does carry weight beyond the floor. But "+0.65 over MPA" is not defensible.

### 2.3 Against the only non-broken comparator the effect is +0.03

Paired per-session test (seed-averaged, 8 sessions as the unit — the analysis §9.1 should
have run instead of pooling 24 non-independent values):

| N | CADENCE − no-adapt | p | sessions won |
| --- | --- | --- | --- |
| 25 | **+0.029** | 0.001 | 8/8 |
| 50 | +0.047 | 0.008 | 7/8 |
| 100 | +0.056 | 0.038 | 7/8 |
| 200 | +0.047 | 0.140 | 7/8 |
| 500 | **−0.013** | 0.777 | 5/8 |
| 2000 | +0.106 | 0.004 | 8/8 |

The effect is **consistent and significant but an order of magnitude smaller than
advertised**, and the curve is non-monotonic — CADENCE falls *below* no-adapt at N=500,
unexplained and unmentioned. Note the direction of both errors: the pooled-CI analysis
understates the reliability while the abstract overstates the magnitude.

### 2.4 "Full calibration (N=2000)" is off by an order of magnitude

Indy sessions carry **~15,000 training windows (~300 s)**; N=2000 is 13% of available
calibration data (~40 s). §9.3's central concession — "ties MPA at full calibration" —
describes a regime never tested. The trend runs against us: over N=500→2000, MPA climbs
0.250→0.510 while CADENCE is flat 0.396→0.515. A reviewer who runs N=15000 likely converts
the conceded tie into a loss. **This is the highest-risk unrun experiment in the project.**

### 2.5 The pre-registered make-or-break ablation is confounded

`free_lora` is `CADENCE(head='lora')` (`b2ss/tta_baselines.py:121`). Inside `adapt`,
`head='affine'` takes the **closed-form** branch and `head='lora'` takes the **Adam**
branch (`b2ss/cadence.py:157-178`). So "structure vs dense at matched parameter count" is
entangled with "closed-form vs gradient descent." As stated, §9.2 claim 2 does not follow.

The clean comparison is already in our data and supports us: **Tent** (gradient, diagonal,
2C params, same objective/lr/steps) vs **free-LoRA** (gradient, dense rank-1, 2C params) —
identical in everything but structure. At N=25: 0.306 vs −0.579.

### 2.6 Collapse-rate does not discriminate

no-adapt, MPA, CADENCE, CoTTA and RDumb all score exactly 0.100 across all three seeds —
one hard session in ten visits, hard for everyone. CADENCE's worst-session R² is
byte-identical to No-Adapt's, meaning on the session that matters the adapter contributes
nothing. A metric that assigns the same value to the method and to the null method cannot
support a collapse-resistance contribution; it only separates methods that go to −1.0,
which needs no metric.

### 2.7 Two of three advertised components are inert in the headline

`set_anchor` / `ema_anchor` are never called in `run_indy_stream.py` or
`run_indy_calibration.py` — `aligner.delta` stays zero, so the conduction anchor is an
identity op. The collapse controller never fires for the affine head; our own self-check
comment says so (`b2ss/cadence.py:252-253`). The module docstring's three-stage
architecture describes a system that is not the one producing the numbers.

### 2.8 Statistics

- 3-seed Student-t CIs yield **Tent collapse-rate 0.167 [−0.120, 0.454]** — a negative
  lower bound on a fraction, shipped in a results table.
- §9.1 pools 8 targets × 3 seeds as 24 iid draws (`scripts/run_indy_calibration.py:121`).
  Seeds are not independent samples; sessions are the unit. Pseudoreplication.
- No correction across 6 budgets × 5 methods, in a repo that ships Bonferroni/BH
  (`b2ss/stats.py`).

### 2.9 Unvalidated hyperparameters

`shrink_tau=200` sits mid-range of the tested budgets {25…2000} with no held-out selection
protocol. Same for `std_floor=0.1`, `collapse_z=3.0`, `anchor_ema=0.9`, `n_groups=8`.
Nothing in the repo refutes "tuned on the evaluation sessions."

### 2.10 The retraining ceiling is a strawman

`scripts/run_indy_stream.py:189` trains a **fresh** GRU per session (0.124 R²,
collapse-rate 0.875). Nobody deploys that. The realistic ceiling is fine-tuning the frozen
source decoder on the session — our own MC_Maze cross-session run reaches 0.759 that way.
§9.2 claim 3 will not survive a fair ceiling.

### 2.11 Residual circularity in the conduction results

`inject_group_latency` groups by `arange(C) % 8` (`b2ss/intracortical.py:165`);
`ConductionDelayAligner` groups by `arange(n_chan) % 8` (`b2ss/transfer.py:37`) — same map,
same K. §8.1's "structured beats free per-channel delays" hands the adapter the exact
generative structure *and* the negated ground-truth delay. "Injected and known" undersells
this; the grouping is also an oracle.

### 2.12 Novelty and repo narrative

`w = n/(n+τ)` shrinkage toward a prior is textbook empirical Bayes / James–Stein, and
shrinking normalization statistics toward source statistics is established in the TTA
literature. Neither is cited or compared against. Separately, `README.md` still leads with
the pre-pivot conduction-velocity decoder and `PAPER_OUTLINE.md` is two pivots stale
(still recommending the Phase-10 framing) — a reader of the artifact cannot tell what is
being claimed.

---

## 3. P0 — required to be publishable

Target: one week. Nothing below is optional; each item closes a finding that a reviewer
will find in the artifact unaided.

### P0.1 Floor the MPA baseline and re-run §9.1 — closes 2.2

Give MPA the same floor CADENCE has, and report both variants.

```python
# b2ss/ibci_baselines.py
class MPA:
    def __init__(self, decoder, src_stats_x, device="cpu", std_floor=0.1):
        ...
        self.std_floor = float(std_floor)      # sparse arrays: many near-silent channels

    def adapt(self, X, Y=None):
        mu, sd = source_input_stats(X)
        self.mu_t, self.sd_t = mu, np.maximum(sd, self.std_floor)
```

Report `mpa` and `mpa-floor` as separate rows. **Turn the liability into a contribution:**
"per-channel standardisers need a scale floor on sparse multi-electrode arrays — here is
the failure mode and its magnitude" is a small, true, useful finding, and stating it first
removes the reviewer's best attack.

_Acceptance:_ §9.1 table has both rows; the abstract's margin is quoted against
`mpa-floor`, not `mpa`.

### P0.2 Extend the curve to true full calibration — closes 2.4

Add `N = all available train windows` (~15k) to `--budgets`. Run it before writing
anything. If MPA wins at the real full-calibration point, print that. Highest-risk unrun
experiment; do it first so the framing is built on the outcome rather than adjusted to it.

_Acceptance:_ `results/indy_calibration.json` contains a full-session budget; §9.3's
concession describes the measured regime, not an extrapolation.

### P0.3 Re-frame the headline — closes 2.3

Replace "CADENCE beats every competitor" with what the data supports:

> Per-session standardisation is actively harmful below ~500 calibration windows.
> Shrinking each per-channel estimate toward the source prior makes it safe, recovering
> **+0.03–0.06 R² over not adapting** (8/8 sessions at N=25, p ≤ 0.04) and **+0.14–0.41
> over a correctly-floored standardiser**.

Every number in that sentence is verified. Lead §9.1 with the paired table from 2.3, not
with pooled CIs. Explain the N=500 dip or flag it as unexplained — do not leave it in a
table unmentioned.

_Acceptance:_ no claim of a margin larger than what a floored baseline yields; the paired
per-session analysis is the primary one.

### P0.4 State the §9.2 MPA result in the text — closes 2.1

Write it plainly: on the 1500-window stream, MPA (0.563) beats CADENCE (0.540) at
identical collapse-rate and BWT. This is consistent with our own curve — at n≈1500 the
shrinkage costs accuracy — so it costs nothing to say and everything to omit. Also fix the
auto-verdict in `run_indy_stream.py:215` so it cannot print "PARETO WIN" for a method that
is dominated on accuracy.

_Acceptance:_ §9.2 prose names MPA as the stronger method on that table; the verdict string
requires a strict accuracy win.

### P0.5 Swap the structure ablation to Tent vs free-LoRA — closes 2.5

Unconfounded, already run, supports the claim. Rewrite §9.2 claim 2 around Tent (0.306)
vs free-LoRA (−0.579): same objective, same optimizer, same lr and steps, same parameter
count, differing only diagonal vs dense. Keep CADENCE vs free-LoRA as a secondary
observation and label the closed-form/gradient difference explicitly.

_Acceptance:_ the pre-registered ablation is reported on a pair that differs in exactly one
factor.

---

## 4. P1 — required to be good

Target: one further week.

### P1.1 τ sweep with leave-one-session-out selection — closes 2.9

Sweep `shrink_tau ∈ {50, 100, 200, 400, 800}`, select on held-out sessions, report the full
sweep. Do the same for `std_floor`. This permanently removes the tuned-on-test objection,
and the sweep itself is informative — τ has a natural reading as "how many windows before
you trust the session over the prior."

### P1.2 Fair retrain ceiling — closes 2.10

Replace from-scratch per-session training with fine-tuning the frozen source decoder on
the session's data. Re-state or drop §9.2 claim 3 depending on the outcome. Keep
from-scratch as a labelled secondary row if it's informative.

### P1.3 Fix the statistics — closes 2.8

- Primary analysis: paired per-session tests, sessions as the unit, seeds averaged within
  session (a nuisance dimension, not a sample).
- Drop 3-seed t-CIs from tables, or report seed range instead of a t-interval.
- BH-correct across the budget × method grid using the existing `b2ss/stats.py` helpers.

### P1.4 Decide about the conduction anchor — closes 2.7

Two acceptable resolutions; pick one.

- **Wire it in.** Call `ema_anchor` in the stream and show what it does (probably ~0 on
  Indy, which we already predict). Then the architecture diagram is honest.
- **Cut it from the method.** Describe CADENCE as the shrinkage adapter, and keep
  conduction purely as the §9.4 diagnostic instrument.

Given §5, the second is the better paper. Shipping inert components in an architecture
figure is the failure mode reviewers punish hardest.

### P1.5 Replace collapse-rate with a metric that discriminates — closes 2.6

Proposed: **adaptation regret vs No-Adapt** — the fraction of sessions where adapting is
*worse* than not adapting, plus the mean shortfall on those sessions. On the existing data
this separates Tent, MPA and CADENCE cleanly, which the current floor-crossing metric does
not. Keep collapse-rate as a secondary column for the methods that genuinely diverge.

---

## 5. The reframe — make the decomposition the spine

This is the highest-leverage change in the document, and it needs no new experiments.

We have the only calibrated instrument in the literature for asking **which axis the
cross-session iBCI gap actually lies on**: a positive control that separates (+0.250
[+0.191, +0.309], injected MC_Maze) and two independent real-data nulls (−0.015
intracortical, −0.003 EEG), all in the same units. That is §9.4 — currently a
seven-line subsection.

Proposed spine:

> **How much of cross-session iBCI drift is conduction timing, and what actually helps?**
> A decomposition, validated by a positive control, showing conduction timing contributes
> essentially nothing to real multi-session drift — and the practical consequence: the
> drift that remains is per-channel gain/offset, whose per-session estimation is
> data-starved below ~500 windows and must be shrunk toward the source prior rather than
> estimated raw.

Why this survives every finding in §2:

| Finding | Status under the reframe |
| --- | --- |
| 2.1 MPA dominates the stream | No longer a claim we make; MPA becomes a reported comparator |
| 2.2 margin inflated by baseline bug | Becomes a stated finding about sparse-array standardisation |
| 2.3 effect is small | Correct scale for a "practical consequence," not a headline SOTA claim |
| 2.7 conduction anchor inert | The conduction term's job becomes diagnosis — which is what it does |
| 2.11 circular grouping | An oracle *positive control* is exactly right for validating an instrument |
| Phases 8–10 negatives | Become the evidence chain, not an apology |

Additional moves that fit the reframe:

- **Promote the honesty ledger into the paper** as "what we ruled out and how." Four
  cleanly-falsified hypotheses with mechanisms is workshop-native content.
- **Retire or explain the B2SS branding in one line.** A test-time-continual-learning
  reviewer does not care about g-ratios and will spend their first paragraph confused about
  what they are reading.

---

## 6. Reproducing the two checks in this document

Both re-runs used the repo as-is on the 11 downloaded Indy sessions; ~10 min CPU each.

**Fair-MPA curve (2.2).** Subclass `MPA` overriding `adapt` with
`np.maximum(sd, 0.1)`, drop it into `run_indy_calibration.py`'s method loop alongside the
shipped `MPA` and `CADENCE`, 3 seeds × 8 targets × the standard budgets. Also logs the
fraction of channels with `sd < 0.1` at N=25 (0.145).

**Paired per-session test (2.3).** Dump the raw per-`(seed, target)` R² lists, reshape to
`(3 seeds, 8 targets)`, average over seeds, and run `scipy.stats.ttest_rel` on the 8
per-session values. This is the analysis P1.3 makes primary.

Working scripts:
`/private/tmp/claude-504/-Users-buno-Documents-coding-saidlaboratory-B2SS/37ebc794-60a8-4a29-a7ce-a125e427cd1e/scratchpad/{mpa_fair.py,paired.py}`.
Fold them into `scripts/` when P0.1 and P1.3 land.

---

## 7. Claims we can currently defend

Safe to write today, in these words:

- Per-session per-channel standardisation **diverges** below ~100 calibration windows on a
  96-electrode array unless the scale is floored (14.5% of channels have sd < 0.1 at N=25).
- Shrinking per-channel calibration estimates toward the source prior by `n/(n+τ)` beats a
  correctly-floored standardiser by **+0.14 to +0.41 R²** for N ≤ 500, and by ~0 at N=2000.
- The same shrinkage beats not adapting at all by **+0.029 to +0.056 R²** for N ≤ 100
  (7–8/8 sessions, p ≤ 0.04), and by nothing at N=500.
- Matched-parameter **dense** unstructured adaptation collapses where matched-parameter
  **diagonal** adaptation does not, at identical objective/optimizer (Tent 0.306 vs
  free-LoRA −0.579 at N=25).
- Conduction/timing alignment recovers **+0.250 [+0.191, +0.309]** where timing dominates
  by construction, and **−0.015 / −0.003** on real intracortical / EEG cross-session gaps.

## 8. Claims to delete

- "CADENCE beats every competitor." (2.1, 2.2)
- "Margin over MPA +0.005 → +0.65." (2.2)
- "Ties MPA at full calibration." (2.4 — untested; likely false)
- "PARETO WIN" as an auto-verdict. (2.1, 2.6)
- "The interpretable structure — not the parameter budget — confers the collapse-resistance,"
  as currently evidenced. (2.5 — true claim, wrong evidence; re-evidence it)
- "Frozen + cheap adaptation beats per-session retraining." (2.10 — strawman ceiling)
- Any description of CADENCE as a three-component architecture while two components are
  inert. (2.7)
