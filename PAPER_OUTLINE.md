# Paper scoping — the decomposition paper

The framing to commit to before writing, and the outline if we proceed. Numbers live in
[RESULTS.md](RESULTS.md) and are not duplicated here; this document is the argument.

> **Rebuilt against [BETTER.md](BETTER.md)** — an adversarial review that found the previous
> framing ("CADENCE beats every competitor in the data-scarce regime") contradicted by our
> own `results/indy_stream.json`, and its headline margin inflated by a missing scale floor
> in the baseline. Every claim below is one that survived that review.

---

## 1. The question, and why it is the right one

**"How much of the cross-session drift in an intracortical BCI is conduction timing — and
what do you do about the rest?"**

This is a measurement paper with a practical consequence. It is not a SOTA paper and must
not be written as one: on the continual stream a plain per-session standardiser gets higher
cumulative R² than our adapter, and full per-session recalibration beats both by a wide
margin. We say both in the results section rather than letting a reviewer find them in the
JSON.

The question is worth asking because the field has two incompatible habits. Conduction
delay is treated as a real, measurable, biophysically-grounded quantity (g-ratio → velocity
→ latency) in the neuroscience literature; and cross-session BCI drift is treated as an
undifferentiated blob to be absorbed by whatever adapter is fashionable. Nobody has put a
number on how much of the second is the first. We have, with a positive control.

## 2. Contributions

1. **A calibrated decomposition of the cross-session gap.** A conduction/timing marginal
   measured in velocity-R² units, with (a) a positive control on real spikes where timing
   dominates by construction and the marginal separates from zero, and (b) two independent
   real-data nulls — intracortical multi-session and EEG cross-session. The control is what
   licenses the nulls: an instrument that reads zero everywhere measures nothing.
2. **Online calibration data is not a random sample of the session it calibrates for —
   and that, not sampling noise, is why test-time adaptation fails at small budgets.**
   Hold N fixed and change only how the windows are drawn: a per-session standardiser
   scores 0.027 on the first 25 windows of a session and **0.520 on 25 random ones**
   (+0.493, p=0.003, 7/8 sessions). Twenty-five well-spread windows beat two thousand
   consecutive ones. Three things follow: the standard "vary N and watch the curve"
   diagnostic cannot see this (it confounds sample size with sample position); a
   principled variance-based correction is structurally blind to it (our own textbook
   empirical-Bayes version collapses onto the baseline, −0.400 R² at N=25); and the
   effective fix is temporal spread rather than a better estimator. This is the
   contribution that generalises past BCI.

3. **A failure mode of the standard per-session standardiser.** On a sparse 96-electrode
   array, ~15% of channels are near-silent over a 25-window calibration slice, and dividing
   by that scale does not degrade gracefully — it **diverges to negative R²**. A scale floor
   stops the divergence; conservative shrinkage (`w = n/(n+τ)`) additionally hedges the bias
   in (2) and recovers most of the gap. We report the failure, the floor-only fix, and the
   shrinkage fix as separate rows, and we say plainly that the shrinkage works for a
   different reason than its derivation claims.
4. **A continual-stream iBCI protocol with a stability metric that discriminates.** Streaming
   sessions in temporal order with revisits, scored on cumulative/worst-session R², backward
   transfer, and **regret vs No-Adapt** (how often, and how badly, adapting loses). We adopted
   regret after finding that a fixed R²-floor collapse-rate assigns the identical value to our
   method and to doing nothing — a metric that cannot lose is not evidence.
5. **A negative result chain worth citing.** Four falsified hypotheses with mechanisms, not
   just outcomes: within-subject CV gating (a decoder learns fixed structural delays from
   data), delay alignment (same reason), real cross-session conduction normalisation (the gap
   is not timing), EEG conduction alignment (ditto). §7.

## 3. Positioning — what is ours and what is borrowed

State this in the paper, in these words, before a reviewer does. See
[BACKGROUND.md §10](BACKGROUND.md) for the citations.

- **The estimator is not new.** `w = n/(n+τ)` shrinkage toward a prior is textbook empirical
  Bayes / James–Stein. Interpolating normalization statistics between source and target is
  established in TTA (AdaBN/PTBN family; the α-blend of Schneider et al. 2020). Per-session
  input re-centering is standard BCI transfer (Euclidean/Riemannian alignment). We use all
  three off the shelf.
- **The measurement is ours**, and so is the observation that the standard estimator
  *diverges* rather than degrades in this regime, and the protocol that makes it visible.
- **We cannot claim the principled pedigree either.** The textbook empirical-Bayes version
  of our own estimator fails on this data (§I.3), because it models variance and the problem
  is bias. The fixed τ is a conservative hedge that works for a reason its derivation does
  not give. Say so; a reviewer who derives the "correct" version will find the same thing.
- **vs learnable-delay models (DCLS/SNN)**: they learn per-connection delays from data; we
  measure whether delay is where the cross-session gap lives at all, and answer no.
- **vs LFADS / NoMAD / MPA**: decoder-side stabilisers. NoMAD and MPA are our baselines, not
  our contrast class — MPA wins the stream and that is reported.

## 4. Outline & experiment mapping

| Section | Content | Evidence |
| --- | --- | --- |
| Intro | cross-session drift as an undifferentiated blob; the two habits | BACKGROUND §1–2, §5 |
| The instrument | conduction alignment as a measurement, not an adapter; grouping + delay model | `b2ss/transfer.py` |
| Positive control | injected per-group latency on real MC_Maze spikes; marginal separates | `run_transfer_modes.py` |
| Null I | real multi-session intracortical (MC_Maze S/M/L, per-electrode) | `run_xsession.py` |
| Null II | EEG cross-session (Zhou2016) | `run_moabb_transfer.py` |
| **The decomposition figure** | all three in the same units — the paper's centre | `run_decomposition_figure.py` |
| Consequence | per-channel gain/offset drift; the standardiser's divergence; shrinkage | `run_indy_calibration.py` |
| **Why it fails** | same N, different draw — bias not noise; the general lesson | `run_calibration_bias.py` |
| Hyperparameters | τ sweep + LOSO selection; the empirical-Bayes negative | `run_tau_sweep.py` |
| Continual stream | protocol, regret metric, recalibration ceilings, structure ablation | `run_indy_stream.py` |
| What we ruled out | the four falsified hypotheses and why each failed | RESULTS §1–8 |
| Limitations | §5 | — |

## 5. Limitations — stated in the paper, not the rebuttal

- **Per-session recalibration beats every label-free adapter here, and it is not close.**
  Given the session's own input normalisation, a recalibrated decoder reaches ~0.75 velocity
  R² against the best adapter's 0.575. What label-free adaptation buys is **cheapness** —
  2·n_chan parameters, no labels, no training — at a real accuracy cost. Any framing that
  implies otherwise is wrong; an earlier version of our own harness implied otherwise because
  it fed the ceiling source-normalised inputs, and that error is documented rather than
  quietly fixed.
- **One subject.** Every stream result is monkey Indy, one 96-electrode array, 11 sessions
  over ~1 month. The same Zenodo record carries a second monkey (`loco`) on the same rig, and
  the code takes `--subject loco`, but we did not run it (~12 GB). A single-subject result is
  a single-subject result.
- **Our adapter does not win the stream.** With ample calibration data a plain floored
  standardiser gets higher cumulative R². The shrinkage helps in the data-scarce regime and
  costs accuracy outside it. Both ends of the curve are reported.
- **The positive control is an oracle in two ways**, not one. The conduction difference is
  injected and known, *and* the aligner is given the same channel→group map used to generate
  it. It is an upper bound on what perfect conduction knowledge could buy — which is the
  right thing for a control and the wrong thing to quote as a result.
- **No measured CV exists in any dataset used here.** The decisive test of the original
  hypothesis still needs an acquisition pairing a decode task with a per-subject CV;
  BACKGROUND §9 documents that no public dataset does.
- **The estimator is standard.** See §3. The novelty claim is narrow and deliberately so.
- Small N elsewhere: EEG 8 subjects; single MC_Maze sessions in the bracket experiments.

## 6. Venue and go/no-go

**Recommendation: submit, as a measurement paper, to a test-time/continual-adaptation
workshop.** The decomposition is workshop-native (it tells the room something about their
problem that they did not know), the negative chain is the kind of content workshops
reward, and the artifact reproduces end-to-end.

**Do not** submit it as a method paper. The method is one line of empirical Bayes, it loses
the stream to its nearest baseline, and a reviewer will find both facts in ten minutes.
Framing it as SOTA is the one thing that turns a defensible workshop paper into a reject.

**The single highest-value follow-up** is the one this repo cannot do: a cohort with both a
decode task and a measured per-subject conduction velocity. Everything here says the
cross-session gap is representation drift rather than timing — but "timing does not explain
the drift between two days of the same monkey" is not the same claim as "conduction velocity
is useless as a decoder prior," and only measured-CV data separates them.
