# Design Spec — CADENCE: collapse-resistant structured test-time adaptation

_Date: 2026-07-23. Status: approved (brainstorming; PI confirmed the 4 strategic forks)._
_Team-facing plan: [`ROADMAP.md`](../../../ROADMAP.md) Phase 11. Prior pivot:
[`2026-07-06-b2ss-transfer-pivot-design.md`](2026-07-06-b2ss-transfer-pivot-design.md)._

## Context

Phase 10 (RESULTS.md §8) settled the conduction-normalisation question with two facts
that must both be respected:

1. On **controlled** data (real MC_Maze spikes, injected per-group conduction), zero-shot
   measured-CV alignment transfers to a new subject *better than retraining* — 0.649 vs
   0.403 R², +0.250 over naive transfer with zero target data.
2. On **real** cross-session gaps (MC_Maze S/M/L −0.015, MOABB Zhou2016 −0.003), conduction
   alignment does **nothing** — the real gap is unit turnover / firing-rate drift / tuning
   change, not timing.

A paper that only reports (1)+(2) is a characterization of a null. This spec turns the
project into a genuine, honest **victory** by changing the arena and the win axis, not by
overclaiming the conduction term.

## Target venue

NeurIPS 2026 Workshop "Towards Test-Time Continual Learning Agents". The exact CFP was not
locatable at design time; the timeline/format is calibrated to **ICLR-2026 Test-Time Updates
(TTU)** (organised by the TENT author) and **ICML-2026 Continual Adaptation at Scale (CATS)**
as first-class same-paper fallbacks. These venues reward source-free, label-free, online
adaptation under a compute/label budget, judged on **continual metrics and an accuracy-vs-cost
Pareto** — not a leaderboard R². "Agent" here is honestly read as *a decoder that keeps
re-normalising itself as its neural signal drifts across a lifelong stream* — a frozen
knowledge store plus an online sense/adapt/revert control loop.

## Thesis

> A structured, biophysically-bounded, source-free test-time adapter over a **frozen** neural
> decoder occupies a Pareto point no existing method reaches on a long recurring session
> stream — strictly above No-Adapt in cumulative and worst-session accuracy while strictly
> below free/entropy TTA in catastrophic-collapse rate, at 2–3 orders of magnitude fewer
> adapted parameters — and its **low degrees of freedom, not mere parameter count**, are what
> make it collapse-resistant.

The conduction/timing term is deliberately **not** the accuracy hero (it is null on real gaps;
this is conceded in the abstract). It earns its place in two load-bearing roles: the
biophysically-valid **safe revert-anchor** that makes collapse-sensing safe, and a
ground-truth-validated **drift-decomposition diagnostic**.

## Architecture (CADENCE)

```
session stream ─▶ [ fast repr. head ]─▶[ slow conduction anchor ]─▶ FROZEN decoder ─▶ output
                        │  CORAL/EA + NoMAD-style latent-moment match (borrowed)
                        │  ConductionDelayAligner δ∈R^K, EMA-consolidated (anchor)
                   [ collapse sensor ] ── on divergence: reset fast head → anchor state
```

- **Frozen backbone** — `baselines.GRUDecoder`, pretrained once on the earliest held-in Indy
  sessions, never updated. (Not the B2SS decoder, which is within-session-uncompetitive; the
  contribution is the adapter, so the backbone should be strong and boring.)
- **Slow anchor** — `transfer.ConductionDelayAligner` (K≈8–16 grouped fractional-shift delays,
  clamped), updated by heavy EMA / low LR so it consolidates only the slow timing component and
  cannot chase per-session noise. Interpretable, biophysically bounded, hard to drift.
- **Fast head** — borrowed representation aligner: Euclidean/CORAL covariance re-centering plus
  a NoMAD-style match of the frozen pre-head latent (captured via the `decoder.head` forward
  hook already in `transfer.py`), rank-1…8. Absorbs the fast turnover/tuning drift the anchor
  provably cannot. Total adapted DOF ≈ 16–32.
- **Controller** — collapse sensor on the unsupervised loss + latent-norm trajectory; past a
  threshold, reset the fast head toward the anchor's implied latent. Safe *because* the anchor
  is a low-DOF biophysically-valid state (reverting to raw source is not).
- **Fitting modes (one calibration-cost curve):** zero-shot (delays from measured/atlas CV),
  unsupervised (delay-sensitive cross-covariance objective — an upgrade over the current
  CORAL-moment version that fails at 0.302), few-shot (a handful of labeled trials).
- **Streaming protocol** — Indy sessions in temporal order with PeTTA-style revisits; adapter
  carried forward across sessions (no oracle reset); backbone frozen throughout, so forgetting
  is bounded by construction and the free-TTA collapse contrast is forced by stream length.

## Experiments

- **Headline — Indy recurring stream.** Monkey-Indy ~23-session self-paced grid reach
  (O'Doherty/Makin/Sabes, DANDI): the only public long+labeled stream where continual metrics
  compute, and the data MPA/NoMAD already report on. Pretrain+freeze on earliest sessions, adapt
  online across the rest with revisits. **Decisive figure = the accuracy×stability plane**
  (x = cumulative online velocity R²; y = collapse-rate + worst-session R²), plus a
  backward-transfer bar and an accuracy-vs-params/label Pareto.
- **Make-or-break ablation (pre-registered)** — free-LoRA head at *matched parameter count*.
  Survives at matched params ⇒ the win is the interpretable structure (novel); survives only at
  fewer params ⇒ it is the known "low-dim beats free" result, reported straight.
- **Supporting 1 — drift decomposition, timing bracket** (in hand): the injected-delay MC_Maze
  rig (zero-shot 0.649 vs retrain 0.403) validates the diagnostic against ground truth.
- **Supporting 2 — drift decomposition, representation bracket** (in hand): real MC_Maze S/M/L,
  timing null −0.015, reported straight.
- **Supporting 3 — EEG breadth** (secondary): MOABB Zhou2016 cross-session, frozen EEGNet, with
  EA/Riemannian + T-TIME baselines.

**Baselines:** No-Adapt, RDumb (credibility gate), Tent, CoTTA, EATA/SAR, RoTTA (re-derived for
the BN-free GRU regressor), MPA (nearest neighbour / scoop risk), NoMAD (strong iBCI stabiliser),
free-LoRA(matched), full per-session retrain (cost ceiling).

## Success criteria (honest go/no-go)

- CADENCE cumulative online R² > No-Adapt (separated CIs) **and** collapse-rate ≤ No-Adapt —
  genuine two-axis Pareto-dominance.
- CADENCE collapse-rate < every free-TTA baseline and < MPA; worst-session R² strictly higher
  (separated CIs) over the full stream.
- CADENCE > RDumb periodic reset on cumulative online R².
- Matched-param free-LoRA loses on collapse-rate/worst-session (structure carries stability). A
  tie narrows the contribution to protocol + decomposition — still a paper, known by Week 6.
- Decomposition brackets: CI-separated positive timing marginal on the injected rig; honestly-
  reported near-null on real cross-session.
- Concessions in the abstract: peak R² conceded to full-retrain; timing term ≈0 accuracy on
  turnover-dominated gaps (its value is consolidation/collapse-safety + diagnostics); the
  accuracy lift over No-Adapt is carried by the borrowed alignment head.

## Novelty positioning

Load-bearing novelty is deliberately off the null phase-accuracy claim and onto: (1) the **first
continual-stream evaluation protocol for iBCI TTA** reporting collapse-rate, backward-transfer/
forgetting, and worst-session R² (MPA/NoMAD/SPINT report only static per-session R²); (2) a
biophysically-**structured** low-DOF adapter as a collapse-resistant **safe revert target**,
isolated from CoTTA/RoTTA/PeTTA by the matched-param free-LoRA ablation; (3) a timing-vs-
representation **drift decomposition** validated against injected ground truth. Differentiate
MPA / NoMAD / T-TIME / DCLS / PeTTA / EcoTTA explicitly in one table.

## Risks

- **Indy stream too gradual to force free-TTA collapse** (biggest, out of our control) — amplify
  with revisits + batch=1/short-window stress; Week-4 go/no-go pivots to the fallback.
- **Novelty desk-reject** — answered by the protocol + matched-param ablation + up-front null.
- **Baseline fidelity is the critical path (~2.5 wk)** — never drop RDumb/Tent/MPA/free-LoRA;
  drop only the weakest 2 free-TTA variants if time slips.
- **Indy DANDI port is real work** — W1–2 tap-root with backbone validation before anything
  downstream; MC_Maze + MOABB fallbacks already load.

## Fallback (so 2 months is never wasted)

The **drift-decomposition benchmark** — ~80% in hand, needs no Indy port, no free-TTA collapse,
no baseline reimplementations. Contribution: a controlled+real suite partitioning each
cross-session gap into a timing/conduction component (ground-truthed on the injected rig) vs a
representation/tuning component (real MC_Maze + EEG nulls), plus the first continual-stream iBCI
metric harness — a calibrated plug-in probe telling a practitioner *which* drift they face before
they pay to recalibrate. Converts the honest null from a liability into the thesis.

## Reuse

`transfer.py`, `intracortical.py` (port target), `baselines.GRUDecoder`, the injected rig in
`run_transfer_modes.py`, the MOABB loader in `run_moabb_transfer.py`, `stats.py` (extend with
continual metrics). 21 tests currently green.
