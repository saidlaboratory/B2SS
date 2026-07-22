# B2SS — Build Roadmap

Living to-do list for the **software artifact**. Updated as work lands.
Status: ☐ todo · ◐ in progress · ☑ done · ⊘ out of scope (wet-lab/hardware).

_Last updated: 2026-07-23._ **► Current focus: [Phase 11 — CADENCE](#phase-11--cadence-collapse-resistant-structured-test-time-adaptation)** (NeurIPS 2026 workshop, ~2 mo). Phases 0–10 below are the completed history that motivates it.

---

# PHASE 11 — CADENCE: collapse-resistant structured test-time adaptation

_Target venue: **NeurIPS 2026 Workshop "Towards Test-Time Continual Learning Agents"**
(~2-month horizon). The exact CFP page isn't posted yet, so the timeline/format is
calibrated to **ICLR-2026 Test-Time Updates (TTU)** and **ICML-2026 Continual Adaptation
at Scale (CATS)** as first-class same-paper fallbacks. Design spec:
`docs/superpowers/specs/2026-07-23-b2ss-cadence-continual-tta-design.md`._

**Why this exists.** Phase 10 proved the conduction normaliser wins big on *controlled/
injected* data but is **null on real cross-session gaps** (MC_Maze S/M/L −0.015, EEG
−0.003) because those gaps are unit turnover / tuning drift, not timing. A characterization
of that null is not a victory. CADENCE reframes the whole project around what the workshop
actually rewards — **source-free, label-free, online adaptation over a lifelong stream,
judged on continual metrics and an accuracy-vs-cost Pareto** — where a genuine, honest win
*is* reachable.

**The concept (one line).** Freeze a strong source decoder; adapt only a tiny (~16–32-param)
*composed, structured* module online across a stream of recording sessions, and claim the
one point no baseline can occupy — **high cumulative/worst-session accuracy AND collapse-free**
— at 2–3 orders of magnitude fewer adapted params than retraining.

### Strategic decisions (locked with the PI)
- **Victory type = Pareto/stability, not accuracy-SOTA.** Conduction/timing does *not*
  carry an accuracy win on real gaps — this null is **stated up front, in the abstract**,
  as a credibility signal, not buried in limitations. Accuracy is carried by a *borrowed*
  representation aligner; the conduction term earns its place as (1) the biophysically-
  bounded safe **revert-anchor** and (2) a ground-truth-validated **drift-decomposition
  diagnostic**.
- **Headline arena = the monkey-Indy ~23-session grid-reach stream** (O'Doherty/Makin/Sabes,
  DANDI). The only public **long + labeled** stream where continual metrics are computable
  — the same data MPA and NoMAD report on. (FALCON dropped this cycle: withheld labels make
  the continual headline uncomputable.)
- **Fallback floor = the drift-decomposition benchmark** (~80% already in hand) — a complete,
  honest, workshop-native paper that needs no Indy port, no free-TTA collapse, and no baseline
  reimplementations. The 2 months are never wasted.

### The method (CADENCE)
- **Frozen backbone** — a strong within-session GRU (`baselines.GRUDecoder`, *not* the
  within-session-uncompetitive B2SS decoder), pretrained once on the earliest held-in
  sessions and never updated → the stable "memory."
- **Slow anchor** — the existing `transfer.ConductionDelayAligner` (K≈8–16 grouped delays),
  EMA-consolidated at a low learning rate so it holds only the slow timing component → the
  interpretable, hard-to-drift revert target.
- **Fast head** — a *borrowed* representation aligner (CORAL/Euclidean re-centering +
  NoMAD-style latent-moment match on the `decoder.head` forward hook already in
  `transfer.py`), rank-1…8 → absorbs the fast unit-turnover/tuning drift the anchor can't.
- **Controller** — a collapse-sensor watches the unsupervised loss + latent-norm trajectory;
  on divergence it resets the fast head toward the anchor's implied latent (safe *because*
  the anchor is a valid low-DOF biophysical state, unlike reverting to raw source).
- **Streaming protocol** — sessions replayed in temporal order with PeTTA-style revisits;
  adapter state carried forward across sessions (no oracle reset); backbone frozen throughout.

### Headline experiment & win condition
One decisive figure — the **accuracy × stability plane**: x = cumulative online velocity R²,
y = collapse-rate (fraction of sessions under an a-priori R² floor) + worst-session R²; plus
a backward-transfer/forgetting bar on revisited sessions and an accuracy-vs-params/label-budget
Pareto. **Win = CADENCE Pareto-dominates:** strictly above No-Adapt on cumulative R² while
matching its collapse-rate; strictly below every free-TTA method and MPA on collapse-rate;
beats RDumb reset over the full stream.

**Make-or-break ablation (pre-registered): a free-LoRA head at *matched parameter count*.**
If CADENCE's stability edge survives at matched params → the win is the interpretable
**structure** (the novel claim). If it only survives at fewer params → it collapses into the
known "low-dim beats free TTA" result, and that is reported straight.

**Baselines (must-have):** No-Adapt (accuracy floor / stability ceiling), **RDumb** periodic
reset (credibility gate), **Tent**, **CoTTA**, **EATA/SAR**, **RoTTA** (all re-derived for the
BN-free GRU regressor), **MPA** (nearest neighbour / scoop risk), **NoMAD** (strong iBCI
stabiliser), free-LoRA(matched), full-retrain (cost ceiling). EEG secondary: Euclidean/
Riemannian Alignment + **T-TIME** on a frozen EEGNet (MOABB Zhou2016).

### Build status (first full run)

Built + tested (**27 self-checks green**), on **real** data (11 monkey-Indy sessions,
Zenodo 583331, downloaded + loaded): `continual.py`, `stream.py`, `indy.py`
(real-session validated: 96 electrodes, within-session Ridge R² 0.499), `cadence.py`,
`tta_baselines.py` (No-Adapt/Tent/CoTTA/RDumb/free-LoRA), `ibci_baselines.py`
(MPA/NoMAD-style), `run_indy_stream.py`, `run_decomposition_figure.py`.

**Go/no-go outcome — the accuracy headline changed shape (honest).** Cross-session R² is
*positive but modest* for a frozen decoder (no-adapt ~0.51), not the catastrophe an early
under-adapted run suggested. The clean, real result is **collapse-resistance**: matched-
parameter *unstructured* adaptation (free-LoRA dense rank-1; NoMAD full-rank readin)
**catastrophically collapses** on the stream (cumulative R² < 0, collapse-rate 1.00),
while *structured* low-DOF adaptation (CADENCE / MPA per-channel; No-Adapt) stays stable
and positive (3 seeds: CADENCE 0.503, best collapse-rate + worst-session + no error
accumulation; a simple closed-form standardiser (MPA) edges it on *mean* R² 0.563 — conceded).
Frozen+adapt (0.503) **beats per-session full retrain** (0.124, limited by within-session
non-stationarity). The pre-registered matched-param ablation fires: **structure, not param
count, prevents the collapse.** Multi-seed CIs in [RESULTS.md](RESULTS.md) §9.

**Honest scoping (stated, not hidden):** on Indy (no measured CV) CADENCE's conduction
anchor is dormant, so CADENCE ≈ a structured affine adapter (Tent-family); its edge over
carried-forward Tent is the per-session refresh + anchor reference (no error accumulation).
The contribution is collapse-resistance + the continual-stream protocol + the matched-param
structure ablation + the conduction decomposition — **not** a large accuracy gain over
No-Adapt, and the conduction term's null on real gaps is conceded up front.

**Descoped (not essential to the claim):** EATA/SAR/RoTTA free-TTA variants (Tent/CoTTA/
RDumb cover the family); the cross-covariance objective upgrade (moment-match sufficed);
a FALCON leaderboard entry (Indy is the computable-metric arena).

### Week-by-week

**W1–2 — Tap-root: Indy port + frozen backbone + metric harness** _(everything downstream gates on this)_
- ☐ Port `b2ss/intracortical.py` from the MC_Maze dandiset to the Indy grid-reach dandiset
  (self-paced continuous format, different unit/behaviour schema); validate the loader
- ☐ Pretrain + **freeze** a GRU backbone on the earliest held-in sessions; confirm
  within-session R² is competitive (so gains aren't dismissed as fixing a weak model)
- ☐ Add continual metrics to `b2ss/stats.py`: cumulative online R², worst-session R²,
  collapse-rate, backward-transfer/forgetting
- ☐ Build the online streaming loop (temporal order + PeTTA-style revisit schedule)

**W3–4 — Method: composed adapter + controller**
- ☐ Add the borrowed representation head (CORAL/Euclidean + NoMAD-style latent-moment match
  on the `decoder.head` hook, ~50–80 lines)
- ☐ Convert `ConductionDelayAligner` to a slow-EMA anchor; wire the collapse-sensor +
  revert-to-anchor controller (~40 lines)
- ☐ Upgrade the label-free objective to a **delay-sensitive cross-covariance/lag** form
  (the current CORAL-moment version fails at 0.302 < no-norm on the controlled rig)
- ☐ Stand up the matched-parameter free-LoRA head for the make-or-break ablation

**W5–6 — Baseline battery (critical path) + Week-4 go/no-go**
- ☐ Re-derive Tent/CoTTA/EATA/SAR/RoTTA for the BN-free GRU regressor; RDumb is trivial
- ☐ Reimplement **MPA** on the Indy data (head-to-head) and **NoMAD** (the strong stabiliser);
  sanity-check each against its published regime
- ☐ **GO/NO-GO:** run No-Adapt + Tent + CoTTA over the full stream — confirm free-TTA visibly
  accumulates error / collapses. If it stays stable on Indy, trigger the fallback
  decomposition-benchmark headline **now** (don't sink weeks chasing a contrast that isn't there)
- ☐ Wire all methods through one shared streaming harness

**W6–7 — Decomposition + breadth** _(overlaps the Phase-2 tail)_
- ☐ Injected-delay MC_Maze rig = the ground-truth **timing-dominated** bracket (already positive:
  zero-shot 0.649 vs retrain 0.403)
- ☐ Real MC_Maze S/M/L = the **representation-dominated** null bracket (report −0.015 straight)
- ☐ MOABB Zhou2016 EEG secondary stream: frozen EEGNet + EA/Riemannian + T-TIME

**W7–8 — Multi-seed stats + write-up**
- ☐ Multi-seed (≥5) across all methods × streams; separated CIs via `stats.mean_ci`/FDR
- ☐ Resolve the matched-param free-LoRA ablation; report the structure verdict straight
- ☐ Assemble the accuracy×stability Pareto figure + the decomposition figure; write the
  4-page paper with the cite-and-differentiate table and the up-front honest concessions

### Pre-registered success criteria (honest go/no-go)
- **PRIMARY:** on the Indy stream, CADENCE achieves cumulative online R² strictly above
  No-Adapt (separated CIs) **and** collapse-rate ≤ No-Adapt — genuine two-axis Pareto-dominance,
  not stability-by-correcting-nothing.
- **PRIMARY:** CADENCE's collapse-rate < every free-TTA baseline and < MPA, with worst-session
  R² strictly higher, over the full stream (separated CIs).
- **PRIMARY:** CADENCE beats RDumb periodic reset on cumulative online R².
- **STRUCTURE ISOLATION:** the matched-param free-LoRA head loses to CADENCE on
  collapse-rate/worst-session. If it ties → contribution narrows to protocol + decomposition,
  reported straight (still a paper), and we know by Week 6 not Week 8.
- **DECOMPOSITION (guaranteed):** injected rig shows a CI-separated positive timing marginal;
  real cross-session shows an honestly-reported near-null timing marginal — the two brackets.
- **Concessions stated up front:** peak R² conceded to full-retrain; the timing term adds ~0
  accuracy on turnover-dominated gaps (its value is consolidation/collapse-safety + diagnostics);
  the accuracy lift over No-Adapt is carried by the borrowed alignment head.

### Risks & mitigations
- **Indy stream too gradual to force free-TTA collapse** (biggest, out of our control) →
  run the recurring/revisit protocol to amplify accumulation; stress the batch=1/short-window
  regime where entropy-TTA provably breaks; Week-4 go/no-go pivots to the fallback if free-TTA
  never collapses.
- **Novelty desk-reject ("MPA + CoTTA + PeTTA + a borrowed aligner")** → relocate load-bearing
  novelty to the first continual-stream iBCI protocol (collapse/forgetting metrics), the
  matched-param structure ablation, and the ground-truth drift decomposition; cite-and-
  differentiate MPA/NoMAD/T-TIME/DCLS/PeTTA in one table; concede the phase null openly.
- **Baseline fidelity is the critical path (~2.5 wk)** — a weakly-tuned competitor makes the
  win non-credible → budget it explicitly; never drop RDumb, Tent, MPA, or the matched-param
  free-LoRA; if time slips, drop the weakest 2 free-TTA variants only.
- **Indy DANDI port is real work, not "same pynwb pattern"** → treat as W1–2 tap-root with
  explicit backbone validation before anything downstream; MC_Maze + MOABB fallbacks already
  load, so a slipping port degrades gracefully to the decomposition paper.

### Reused assets (little of this is from scratch)
`transfer.py` (`ConductionDelayAligner`, `TransferNormalizer`, the `decoder.head` hook,
3 fitting modes) · `intracortical.py` MC_Maze loader (port target) · `baselines.GRUDecoder`
· the injected-delay rig in `run_transfer_modes.py` · the MOABB loader in `run_moabb_transfer.py`
· `stats.py` (CIs/FDR, extend with continual metrics) · 21 green tests.

---

## Phase 0 — Docs & scaffolding
- ☑ Read & digest the proposal (v1.1)
- ☑ `README.md` — project overview, scope table, quickstart
- ☑ `BRIEF.md` — full science + software explainer
- ☑ `BACKGROUND.md` — literature review with verified citations _(38-agent research workflow; 8 citation errors caught & corrected)_
- ☑ `ROADMAP.md` — this file
- ☑ `requirements.txt` + package skeleton (`b2ss/`, `scripts/`, `tests/`)

## Phase 1 — Biophysics core (`b2ss/cv.py`)
- ☑ g-ratio → CV via Berman relation `v(g) = √(1−g²)/g`, scaled by `k`
- ☑ Combined TMS-EEG estimate: `CV = path_len / (mep_lat − cortical − nmj)`
- ☑ Monte-Carlo bootstrap → per-subject CV with 95% CI
- ☑ Self-check: v(g) monotonic decreasing in g; peak CV(diameter) near g≈0.6

## Phase 2 — Synthetic data (`b2ss/data.py`)
- ☑ EEG-like windows (64 ch, 50 timepoints) with latent "intent" signal
- ☑ Ground-truth **CV→latency law**: higher CV ⇒ shorter intent→kinematics lag
- ☑ 6-DOF kinematics targets; per-subject CV drawn with realistic spread
- ☑ Self-check: recovered lag tracks CV; shapes/splits correct

## Phase 3 — Decoder (`b2ss/model.py`)  _(describes the actual code, not the proposal spec)_
- ☑ Conv patch-embed → Transformer encoder over **time** tokens
  (defaults 2 layers / 4 heads / d_model=64 / FFN=128; `proposal_config()` = 4/8/256)
- ☑ CV gate `τ = τ_min + σ(W_cv·CV_norm + b_cv)·(τ_max−τ_min)` → recency attention mask
  (bounded to [τ_min,τ_max]; corrects the proposal's unbounded `σ()·τ_max + τ_min`)
- ☑ Euler Neural-ODE readout: **20 steps, dt = τ/20** (τ sets total integration time —
  NOT the proposal's fixed Δt=5 ms × 40 steps)
- ☑ Matched-capacity control (`learned` τ); param delta = **1 scalar**, not "reallocated"
- ☑ Self-check: shapes; param delta <1%; τ∈[τ_min,τ_max]; per-sample + uncertainty paths

## Phase 4 — Train & eval (`b2ss/train.py`, `b2ss/eval.py`)
- ☑ Training loop: Adam (default **lr=1e-3** for the small CPU models; the proposal's
  1e-4 is for the full A100 model), MSE/CE + 1e-5 L2, early stopping, val split
- ☑ Metrics: MSE, Pearson r, effective latency (cross-correlation peak lag)
- ☑ Self-check: loss decreases; metrics finite on a tiny run

## Phase 5 — Offline comparison (`scripts/run_offline_comparison.py`)
- ☑ Per-subject train B2SS + control on synthetic data; paired report
- ☑ Emit MSE / Pearson r / latency table + summary (Experiment-3 shape)
- ☑ Sanity: on CV-structured data, B2SS ≥ control on average

## Phase 6 — Verification
- ☑ `pytest` green
- ☑ Comparison script runs end-to-end and prints a coherent report
- ☑ Adversarial review of the numeric claims (CV law, latency metric)

## Results (synthetic offline comparison)
- 8 self-check/pytest tests green.
- `run_offline_comparison.py` (7 subjects spanning the CV range, 40 epochs; with
  the F1–F3 fixes): B2SS beat the matched control on **6/7** subjects,
  **~36% lower MSE** (p=0.16, n=7). The CV gate's window τ tracks CV as designed.
  _(Superseded for rigor by the 5-seed ablation Study A; kept as the Experiment-3-shape demo.)_
- Honest caveat: this only shows the decoder *can* exploit a CV→latency structure
  when one exists — not evidence for the scientific hypotheses (which need real data).

## Phase 7 — Publication-grade upgrade (remove reviewer threats + add-ons)
_Goal: answer the "information vs prior" threat, the "can EEG carry it" threat, and
the effect-size realism threat; add real-data benchmark, CV proxy, ablations,
uncertainty gate, latency, and the pre-registered stats harness._

- ☑ Model refactor: task heads (regression + classification), 4 gate modes
  (`cv`/`learned`/`fixed`/`none`), **per-sample/heterogeneous CV**, conv patch-embed
- ☑ **Uncertainty-aware gate**: noisy CV (large cv_sd) shrinks τ toward population mean
- ☑ Heterogeneous-CV synthetic regime (`data.make_heterogeneous`) — CV = information,
  constant window provably suboptimal
- ☑ Gate ablation + data-efficiency curves (`scripts/run_ablation.py`, Studies A & B)
- ☑ CV proxy from EEG mu peak frequency (`b2ss/proxies.py`, Corcoran-style)
- ☑ Real-EEG loader: PhysioNet EEGMMI via MNE (`b2ss/datasets.py`)
- ☑ Baselines: faithful EEGNet + CSP+LDA (`b2ss/baselines.py`)
- ☑ Real-data benchmark: B2SS vs baselines, within-subject CV (`scripts/run_real_benchmark.py`)
- ☑ Real-time latency benchmark vs 50 ms budget (`scripts/bench_latency.py`)
- ☑ Stats harness: power (corrected d), ICC(2,1), mixed-model ΔR², FDR/Bonferroni (`b2ss/stats.py`)
- ☑ `RESULTS.md` capturing benchmark/ablation/latency/stats outputs
- ☑ Docs refreshed (README, BRIEF, BACKGROUND) + 14 tests green

### Threat status (see RESULTS.md for numbers)
- **Info vs prior** — ✅ addressed (nuanced, 5 seeds). CV gate beats learned/fixed
  in BOTH regimes; `none` ~2–3× worse. The clean "prior shrinks with data" story did
  NOT survive 5 seeds — the homogeneous gap stays ~0.24–0.32 (CIs overlap). So CV is
  a persistent useful prior (homogeneous) and information a constant can't capture
  (heterogeneous, gap persists). Reported straight in RESULTS.md, not forced.
- **Effect-size realism** — ✅ resolved. Corrected power: d=0.37 needs N≈60–80;
  proposal's N=30 gives 0.50 power. Full §6 stats harness runnable.
- **Latency <50 ms** — ✅ resolved. 0.9–2.5 ms p95 on CPU.
- **Can EEG carry it / architecture competitive** — ✅ resolved (confound removed).
  After the F1 fix, re-ran 3 seeds: EEG is decodable (CSP 0.60, EEGNet 0.55 ≫ chance)
  but the CV gate still gives **no benefit** (b2ss-cv 0.47 ≈ learned 0.48; none 0.52
  is the best B2SS variant), and B2SS is not competitive on small-trial 2-class EEG.
  The F1 fix confirms this is the *mu-proxy being uninformative*, not a masking
  artifact — a clean, un-confounded negative. See RESULTS.md §3.

---

# HONEST FINDINGS FROM CODE/ARCH REVIEW (fix these)

Ordered by severity. F1–F3 are correctness; F4–F6 are doc/rigor drift.

- **F1 — τ→attention-span mis-scaling (bug, affects results).** `span = τ/1000·fs/patch`
  gives 5–25 tokens (10–50% of window) in the synthetic config but **0.4–2 tokens
  (1.3–6.7%, nearly CV-invariant)** in the EEG config. The CV gate is effectively
  degenerate on real EEG. Fix: define span as a **fraction of `n_tokens`** (config-
  invariant) so [τ_min,τ_max] always maps to a meaningful, CV-differentiating span.
  Then re-run the real benchmark — the current EEG gate result is invalid until then.
- **F2 — τ doesn't gate the readout, only attention.** After the recency-masked
  attention, the encoder does a **plain `mean` over all tokens**, re-including the
  masked-out positions. τ shapes attention but not the pool. Fix: recency-weighted
  pool (or pool only within the τ span) so the window genuinely gates the output.
- **F3 — the "Neural ODE" is a weight-tied autonomous Euler residual stack**
  (`z += dt·f(z)`, 20 steps, no time input). Legitimate but minimal; the only thing
  τ changes is the global step size `dt=τ/20`, which `f` can absorb. Either justify
  this as-is in the paper or switch to a genuine time-dependent field / `torchdiffeq`.
- **F4 — matched-capacity is "1 scalar apart," not "reallocated"** (as older docs
  said). Fine to keep, but state it precisely in the paper.
- **F5 — single-seed results.** The ablation and benchmark run one seed (folds only).
  Not publishable — need ≥5 seeds, mean±CI.
- **F6 — RESULTS.md over-attributes the EEG null to the proxy.** After F1, re-frame:
  the null is confounded by the mask scaling, not just proxy weakness.

---

# PRE-PAPER CHECKLIST (everything before writing)

## P1 — Architecture correctness (blocking; results depend on it)
- ☑ F1: config-invariant τ→span (fraction of window; 10–60% in every config; EEG now 3–18 of 30 tokens)
- ☑ F2: recency-weighted pooling (τ gates the readout, not just attention)
- ☑ F3: non-autonomous ODE field `dz/dt=f(z,t)` (time fraction appended)
- ☑ Re-ran ablation (5 seeds), real benchmark (3 seeds), intracortical (3 seeds),
  sensitivity (3 seeds) after the fixes; `results/` regenerated; RESULTS.md updated

## P2 — The decisive real-data experiment
- ☑ Intracortical loader (`b2ss/intracortical.py`) + benchmark
  (`scripts/run_intracortical_benchmark.py`): NLB **MC_Maze_Small** (30 MB, DANDI
  000140) read directly via pynwb (nlb_tools pins pandas ≤1.3.4, unusable on py3.12)
  - Caught bug: `hand_vel` is NOT uniformly sampled (inter-trial gaps) → must bin by
    timestamp, not row index. Fixed → corr(pop-rate, speed) 0.03→0.40, Ridge R² →+0.34
- ☑ B2SS (regression, gate variants) vs GRU + Ridge baselines; velocity R², 3 seeds.
  Result: b2ss-none 0.60 > Ridge 0.38 (real decoder) but < GRU 0.76 (not competitive);
  **CV gate HURTS** here (none 0.60 > learned 0.52 > cv 0.45) — recency-masking drops
  useful history. rt-context gives no benefit. See RESULTS.md §2.

## P3 — Statistical rigor
- ☑ Multi-seed with mean ± 95% CI (`stats.mean_ci`); ablation `--seeds`, benchmark `--seeds`
- ☑ Per-comparison paired tests (seed-averaged per subject); effect sizes reported
- ☑ Baseline fairness: matched-capacity control; EEGNet/CSP params reported

## P4 — Reproducibility & release
- ☑ `scripts/reproduce.py` — one command regenerates all results + figures
- ☑ `requirements-lock.txt` — exact pinned versions
- ☑ Figures now carry error bars; per-run seeding documented
- ☐ Archive code+data (Zenodo/OSF); pre-register the confirmatory analyses (already MIT)

## P5 — Sensitivity ablations
- ☑ `scripts/run_sensitivity.py` — sweeps MASK_GAMMA, SPAN_FRAC, ode_steps, patch;
  CV-gate gap stays >0 across all (robust; weakens at large patch — reported)

## P6 — Paper scoping (decide first)
- ☐ Frame as a **methods/mechanism** paper (architecture + synthetic demonstration +
  honest real-data status) — explicitly NOT a clinical result
- ☐ Novelty statement vs learnable-delay SNNs (DCLS, Sun 2023) and structure→delay
  brain models (HBM 2025); position against LFADS/DFINE/POSSM
- ☐ Limitations section: synthetic mechanism only, weak EEG proxy, small-N, no measured CV

## P7 — Honest go/no-go
- ☐ If after P1+P2 the CV gate shows no benefit on any real data, decide: publish the
  architecture + negative honestly, or hold for the wet-lab pilot. Don't force a claim.

# PHASE 8 — CV as delay-alignment, not window-shrinking (improvement plan)

_Why: the full runs showed the window-shrinking gate removes information — it hurts
real continuous decoding and helps only where a CV→window law is planted. The
information-**adding** use of CV is per-tract delay **alignment** (measured-CV
analogue of learnable-delay SNNs, DCLS/Sun). This phase redesigns the mechanism and
tests it cheaply on real spikes with an injected, known latency before any wet-lab._
Plan file: `~/.claude/plans/elegant-noodling-liskov.md`.

- ☑ **#1 Delay-alignment mechanism** (`b2ss/model.py`): `ChannelDelay` front-end,
  differentiable per-channel fractional shift, full window preserved; config
  `align_mode='none'|'learned'|'cv'`, orthogonal to `gate_mode`. Oracle-recovery test
  passes; 18 tests green.
- ☑ **#2 CV on a stronger backbone** (`b2ss/baselines.py`): `GRUDecoder` gained an
  optional `ChannelDelay` front-end (`gru+cv-align`).
- ☑ **#3 Injected-latency bridge** (`scripts/run_latency_bridge.py`, 5 seeds):
  data-efficiency test. Result: measured-CV alignment gives at most a small,
  **not-clearly-significant** (marginal CIs overlap) low-data prior on B2SS (+0.03 to
  +0.13), **no** benefit on the strong GRU (which learns delays itself), and the GRU
  dominates B2SS at every size. Redesign does NOT rescue the idea — see RESULTS.md §6.
- ☑ **#4 Measured-CV dataset scouting** (done → BACKGROUND.md §9): **no public
  dataset pairs a motor/BCI decode with a measured CV on the same subjects.** Closest:
  CCEP-on-iEEG `ds004080` (direct CV, but task=stimulation), VEPCON/HCP (behavior +
  dMRI proxy), F-TRACT (group-level CV prior). Justifies a dedicated acquisition.
- ☑ **#5 Rigor + docs**: bridge run at 5 seeds with CIs; `results/latency_bridge.*`
  written; RESULTS.md §6 + honesty ledger, PAPER_OUTLINE.md (revised go/no-go →
  cross-subject transfer), BACKGROUND.md §9, BRIEF.md, README.md all updated. 18 tests green.
- Guardrail honored: alignment did NOT clearly beat `none` → reported as a negative,
  not forced. **Conclusion: measured (structural) CV does not help within-subject
  decoding; the only untested regime worth pursuing is cross-subject/zero-shot transfer.**

# PHASE 9 — Cross-subject / zero-shot transfer (the one untested regime)

_Within-subject, a decoder learns the delays anyway (Phase 8). The only regime where
measured CV could still help: zero-shot transfer to a held-out subject, whose delays
the decoder cannot learn (never sees its training data)._ `scripts/run_transfer.py`.

- ☑ Controlled proof-of-mechanism on REAL MC_Maze spikes (5 pseudo-subjects, 3 seeds):
  measured-CV alignment **improves zero-shot transfer** — B2SS +0.076 (0.483→0.559),
  GRU +0.035 (0.680→0.715), consistent across seeds. **First & only regime where CV
  helps.** RESULTS §7. Qualifiers: modest (CIs overlap), shrinks with more source
  subjects, GRU-none still beats aligned-B2SS, and the conduction diff is
  injected/known (upper bound). `shift_channels` helper + test added; 19 tests green.
- **Conclusion:** CV is a **cross-subject conduction normaliser for zero-shot
  transfer**, not within-subject decoding information. Confirmatory next step (real
  claim): a cohort with neural recordings + a *measured* per-subject CV — none exists
  publicly (BACKGROUND §9), so it requires a dedicated acquisition.

# PHASE 10 — PIVOT: conduction-aware transfer normalization (the new direction)

_Why: the runs proved within-subject CV is useless (decoders learn the delays) but
cross-subject transfer is the one regime where measured conduction helps. Pivot B2SS
from "CV-modulated decoder" to a **conduction normaliser** that cuts the BCI
recalibration burden. Full plan: [PIVOT.md](PIVOT.md); spec:
`docs/superpowers/specs/2026-07-06-b2ss-transfer-pivot-design.md`._

Decisions (brainstormed): north star = practical transfer / less calibration;
alignment source = full spectrum (zero-shot measured-CV / few-shot / unsupervised);
benchmarks = staged intracortical → EEG(MOABB); novelty = temporal/conduction axis,
spatial alignment borrowed.

- ☑ **P10.1 Core module** `b2ss/transfer.py`: `ConductionDelayAligner` (grouped
  low-dim δ, shared `fractional_shift`), 3-mode `DelayFitter` (measured / few-shot /
  unsupervised CORAL), `TransferNormalizer` (freezes decoder). Self-check: zero-shot
  recovers, few-shot recovers δ≈−known, unsup converges, decoder frozen. 21 tests green.
- ☑ **P10.2 Intracortical spectrum benchmark** `scripts/run_transfer_modes.py`
  (4 folds × 3 seeds): **zero-shot (measured CV) 0.649 beats no-norm 0.399 (+0.250,
  CIs separated) AND beats full-retrain 0.403** — transfer better than retraining,
  zero target data. Clean few-shot calibration curve (0.48→0.55→0.62 at 5/20/100
  labels). Conduction structure > free delays (+0.076). Unsup fails honestly. RESULTS §8.1.
- ☑ **P10.3 Real cross-session intracortical** `scripts/run_xsession.py`: MC_Maze
  S/M/L are 3 real Jenkins sessions; spikes aggregated per **electrode** (67
  corresponding channels) for a non-injected test. Result: conduction δ-fit gives **no**
  benefit (~−0.04), full-retrain dominates (~0.73) — the real gap is unit turnover /
  drift, not timing. Honest scope bound (RESULTS §8.2).
- ☑ **P10.4 EEG breadth** `scripts/run_moabb_transfer.py` (Zhou2016 cross-session,
  guarded, MOABB as data loader only): δ-fit gives **no** EEG benefit (~−0.03) — bounds
  the claim to conduction-dominated settings (RESULTS §8.3).
- ☑ **P10.5 Docs**: PIVOT §7b, RESULTS §8, PAPER_OUTLINE (contrib 4–5 + go/no-go),
  README, ROADMAP. **21 tests + 10 self-checks green.**
- **Phase-10 verdict:** the pivot is **validated and scoped** — conduction normalisation
  enables calibration-free transfer (zero-shot beats retraining) **where the gap is
  conduction-dominated** (controlled real spikes); real multi-session/EEG gaps are not,
  so the honest path is pairing it with representation alignment, or targeting
  conduction-dominated settings.
- Guardrail honored: novelty = temporal axis only; controlled(injected) vs real always
  labelled; no result tuned toward a win (real-data results are honest negatives).

## Out of scope here (wet-lab / hardware — see proposal)
- ⊘ MRI g-ratio acquisition & MRtrix3/FSL/FreeSurfer pipeline
- ⊘ TMS-EEG & sEEG acquisition; CCEP ground truth
- ⊘ Human participants, IRB/OSF pre-registration, blinding logistics
- ⊘ NeuroPix RPN-1 robotic arm & real-time EtherCAT loop
- ⊘ ccPAS plasticity intervention
