# B2SS — Brain-to-Signal-Speed Decoder

**A conduction-velocity-modulated brain–computer interface for fluid prosthetic control.**

Current BCI decoders read motor intent from the *spatial* pattern of neural
activity — which neurons fire, and when — but treat every spike as if it
arrives instantaneously. They ignore **conduction velocity (CV)**: the speed at
which action potentials travel along white-matter axons, set by each person's
myelination (the MR *g-ratio*). B2SS uses an individually-measured CV estimate
as a **structural prior** that tunes the decoder's temporal integration window:
faster tracts → shorter window → lower latency; slower tracts → longer window →
higher accuracy.

This repository is the **software artifact** of the [B2SS research
proposal](#scope) — the decoder, the CV-estimation math, a synthetic-data
harness, and the offline B2SS-vs-control comparison. The wet-lab experiments
(MRI, TMS-EEG, sEEG, robotic arm, human participants) live in the proposal, not
here. See [BRIEF.md](BRIEF.md) for the full science, [BACKGROUND.md](BACKGROUND.md)
for the literature, and [ROADMAP.md](ROADMAP.md) for build status.

> **⚠️ The project has pivoted — read [PIVOT.md](PIVOT.md) first.** The multi-seed
> runs showed the original claim (CV improves *within-subject* decoding) is false —
> a decoder learns the conduction delays from data — but that CV helps *cross-subject
> transfer*. B2SS is being rebuilt from a "CV-modulated decoder" into a **conduction
> normaliser** that cuts BCI recalibration burden. The results below
> ([RESULTS.md](RESULTS.md)) are the evidence that motivated the pivot.

## Scope: what this repo is (and isn't)

| Buildable in software (here) | Wet-lab / hardware (not here) |
| --- | --- |
| CV estimation from g-ratio (Berman relation) + TMS-EEG latency | MRI g-ratio mapping, TMS-EEG acquisition |
| B2SS decoder: Transformer + CV gate + Neural-ODE readout | Human participants, IRB, robotic arm |
| Matched-capacity spatial-only control decoder | sEEG ground-truth recordings |
| Synthetic neural data with a ground-truth CV→latency law | ccPAS plasticity intervention |
| Offline training + B2SS-vs-control evaluation (Experiment 3) | Closed-loop prosthetic control (Experiment 4) |

The synthetic-data harness exists so the decoder can be exercised, tested, and
compared *before* any human data exists — it bakes in the CV→latency
relationship the proposal hypothesizes, so a correctly-built CV-modulated
decoder should measurably beat the control on it. It is a testbed, **not**
evidence for the scientific hypotheses.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.x (CPU is fine for the synthetic harness).

## Quickstart

```bash
# 1. Sanity-check the biophysics + model (fast, no training)
python -m pytest tests/ -q

# 2. Run the offline B2SS-vs-control comparison on synthetic data
python scripts/run_offline_comparison.py --subjects 7 --epochs 40
```

The comparison trains both decoders per synthetic subject (CVs spanning the
plausible range) and reports MSE, Pearson r, and effective latency — the
Experiment-3 metrics. On CV-structured data B2SS typically wins ~6/7 subjects
(~12% lower MSE), and its gate's integration window τ tracks CV as designed
(slow ~28 m/s → ~76 ms window; fast ~70 m/s → ~42 ms). Add `--realistic` for a
normal cohort (mid-range-clustered, so the effect is diluted), or
`--proposal-size` for the full model dimensions. ~2 min on CPU.

### Publication-grade experiments (v2)

These remove the reviewer threats (info-vs-prior, can-EEG-carry-it, effect-size
realism) and add the promised extras. Outputs land in `results/`.

```bash
# Gate ablation: cv vs learned vs fixed vs none. Study A (homogeneous, data-
# efficiency) shows the PRIOR benefit; Study B (heterogeneous CV) shows CV as
# genuine INFORMATION a learned constant can't capture.
python scripts/run_ablation.py

# Real EEG (PhysioNet, downloaded via MNE): B2SS vs EEGNet vs CSP+LDA, within-
# subject CV, using the MRI-free mu-frequency CV proxy. Honest finding: the EEG is
# decodable (baselines ≫ chance) but B2SS is not competitive on small-trial 2-class
# EEG — see RESULTS.md for the straight story.
python scripts/run_real_benchmark.py            # ~30-40 min CPU; --quick to smoke-test

# Decisive experiment: continuous intracortical velocity decoding (NLB MC_Maze,
# the REGRESSION regime B2SS is built for). B2SS vs GRU vs Ridge, velocity R².
# Needs pynwb + a 30 MB download (see b2ss/intracortical.py docstring).
python scripts/run_intracortical_benchmark.py

# Robustness: is the CV-gate benefit a knife-edge or robust to hyperparameters?
python scripts/run_sensitivity.py

# Phase 8: CV as delay-ALIGNMENT (not window-shrinking). Inject a known latency into
# real MC_Maze spikes; does measured-delay alignment recover decoding? (data-efficiency)
python scripts/run_latency_bridge.py

# Phase 10 (THE PIVOT): conduction normalization for calibration-light transfer.
python scripts/run_transfer_modes.py    # calibration-cost spectrum (zero/few/unsup vs retrain)
python scripts/run_xsession.py          # REAL cross-session transfer (MC_Maze S/M/L, per-electrode)
python scripts/run_moabb_transfer.py    # EEG breadth (Zhou2016 cross-session; needs moabb)

# Real-time inference latency vs the proposal's <50 ms budget.
python scripts/bench_latency.py --proposal-size

# The pre-registered statistical harness (corrected effect sizes).
python -m b2ss.stats

# Regenerate EVERYTHING (tests + all experiments + figures) in one command:
python scripts/reproduce.py            # add --fast to skip downloads/long runs
```

See [RESULTS.md](RESULTS.md) for the numbers and what they do (and don't) show, and
[PAPER_OUTLINE.md](PAPER_OUTLINE.md) for the honest paper framing and go/no-go.
Pinned versions are in `requirements-lock.txt`; the intracortical benchmark also
needs `pynwb` (`pip install pynwb`).

## Layout

```
b2ss/
  cv.py           CV from g-ratio (Berman), combined TMS-EEG estimate, bootstrap CIs
  data.py         synthetic EEG + kinematics; homogeneous & heterogeneous CV regimes
  model.py        B2SS decoder (patch-embed → CV gate → Neural-ODE); 4 gate modes
  train.py        training loop (Adam, MSE/CE, early stopping); regression + classification
  eval.py         metrics: MSE, Pearson r, effective latency (xcorr peak lag)
  proxies.py      CV proxy from EEG mu peak frequency (Corcoran-style)
  datasets.py     PhysioNet EEGMMI loader (MNE) + cropped-window helpers
  intracortical.py NLB MC_Maze loader (pynwb): binned spikes + hand velocity
  baselines.py    EEGNet, CSP+LDA, GRU, Ridge
  stats.py        power (corrected d), ICC(2,1), mixed-model ΔR², FDR/Bonferroni, CIs
scripts/
  run_offline_comparison.py     synthetic Experiment-3 comparison
  run_ablation.py               gate ablation (Study A prior / Study B information)
  run_real_benchmark.py         real EEG: B2SS vs EEGNet/CSP (multi-seed)
  run_intracortical_benchmark.py real intracortical velocity R² vs GRU/Ridge
  run_sensitivity.py            hyperparameter robustness of the CV-gate benefit
  run_latency_bridge.py         Phase-8 delay-alignment on real spikes + injected latency
  run_transfer.py               Phase-9 cross-subject zero-shot transfer with measured CV
  bench_latency.py              inference latency vs 50 ms budget
  reproduce.py                  one command → all results + figures
tests/
  test_b2ss.py                  16 runnable checks
```

## License

MIT (as stated in the proposal). Code only; no participant data.
