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

## Layout

```
b2ss/
  cv.py      CV from g-ratio (Berman), combined TMS-EEG estimate, bootstrap CIs
  data.py    synthetic EEG + kinematics with a ground-truth CV→latency law
  model.py   B2SS decoder (Transformer + CV gate + Euler Neural-ODE) & control
  train.py   training loop (Adam, MSE, early stopping)
  eval.py    metrics: MSE, Pearson r, effective latency (xcorr peak lag)
scripts/
  run_offline_comparison.py   Experiment-3 offline comparison
tests/
  test_b2ss.py                runnable self-checks
```

## License

MIT (as stated in the proposal). Code only; no participant data.
