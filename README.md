# B2SS — conduction velocity, cross-session drift, and what actually helps

**How much of the cross-session drift in an intracortical BCI is conduction timing —
and what do you do about the rest?**

The project began as a decoder that used an individually-measured **conduction velocity**
(CV, the speed action potentials travel along myelinated axons, set by the MR *g-ratio*)
to tune its temporal integration window. That claim did not survive contact with data, and
neither did the two reframings after it. What survived is better than the original idea:
a **calibrated instrument** for splitting a cross-session decoding gap into a timing
component and a representation component, and the practical consequence of what it
measures.

The instrument says: on real multi-session intracortical recordings, conduction timing
contributes **essentially nothing** (Δ velocity R² = −0.015 [−0.027, −0.004]), while the
same measurement on a gap that is timing-dominated by construction reads **+0.250**
[+0.191, +0.309]. The positive control is what makes the null worth believing.

The consequence, and the part that generalises past BCI: the drift that *does* dominate is
per-channel gain and offset, and the calibration data you get online is a **biased** sample
of the session, not merely a small one. A standardiser fitted on the first 25 windows scores
0.027; on 25 *random* windows it scores 0.520. Same estimator, same N — so the usual
"vary N and watch the curve" diagnostic cannot see the real problem. **CADENCE** shrinks each
estimate toward the source prior, which helps by declining to commit to the biased estimate;
the textbook empirical-Bayes version, which models variance instead, fails outright.

See [RESULTS.md](RESULTS.md) for every number and what it does and does not show,
[BETTER.md](BETTER.md) for the adversarial review this repo was rebuilt against,
[BACKGROUND.md](BACKGROUND.md) for the literature, and [ROADMAP.md](ROADMAP.md) for build
status. The wet-lab experiments (MRI, TMS-EEG, sEEG, robotic arm, human participants) live
in the proposal, not here.

> **Three claims this repo does NOT make.** (1) CADENCE does not lead the continual stream —
> a plain per-session standardiser gets higher cumulative R² once calibration data is ample,
> and we print that. (2) Label-free adaptation does not beat recalibration: given the
> session's own input normalisation, a recalibrated decoder reaches ~0.75 R² against the best
> adapter's 0.58. What these methods buy is **cheapness, not accuracy**. (3) The conduction
> term carries no accuracy on any real gap we measured; it earns its place as the diagnostic,
> not as the adapter.

## Scope: what this repo is (and isn't)

| Here (public data + software) | Not here (wet-lab / hardware) |
| --- | --- |
| The drift decomposition: timing vs representation, in velocity-R² units | MRI g-ratio mapping, TMS-EEG acquisition |
| Real intracortical streams: MC_Maze, 11 monkey-Indy sessions | Human participants, IRB, robotic arm |
| CADENCE + the TTA baseline battery over a frozen decoder | sEEG ground-truth recordings |
| CV estimation from g-ratio (Berman relation), synthetic CV→latency harness | ccPAS plasticity intervention |
| The B2SS decoder itself (superseded — it loses to a GRU; kept for the record) | Closed-loop prosthetic control |

**No dataset here has a measured conduction velocity.** Where a "measured CV" appears in an
experiment it is *injected and known* — a positive control that bounds what perfect
conduction knowledge could buy, not evidence about brains. The synthetic harness likewise
bakes in the CV→latency law it tests, so it proves a mechanism can work, not that it exists.
BACKGROUND §9 documents the (verified) absence of any public dataset pairing a decode task
with a per-subject CV.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.x (CPU is fine for the synthetic harness).

## Quickstart

```bash
# 1. Everything runnable, no downloads (32 checks, ~20 s)
python -m pytest tests/ -q

# 2. The paper's central figure — timing vs representation, same units.
#    Consumes results/transfer_modes.json + results/xsession.json (both committed).
python scripts/run_decomposition_figure.py
```

That prints the whole claim in three lines: conduction alignment is worth +0.250 R² where
timing dominates by construction, and −0.015 on the real cross-session gap.

To exercise the *original* CV-modulated decoder on synthetic data where the CV→window law
is planted (a mechanism demo, not evidence):

```bash
python scripts/run_offline_comparison.py --subjects 7 --epochs 40   # ~2 min CPU
```

### Full experiment set

Everything below writes to `results/`. The Phase 7–10 runs are the evidence chain that
falsified the original hypotheses; Phase 11 is the current work.

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

# Phase 10: conduction normalization for calibration-light transfer. These two feed the
# decomposition — a timing-dominated positive control and a real-data null in the same units.
python scripts/run_transfer_modes.py    # calibration-cost spectrum (zero/few/unsup vs retrain)
python scripts/run_xsession.py          # REAL cross-session transfer (MC_Maze S/M/L, per-electrode)
python scripts/run_moabb_transfer.py    # EEG breadth (Zhou2016 cross-session; needs moabb)

# Phase 11 (the current work) — real monkey-Indy stream, 11 sessions over ~1 month.
python scripts/run_decomposition_figure.py   # THE SPINE: timing vs representation marginal
python scripts/run_indy_calibration.py       # online data-efficiency curve, out to full calibration
python scripts/run_calibration_bias.py       # WHY it fails: same N, different draw (bias, not noise)
python scripts/run_tau_sweep.py              # tau not tuned on eval (LOSO) + the empirical-Bayes null
python scripts/run_indy_stream.py            # the continual stream: regret + recalibration ceilings

# Real-time inference latency vs the proposal's <50 ms budget.
python scripts/bench_latency.py --proposal-size

# The pre-registered statistical harness (corrected effect sizes).
python -m b2ss.stats

# Regenerate EVERYTHING (tests + all experiments + figures) in one command:
python scripts/reproduce.py            # add --fast to skip downloads/long runs
```

See [RESULTS.md](RESULTS.md) for every number and what it does (and doesn't) show, and
[PAPER_OUTLINE.md](PAPER_OUTLINE.md) for the framing and go/no-go. Pinned versions are in
`requirements-lock.txt`; the intracortical benchmarks also need `pynwb` and `h5py`.

Historical context, kept because the falsifications are part of the argument:
[BRIEF.md](BRIEF.md) (the original science), [PIVOT.md](PIVOT.md) (why the within-subject
claim was abandoned), [BETTER.md](BETTER.md) (the review this repo was rebuilt against).

The Indy sessions are not redistributed here — download the `.mat` files from Zenodo record
583331 into `~/b2ss_data/indy/` (see [`b2ss/indy.py`](b2ss/indy.py)). The same record carries
a second monkey (`loco`); every Indy script takes `--subject loco`, but we have not run it —
those files are ~1.1–1.6 GB each (~12 GB for the set), so **every stream result here is
single-subject**.

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
  intracortical.py NLB MC_Maze loader (pynwb); injected-latency rig for the positive control
  indy.py         monkey-Indy multi-session loader (h5py); --subject switches monkey
  baselines.py    EEGNet, CSP+LDA, GRU, Ridge
  transfer.py     frozen decoder + conduction-delay aligner (the measurement instrument)
  cadence.py      CADENCE: frozen decoder + consolidation-shrinkage per-channel affine
  tta_baselines.py No-Adapt, Tent, CoTTA, RDumb, free-LoRA (the structure ablation)
  ibci_baselines.py MPA (floored + unfloored), NoMAD
  stream.py       online replay harness (temporal order + revisits)
  continual.py    cumulative / worst-session R², collapse-rate, regret, BWT
  stats.py        power, ICC(2,1), mixed-model ΔR², paired-by-unit tests, FDR/Bonferroni
scripts/
  run_decomposition_figure.py   THE SPINE: timing vs representation marginal
  run_indy_calibration.py       online data-efficiency curve out to full calibration
  run_calibration_bias.py       same N, different draw — bias vs noise (the general lesson)
  run_tau_sweep.py              τ / std_floor sweep, LOSO selection, empirical-Bayes null
  run_indy_stream.py            continual stream: regret, ceilings, structure ablation
  run_transfer_modes.py         calibration-cost spectrum (the timing-dominated control)
  run_xsession.py               real cross-session MC_Maze (the representation null)
  run_moabb_transfer.py         EEG cross-session (the second null)
  run_ablation.py               gate ablation (Study A prior / Study B information)
  run_real_benchmark.py         real EEG: B2SS vs EEGNet/CSP (multi-seed)
  run_intracortical_benchmark.py real intracortical velocity R² vs GRU/Ridge
  run_sensitivity.py            hyperparameter robustness of the CV-gate benefit
  run_latency_bridge.py         Phase-8 delay-alignment on real spikes + injected latency
  run_transfer.py               Phase-9 cross-subject zero-shot transfer
  run_offline_comparison.py     synthetic Experiment-3 comparison
  bench_latency.py              inference latency vs 50 ms budget
  reproduce.py                  one command → all results + figures
tests/
  test_b2ss.py                  32 runnable checks
```

## License

MIT (as stated in the proposal). Code only; no participant data.
