# B2SS — Paper Scoping (P6/P7)

The honest framing to decide *before* writing, and the outline if we proceed.
Numbers are filled from [RESULTS.md](RESULTS.md) once the full multi-seed runs land.

## 1. What this paper is — and is not

**Is:** a *methods / mechanism* paper. "A conduction-velocity-modulated decoder
architecture (Transformer + CV-gated integration window + Neural-ODE readout), and
a rigorous, adversarial test of *when* a measured-CV prior helps decoding."

**Is not:** a clinical or neuroscience result. We make **no** claim that B2SS
improves prosthetic control in humans, that CV is decodable from a person's brain
in real time, or that any hypothesis H1–H6 is confirmed. Those need the wet-lab
study (MRI g-ratio, TMS-EEG, sEEG, closed-loop) that this software does not touch.

Framing it any stronger than "methods + mechanism + honest real-data status" is not
supported by the evidence and will (rightly) be rejected.

## 2. Contributions (what is genuinely new)

1. **The CV-gate mechanism**, formalized: an integration window τ set from a
   structural CV prior, gating attention span + ODE integration time, with an
   uncertainty-aware fallback to the population window.
2. **A clean separation of "prior" vs "information"** (the central methodological
   result): on homogeneous-CV data the gate is a *data-efficiency prior* whose
   advantage shrinks with data; on heterogeneous-CV data (CV varies per context) it
   is *genuine information* a learned constant cannot capture, and the gap persists.
   Backed by a sensitivity sweep showing robustness across hyperparameters.
3. **Honest real-data status**: on two public datasets *without measured CV*, the
   gate gives no benefit — reported plainly, not buried. This is the paper's
   integrity anchor and its call for the measured-CV experiment.

## 3. Positioning / novelty (vs prior work — see BACKGROUND.md)

- **vs LFADS / DFINE / POSSM** (neural decoders): those learn latent dynamics with
  no structural prior; B2SS injects a *measured biophysical* prior (CV) into the
  temporal scale. We benchmark against the decoder family (GRU/Ridge here; the
  LFADS/POSSM numbers are context).
- **vs learnable-delay SNNs (DCLS, Sun 2023)**: they *learn* per-connection delays
  from data; B2SS *measures* the delay-setting variable (CV) and uses it as a prior
  — complementary, and testable head-to-head as future work.
- **vs structure→delay brain models (HBM 2025) & CV-from-MRI (Drakesmith 2019,
  Asadi 2025)**: those map structure→conduction delay or fit delays to neural data;
  none is a *decoder* that consumes CV to set its temporal window. That is the gap.

## 4. Outline & experiment mapping

| Section | Content | Evidence |
| --- | --- | --- |
| Intro | spatial-only bottleneck; CV as a structural prior | BACKGROUND §1–2 |
| Architecture | conv patch-embed → CV gate → Neural-ODE; uncertainty gate | `b2ss/model.py`; F1–F3 fixed |
| Mechanism (synthetic) | Study A (prior) vs Study B (information); ablation | `run_ablation.py` |
| Robustness | sensitivity to γ, span-fraction, ODE steps, patch | `run_sensitivity.py` |
| Real data I (EEG) | competitive-ness vs EEGNet/CSP; gate w/ mu proxy | `run_real_benchmark.py` |
| Real data II (intracortical) | velocity R² vs GRU/Ridge; gate w/ context | `run_intracortical_benchmark.py` |
| Stats/power | corrected effect sizes; §6 harness | `b2ss/stats.py` |
| Limitations | see §5 | — |

## 5. Limitations (must be stated plainly)

- The CV mechanism is **demonstrated only in synthetic data** where the CV→window
  law is planted; that is a proof of mechanism, not of its existence in brains.
- On **real EEG** the CV proxy (mu peak frequency) is weak/indirect and the gate
  gives no benefit; the architecture is not competitive with EEGNet/CSP on
  small-trial classification (wrong regime).
- On **real intracortical** continuous decoding, the gate does **not** help (the
  recency window discards useful history; a plain GRU is stronger), and the
  reaction-time context is not a conduction-velocity signal.
- No **measured** CV exists in any real dataset used here; the decisive test
  (measured CV → decoding benefit) requires the wet-lab pilot.
- Small N (EEG 8 subjects; one intracortical session).

## 6. Go / no-go (P7) — recommendation

Two defensible paths; pick based on appetite:

- **Publish now as methods+mechanism (with honest negatives).** Venue: a methods/ML
  track (e.g. *J. Neural Eng.*, a NeurIPS/ICLR workshop). Claim = the architecture
  + the prior-vs-information result + transparent real-data negatives. This is
  honest and citable, but it is a "mechanism works in silico; unproven in vivo"
  paper — modest impact.
- **Hold for the measured-CV experiment.** Run the wet-lab pilot (or find a public
  dataset pairing neural recordings with a structural CV/g-ratio proxy) and only
  then claim CV helps real decoding. Higher impact, much higher cost/time.

**Recommendation:** if the goal is a real scientific claim, **hold** — the current
evidence does not show CV helping any real decode, and a methods paper resting on a
synthetic mechanism + two negatives is thin. If the goal is to stake the idea and
release the framework, **publish now** with the framing above and the negatives
front-and-center. Do **not** publish anything claiming CV improves real BCI
decoding on the current evidence.
