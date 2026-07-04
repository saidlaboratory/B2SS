# B2SS — Build Roadmap

Living to-do list for the **software artifact**. Updated as work lands.
Status: ☐ todo · ◐ in progress · ☑ done · ⊘ out of scope (wet-lab/hardware).

_Last updated: 2026-07-04_

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

## Phase 3 — Decoder (`b2ss/model.py`)
- ☑ Transformer encoder (4 layers, 8 heads, d=64, FFN=256) over channel tokens
- ☑ CV gate `τ = σ(W_cv·CV + b_cv)·τ_max + τ_min` → temporal attention mask
- ☑ Neural-ODE readout, fixed-step Euler (Δt=5 ms, 40 steps) → 6-DOF
- ☑ Matched-capacity control decoder (no gate; τ learned constant; params reallocated)
- ☑ Self-check: forward-pass shapes; param counts within ~2% of each other; τ within [τ_min, τ_max]

## Phase 4 — Train & eval (`b2ss/train.py`, `b2ss/eval.py`)
- ☑ Training loop: Adam(1e-4), MSE + 1e-5 L2, early stopping, val split
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
- `run_offline_comparison.py` (7 subjects spanning the CV range, 40 epochs,
  n_train=150): B2SS beat the matched control on **6/7** subjects,
  **~12% lower MSE**, higher Pearson r throughout. The CV gate's window τ tracked
  CV as designed (slow 28 m/s → 76 ms; fast 70 m/s → 42 ms).
- Honest caveat: this only shows the decoder *can* exploit a CV→latency structure
  when one exists — not evidence for the scientific hypotheses (which need real data).

## Out of scope here (wet-lab / hardware — see proposal)
- ⊘ MRI g-ratio acquisition & MRtrix3/FSL/FreeSurfer pipeline
- ⊘ TMS-EEG & sEEG acquisition; CCEP ground truth
- ⊘ Human participants, IRB/OSF pre-registration, blinding logistics
- ⊘ NeuroPix RPN-1 robotic arm & real-time EtherCAT loop
- ⊘ ccPAS plasticity intervention
