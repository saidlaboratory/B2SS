# B2SS — Background & Literature

The scientific foundation for a conduction-velocity-modulated BCI decoder, with
the proposal's own references verified against the primary sources and enriched
with recent related work (2019–2026).

**How this was built.** Each citation in the proposal (v1.1, §11) was looked up,
its actual finding extracted, and the proposal's claim about it adversarially
fact-checked against the source. Newer work was then searched per theme. Every
paper below was confirmed to exist; DOIs are given. Where the proposal overstates
or mis-attributes a source, it is flagged in **[Citation accuracy notes](#citation-accuracy-notes)** —
those are corrections to the *proposal text*, not to the science, which is sound.

---

## 1. Conduction velocity, the g-ratio, and the biophysics

The premise: action-potential speed on a myelinated axon is set by axon calibre
and the **g-ratio** (inner axon diameter ÷ total fibre diameter). Rushton's
classic cable analysis established that conduction velocity scales with fibre
diameter and is maximised at an optimal g-ratio near 0.6 (Rushton, 1951) — the
biophysical anchor for treating g-ratio as a velocity proxy. (Our
[`b2ss/cv.py`](b2ss/cv.py) reproduces this optimum at g = e^(−½) ≈ 0.607 as a
sanity check.)

Turning this into an *in-vivo, per-tract* number requires MRI. Stikov et al.
(2015) gave the reference recipe for estimating the myelin g-ratio non-invasively
by combining a myelin-volume map (e.g. magnetization transfer) with a
fibre/neurite-density map from diffusion MRI. Drakesmith et al. (2019) then closed
the gap to velocity: a 14-parameter sensitivity analysis of a myelinated-axon
electrophysiology model showed that **~85% of the variance in conduction velocity
is captured by just two MRI-measurable quantities — axon diameter and g-ratio** —
and that a simplified Rushton relation (v ≈ d·√(−ln g)) captures the dependence.
Their whole-brain estimates put corpus-callosum CV at ~8–10 m/s, aligned with
primate electrophysiology, but with a critical caveat: estimates are accurate
(<5% error) only for large axons (diameter > 4 µm, g-ratio 0.6–0.85) and become
unreliable for the sub-micron axons that dominate the CNS. This bounds where a
decoder should *trust* its CV prior.

Berman, Filo & Mezer (2019, *Modelling conduction delays…*) is the specific
source for modelling conduction **delay** from an MRI-measured g-ratio — the
lineage of the proposal's CV = k·v(g) relation. (The proposal's reference list
cites a *different* Berman 2019 paper by mistake; see the notes.)

The most direct validation to date arrived in 2025: Asadi et al. (bioRxiv) built
a multivariate model predicting neurophysiological CV from 7T MRI microstructure
(axon radius, axonal water fraction, extra-axonal diffusivity, T1) and validated
it against **ground-truth intracranial cortico-cortical evoked potential (CCEP)
latencies** — exactly the sEEG validation the B2SS proposal plans (Experiment 2b).
Their model explained **~29% of the variance** in measured CV. That number is a
realistic ceiling for how much a structural CV prior can contribute, and a useful
prior for B2SS's own expectations. Supporting the microstructure inputs,
volume-electron-microscopy work (Abdollahzadeh et al., 2024) validates the
diffusion-MRI "Standard Model" parameters against histology, and Barakovic-style
cross-species analyses (2024) characterise how axon diameter and g-ratio co-vary
across tracts — relevant because CV depends on both jointly, so they should not be
treated as independent decoder inputs.

## 2. CV carries behaviourally-relevant information

The load-bearing empirical result for B2SS is Clark et al. (2022, *eLife*): in
**217 healthy adults**, the MR g-ratio of the parahippocampal cingulum (hence
inferred CV) was **associated with autobiographical memory recall** — specifically
the number of internal/episodic details. The effect was tract- and task-specific
(present for real-life recall, absent for lab memory tests) and appeared driven by
inner axon diameter rather than myelin. This is the first demonstration, at
BCI-relevant sample size, that an MRI-derived CV of a *specific* tract is a real,
individually-varying predictor of behaviour — the strongest existing support for
feeding a per-subject, per-tract CV estimate into a decoder as a prior.

*Magnitude, stated accurately:* the association is a small correlation,
r(211) = 0.18, p = 0.008 (≈ Cohen's d 0.37), and it is correlational, not
predictive or causal. The proposal's "d ≈ 0.45" overstates it (see notes) — worth
tracking, since B2SS's power analysis leans on this effect size.

## 3. CV is plastic: activity-dependent myelination

B2SS treats CV not as fixed hardware but as a variable that changes with learning
and can be nudged (Experiment 5, ccPAS). The biology backs this:

- **Gibson et al. (2014, *Science*)** — the seminal causal demonstration.
  Optogenetic stimulation of premotor cortex in awake mice drove OPC
  proliferation, oligodendrogenesis, and circuit-selective myelination, with an
  associated improvement in motor function; pharmacological blockade of
  oligodendrocyte differentiation abolished the gain. (Precisely: the readout was
  motor *function* — contralateral forelimb swing speed — via a drug, not a
  motor-*learning* task via a genetic knockout; see notes.)
- **Fields (2015, *Nat Rev Neurosci*)** — the framing review establishing
  activity-dependent myelination as a distinct plasticity mechanism that tunes
  conduction and timing.
- **Pajevic, Basser & Fields (2014)** and follow-ups model how myelin plasticity
  tunes conduction delays to shape oscillations and synchrony.
- Recent computational work makes the mechanism concrete and *transferable to
  decoder design*: an Activity-Dependent Myelination learning rule where net CV
  scales with firing rate, acting as homeostatic timing control (Nat. Comput.
  Sci., 2022); an oligodendrocyte model that synchronises correlated spike trains
  by selectively speeding lagging axons, using only local signals, optimal at
  ~10–40 ms glial time-constants (*eLife*, 2023); and a large-scale model showing
  conduction delays are a **high-gain** control variable — a ~1 ms shift can move
  a gamma-band phase by tens of degrees (PNAS, 2020).
- **Human, in-vivo, during motor learning:** longitudinal DTI + myelin mapping
  (Cerebral Cortex, 2024) shows corticospinal white-matter change during motor-
  skill training that *precedes* cerebellar grey-matter adaptation — direct
  evidence the CV substrate B2SS targets genuinely shifts in users, and that a
  decoder's inputs will be non-stationary over training.

## 4. Measuring and modifying corticospinal conduction: TMS-EEG & ccPAS

B2SS estimates CV non-invasively via TMS-EEG MEP latency and modifies it via
paired stimulation. The supporting methods literature:

- **What an MEP latency actually indexes** (J. Physiol., 2023): MEP latency is the
  sum of intracortical processing, corticospinal conduction, spinal integration,
  and neuromuscular transmission — so a CV pipeline must attribute latency shifts
  to conduction, not cortical excitability. This defines B2SS's target variable
  and its confounds.
- **ccPAS is gated by conduction delay** (Neuromodulation, 2023): paired
  stimulation of the two motor cortices at asynchronies straddling the ~9 ms
  transcallosal delay flipped the sign of plasticity (14 ms strengthened, 4 ms
  weakened, 9 ms no change) — human spike-timing-dependent plasticity tuned to
  axonal conduction time. This is why B2SS sets its ccPAS inter-pulse interval
  from each subject's measured CV.
- **ccPAS review** (Clin. Neurophysiol., 2023): synthesises how ISI-vs-delay
  choice determines direction/magnitude of connectivity change, and flags outcome
  variability and unstandardised ISI selection as the field's open problems —
  motivating per-subject conduction-time measurement over group-average ISIs.
- **Lazari et al. (2022, *Cell Reports*)** — the proposal's plasticity anchor:
  dual-site Hebbian TMS produced an increase in an MRI myelin marker whose
  significant cluster overlapped the tract connecting the stimulated regions,
  measured 24 h later. (Precisely: the marker was magnetization-transfer
  saturation that *increased*, not qT2 that decreased; the effect was a
  brain–behaviour correlation, not a group-mean change; only a single 24 h
  timepoint was measured — see notes.)
- **i-TEP** (Brain Stimulation, 2024): an immediate TMS-evoked EEG potential
  starting ~2 ms post-pulse may expose the earliest corticospinal volleys to EEG,
  a possible route to a real-time, EMG-free conduction feature for closed-loop use.
- **TMS safety** (Rossi et al., 2009): the consensus guidelines B2SS's stimulation
  protocols operate within.

## 5. BCI decoding: state of the art and where B2SS sits

The decoding lineage B2SS extends and must be benchmarked against:

- **Intracortical motor BCIs** — Hochberg et al. (2012, *Nature*): people with
  tetraplegia controlling a robotic arm for reach and grasp (BrainGate2), the
  clinical proof-of-concept B2SS's prosthetic task descends from.
- **Single-trial latent dynamics** — LFADS (Pandarinath et al., **2018**, *Nature
  Methods*): a sequential (variational) autoencoder that infers single-trial
  neural dynamics and substantially improves behavioural decoding over spike
  smoothing. The discrete-time RNN baseline every continuous-time method is
  measured against. (Proposal dates it 2017 with wrong volume/pages; see notes.)
- **Robustness to non-stationarity** — Sussillo et al. (2016, *Nat Commun*): a
  multiplicative-RNN decoder, trained with data augmentation, stayed usable under
  recording-condition changes that crippled a Kalman filter — in 2 macaques, with
  largely *simulated* variability (electrode dropping, stale training data), and
  significant in only one animal (see notes). B2SS reframes structural CV
  variability as a further, previously-unmodelled source of variability.
- **Modern deep decoders** — a high-performance speech neuroprosthesis (Nature,
  2023) hit 62 words/min from intracortical speech cortex via an RNN + language
  model; on the EEG side, CNN-Transformer hybrids (CTNet, 2024; TCFormer, 2025)
  and EEG foundation models (LaBraM, ICLR 2024) define the current motor-decoding
  recipe; SPINT (2025) and POSSM (NeurIPS 2025) target the exact deployment regime
  B2SS aims at — causal, real-time, drift-tolerant decoding, with POSSM matching
  Transformer accuracy at ~9× lower inference cost. These set the
  accuracy/latency/generalisation bar and show where a CV term could be injected
  (the temporal/convolutional front-end).

## 6. Continuous-time models: why Transformer + Neural ODE

B2SS's architecture (encoder → CV gate → Neural-ODE readout) rests on:

- **Transformer** (Vaswani et al., 2017) — self-attention sequence modelling; the
  encoder backbone, and (in B2SS) the surface the CV-derived τ modulates via a
  temporal attention mask.
- **Neural ODEs** (Chen et al., 2018, NeurIPS best paper) — continuous-depth
  models parameterising the derivative of the hidden state. The natural place to
  inject a *velocity-conditioned* term: CV scales the integration time of the
  vector field.
- **Latent ODEs / ODE-RNNs** (Rubanova et al., 2019) — continuous-time latent
  state between observations; the scaffolding for a decoder whose effective
  timescale is not constant.
- **NODE vs RNN, head-to-head** (Sedler et al., 2023): Neural-ODE sequential
  autoencoders recover neural population dynamics and fixed-point structure at the
  true low latent dimensionality where RNNs fail — the strongest single argument
  for choosing a continuous-time vector field over a discrete recurrence for a
  velocity-modulated decoder.
- **Deployable latent dynamics** — DFINE (Nat. Biomed. Eng., 2024) keeps nonlinear
  manifolds with a tractable, causal (Kalman-like) dynamics path robust to dropped
  channels; the template for real-time CV-tunable inference.

## 7. Closest analogues to B2SS

Two lines are the nearest existing work — worth watching and citing directly:

1. **Structure-constrained conduction-delay brain models.** "Mapping Brain Lesions
   to Conduction Delays" (Human Brain Mapping, 2025) builds a *personalised*
   whole-brain oscillator model where conduction delays are set by white-matter
   structure (τ = distance/velocity + damage term), then *inverts* the delay
   parameter from empirical MEG — a published template for exactly B2SS's
   structure→delay→signal→fit loop, and evidence that delay *location* matters.
2. **CV-from-MRI validated against electrophysiology.** Drakesmith et al. (2019)
   and Asadi et al. (2025) together are the methodological spine: MRI → CV, with
   the second validated against CCEP latencies (the B2SS Experiment-2b design) and
   quantifying the ~29% variance ceiling.

**The gap B2SS fills** remains real: no existing decoder uses an individually
measured white-matter CV as a structural constraint on its *temporal integration
window*. The pieces exist separately (CV-from-MRI, conduction-delay brain models,
continuous-time decoders); B2SS is the first to combine them for prosthetic
control.

---

## Citation accuracy notes

Corrections to the **proposal text/reference list** (the underlying science holds;
these are about precision and attribution — several matter for the power analysis
and methods):

| # | Reference | Issue | Accurate statement |
|---|-----------|-------|--------------------|
| 1 | **Berman et al. (2019)**, corpus-callosum age/sex | **Wrong paper + wrong claim.** That paper (actually **2018**, NeuroImage 182:304–313) finds the callosal g-ratio is *stable* with age and shows *no* sexual dimorphism — the opposite of "varies with age and sex" — and contains **no** CV = k·v(g) formula. | The CV(g) relation belongs to **Berman, Filo & Mezer (2019)**, *Modelling conduction delays in the corpus callosum using MRI-measured g-ratio*, NeuroImage (doi:10.1016/j.neuroimage.2019.116001). Cite that for the formula. |
| 2 | **Stikov et al. (2015)** | Wrong volume/pages: cited as 93:239–251. | Actual: **NeuroImage 118:397–405** (doi:10.1016/j.neuroimage.2015.05.023). |
| 3 | **Clark et al. (2022)** | Effect size "d ≈ 0.45" is not in the paper and is inflated; "predicts" overstates a correlation. | Small correlation **r(211)=0.18, p=0.008 (≈ d 0.37)**; *associated with*, not predictive; CV is *inferred* from g-ratio, not measured. Matters for the H4/Exp-4 power analysis, which cites this d. |
| 4 | **Lazari et al. (2022)** | "Decreased qT2 relaxation" — wrong metric and direction; "strictly within the tract" overstates a correlational result; persistence. | Marker was **magnetization-transfer saturation (MT), which *increased***; qT2 was never measured. Effect was a brain–behaviour **correlation** (peak p_corr=0.013), not a significant group-mean change; the cluster *overlapped* the connecting tract. Only **one 24 h** post-timepoint — persistence beyond 24 h untested. |
| 5 | **Gibson et al. (2014)** | "Abolishes motor *learning*" via "blocking oligodendrocyte differentiation." | Readout was motor **function** (contralateral forelimb swing speed on gait analysis), not a motor-learning task; the block was **pharmacological/epigenetic (HDAC inhibitor TSA)**, not genetic. Causal link to motor function stands. |
| 6 | **Sussillo et al. (2016)** | Phrasing implies documented long-term human decay reduction. | **2 macaques**, preclinical; variability largely **simulated** (electrode dropping, withheld "stale" data); robustness significant in one monkey (p<0.01), not the other (p=0.45); the MRNN is fixed/non-adaptive, framed as complementary to recalibration. |
| 7 | **Pandarinath et al. (2017)** (LFADS) | Wrong year/volume/pages (2017, 14(12):1216–1224). | Actual: **2018, Nature Methods 15:805–815** (doi:10.1038/s41592-018-0109-9). Science unaffected. |
| 8 | **Zador et al. (2026)** | — | Verified: arXiv:2604.18637 (*NeuroAI and Beyond*), submitted Apr 2026; preprint, not yet peer-reviewed. |

---

## References

**Proposal references (verified).**

- Berman, S., West, K. L., Does, M. D., Yeatman, J. D., & Mezer, A. A. (2018). Evaluating g-ratio weighted changes in the corpus callosum as a function of age and sex. *NeuroImage*, 182, 304–313. doi:10.1016/j.neuroimage.2017.06.076 *(note: 2018, not 2019; and see note 1 — not the source of the CV formula)*
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. *NeurIPS 31*. arXiv:1806.07366
- Clark, I. A., Mohammadi, S., Callaghan, M. F., & Maguire, E. A. (2022). Conduction velocity along a key white matter tract is associated with autobiographical memory recall ability. *eLife*, 11, e79303. doi:10.7554/eLife.79303
- Fields, R. D. (2015). A new mechanism of nervous system plasticity: activity-dependent myelination. *Nature Reviews Neuroscience*, 16(12), 756–767. doi:10.1038/nrn4023
- Gibson, E. M., et al. (2014). Neuronal activity promotes oligodendrogenesis and adaptive myelination in the mammalian brain. *Science*, 344(6183), 1252304. doi:10.1126/science.1252304
- Hochberg, L. R., et al. (2012). Reach and grasp by people with tetraplegia using a neurally controlled robotic arm. *Nature*, 485(7398), 372–375. doi:10.1038/nature11076
- Lazari, A., Salvan, P., Cottaar, M., Papp, D., Rushworth, M. F. S., & Johansen-Berg, H. (2022). Hebbian activity-dependent plasticity in white matter. *Cell Reports*, 40(3), 110951. doi:10.1016/j.celrep.2022.110951
- Pajevic, S., Basser, P. J., & Fields, R. D. (2014). Role of myelin plasticity in oscillations and synchrony of neuronal activity. *Neuroscience*, 276, 135–147. doi:10.1016/j.neuroscience.2013.11.007
- Pandarinath, C., et al. (2018). Inferring single-trial neural population dynamics using sequential auto-encoders. *Nature Methods*, 15(10), 805–815. doi:10.1038/s41592-018-0109-9 *(proposal dated 2017)*
- Rossi, S., Hallett, M., Rossini, P. M., Pascual-Leone, A., et al. (2009). Safety, ethical considerations, and application guidelines for the use of transcranial magnetic stimulation in clinical practice and research. *Clinical Neurophysiology*, 120(12), 2008–2039. doi:10.1016/j.clinph.2009.08.016
- Rushton, W. A. H. (1951). A theory of the effects of fibre size in medullated nerve. *The Journal of Physiology*, 115(1), 101–122. doi:10.1113/jphysiol.1951.sp004655
- Stikov, N., et al. (2015). In vivo histology of the myelin g-ratio with magnetic resonance imaging. *NeuroImage*, 118, 397–405. doi:10.1016/j.neuroimage.2015.05.023 *(proposal cites 93:239–251)*
- Sussillo, D., Stavisky, S. D., Kao, J. C., Ryu, S. I., & Shenoy, K. V. (2016). Making brain–machine interfaces robust to future neural variability. *Nature Communications*, 7, 13749. doi:10.1038/ncomms13749
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 30*. arXiv:1706.03762
- Wandell, B. A., & Yeatman, J. D. (2013). Biological development of reading circuits is related to reading skills. *PNAS*, 110(36), 14576–14577. *(cited in proposal; peripheral to B2SS)*
- Wolpert, D. M., Diedrichsen, J., & Flanagan, J. R. (2011). Principles of sensorimotor learning. *Nature Reviews Neuroscience*, 12(12), 739–751. doi:10.1038/nrn3112
- Zador, A., Fellous, J.-M., Sejnowski, T., et al. (2026). NeuroAI and beyond: Bridging between advances in neuroscience and artificial intelligence. arXiv:2604.18637

**Additional work surfaced (not in the proposal).**

- Berman, S., Filo, S., & Mezer, A. A. (2019). Modelling conduction delays in the corpus callosum using MRI-measured g-ratio. *NeuroImage*. doi:10.1016/j.neuroimage.2019.116001 *(correct source for CV = k·v(g))*
- Drakesmith, M., et al. (2019). Estimating axon conduction velocity in vivo from microstructural MRI. *NeuroImage*, 203, 116186. doi:10.1016/j.neuroimage.2019.116186
- Asadi, A., et al. (2025). Non-invasive prediction of conduction velocities in the human brain from MRI-derived microstructure features at 7 Tesla. *bioRxiv*. doi:10.1101/2025.10.28.685017
- Abdollahzadeh, A., et al. (2024). Volume electron microscopy in injured rat brain validates white matter microstructure metrics from diffusion MRI. *Imaging Neuroscience*, 2, 1–20. arXiv:2310.04608
- (2024). Interplay between MRI-based axon diameter and myelination estimates in macaque and human brain. arXiv:2407.02227 (*Imaging Neuroscience*).
- Sedler, A. R., Versteeg, C., & Pandarinath, C. (2023). Expressive architectures enhance interpretability of dynamics-based neural population models. *NBDT*. arXiv:2212.03771
- Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019). Latent ODEs for irregularly-sampled time series. *NeurIPS 32*. arXiv:1907.03907
- Abbaspourazad, H., et al. (2024). Dynamical flexible inference of nonlinear latent factors and structures in neural population activity (DFINE). *Nature Biomedical Engineering*, 8(1), 85–108. doi:10.1038/s41551-023-01106-1
- POSSM (2025). Generalizable, real-time neural decoding with hybrid state-space models. *NeurIPS 2025*. arXiv:2506.05320
- Willett, F. R., et al. (2023). A high-performance speech neuroprosthesis. *Nature*, 620. PMID:36711591
- CTNet (2024). A convolutional transformer network for EEG-based motor imagery classification. *Scientific Reports*, 14, 20237. PMC11364810
- Jiang, W.-B., et al. (2024). LaBraM: Large Brain Model for learning generic representations with tremendous EEG data in BCI. *ICLR 2024*.
- SPINT (2025). Spatial Permutation-Invariant Neural Transformer for consistent intracortical motor decoding. arXiv:2507.08402
- TCFormer (2025). Temporal convolutional transformer for EEG-based motor imagery decoding. *Scientific Reports*.
- (2022). Homeostatic coordination and up-regulation of neural activity by activity-dependent myelination. *Nature Computational Science*, 2(10), 665–676. doi:10.1038/s43588-022-00315-z
- (2023). Oligodendrocyte-mediated myelin plasticity and its role in neural synchronization. *eLife*, 12, e81982. doi:10.7554/eLife.81982
- (2020). Activity-dependent myelination: a glial mechanism of oscillatory self-organization in large-scale brain networks. *PNAS*, 117(24), 13227–13237. doi:10.1073/pnas.1916646117
- (2024). Temporal dynamics of white and gray matter plasticity during motor skill acquisition. *Cerebral Cortex*, 34(8), bhae344. doi:10.1093/cercor/bhae344
- (2023). Motor potentials evoked by transcranial magnetic stimulation: interpreting a simple measure of a complex system. *The Journal of Physiology*, 601(14), 2837–2851. doi:10.1113/JP281885
- (2023). Targeted modulation of human brain interregional effective connectivity with spike-timing-dependent plasticity. *Neuromodulation*, 26(4), 745–754. doi:10.1016/j.neurom.2022.10.045
- (2023). Can we manipulate brain connectivity? A systematic review of cortico-cortical paired associative stimulation effects. *Clinical Neurophysiology*, 154, 169–193. PMID:37634335
- (2024). TMS of primary motor cortex elicits an immediate transcranial evoked potential (i-TEP). *Brain Stimulation*, 17(3). doi:10.1016/j.brs.2024.05.003
- (2025). Mapping brain lesions to conduction delays: the next step for personalized brain models in multiple sclerosis. *Human Brain Mapping*, 46(7), e70219. doi:10.1002/hbm.70219

*Author lists abbreviated where long; a few very recent entries are preprints — verify against the final peer-reviewed version before formal citation.*
