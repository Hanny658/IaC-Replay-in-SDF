---
name: cortex-lit-review-2026-08
description: Literature round done 2026-08-30 before phase 9 (XdG/endogenous gating, Bazhenov sleep line, CLS-ER/DualNet slow system, Cayco-Gajic pattern-separation checklist, competitor scan) -- nobody has done unit-level use-dependent (refractory) gating or local sleep in an ANN; Tononi/Cirelli Nat Neurosci 2026 shows induced ON/OFF alternation (not tonic silencing) in awake cortex relieves local sleep pressure and rescues memory; CLS-ER EMA twin raises per-replay value but does not cut replay count; DG recipe = few-input sparse random expansion + adaptive threshold, judged by dimensionality/overlap
metadata:
  type: project
---

Local copies of all papers (txt) were kept in a temporary job directory (may vanish).

1. Gating. XdG (Masse 2018 PNAS): random fixed gates, 80-86.7% of units off per task, task ID given
   externally (PFC-like). LXDG (Tilley, Miller, Freedman ICLR 2023): gates are outputs of trainable
   nets from the input, sparsity/recall/orthogonality losses; task-free at test, task-aware in training
   (stores prior-task input samples). Active Dendrites (Iyer 2022): context = input prototype, on-the-fly
   clustering variant. Sparse-recurrent DFC (Lassig/Grewe 2023): task-free, sparse set chosen by
   feedforward + top-down error + growing WTA. McKee/Miconi (Astera, arXiv 2604.24637, 2026): unit masks
   found by one gradient step + lateral smoothing + kWTA, same procedure recovers the mask at test.
   ALL gate by input/task identity; none by the unit's own use history (our refractory rule) -> open.
2. Bazhenov line: SRC = ANN->Heaviside, Poisson noise from mean pixel intensity, Hebbian inc/dec, global,
   offline, after each task (Tadros 2022 NatComm; Kubo/Delanois/Bazhenov 2025 EP; A. Bazhenov 2026 one
   SRC after 2-5 tasks). Finest granularity = Golden 2022 PLOS CB 100-cycle alternation, still global
   with input silenced. No local/daytime sleep. SRC alone < rehearsal; rehearsal + SRC best (they sell
   SRC as cutting stored data). Sleep decorrelates/sparsens old-task codes (same metric as our overlap).
3. Biology for local sleep: Driessen, Squarcio, Tononi, Cirelli, Nat Neurosci 2026 (bioRxiv
   2025.10.04.680459): optogenetic ON/OFF induction in one hemisphere of awake mice for 30 min ->
   ipsilateral SWA drop in later NREM, lower GluA1/pS845, bilateral induction during 1 h SD rescues
   memory. Equal tonic firing reduction (halorhodopsin) does NOT work: alternation every few 100 ms is
   required -> matches 8B (silent mask < refractory rotation).
4. Slow system: CLS-ER = working net + plastic/stable EMA twins (alpha 0.99-0.999, stochastic update
   rate rS 0.04-0.1 << rP), consistency MSE to the twin with higher true-class softmax, inference from
   stable twin; replays a buffer batch EVERY step (2 extra forwards) -> raises per-replay value (largest
   gain at buffer 200) but does not reduce replay volume. DualNet: slow learner = Barlow-Twins SSL on
   the buffer, 3 SSL iters per labelled batch, in the background; costlier, not cheaper. SIESTA (2023):
   wake = backprop-free readout update, sleep = budgeted offline rehearsal on quantised latents.
5. Pattern separation (Cayco-Gajic & Silver 2019 Neuron; Cayco-Gajic, Clopath, Silver 2017 NatComm):
   expansion + nonlinear mixing raises dimensionality but does not decorrelate; sparse synaptic
   connectivity (GC 4 inputs, KC ~7) is what removes shared-input correlations (Nsyn=16 is worse than
   raw input; ~4 gives up to 8x learning speed-up); high threshold (3 of 4 inputs) decorrelates but
   over-sparsening shrinks coding space; feedback inhibition = adaptive threshold keeping sparseness
   fixed across input density; DG compensates dense input with strong lateral inhibition (WTA),
   adaptive excitability, dendritic thresholds, neurogenesis; judge by dimensionality (participation
   ratio), not sparseness. "Expansion and correlations, not sparse activity, are the major determinants."
   Competitors on the no-replay static axis: FlyModel (Shen 2021: x40 expansion, ~6 of 50 inputs, top-l,
   partial freezing), SDM (Bricken ICLR 2023: top-k, no bias, L2 weights, GABA switch), HiCL (AAAI 2026:
   DG top-5% + cosine prototypes, needs task ID in training).
6. Scan: no ANN work on sleep-pressure-gated replay or local sleep found (SleepGate 2026 = entropy
   trigger for an LLM cache; Wake-Sleep Consolidated Learning 2024 and Robinson 2022 NREM+REM+
   downscaling are scheduled, global).

**Why:** decides phase 9 framing: 8B's refractory local sleep and 7C's controller occupy an empty
niche; the literature points to code separation (few-input DG front-end) as the lever for the 39x cost.
**How to apply:** proposed order: 9 = DG front-end (C-lite, checklist above) + refractory-on-static
check (B) in parallel, then pressure-gated local replay (A) on the better substrate; optional CLS-ER
stable twin as a consolidation target for K=200. Related: [[cortex-phase8b-results]],
[[cortex-phase7c-results]].
