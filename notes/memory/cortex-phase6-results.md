---
name: cortex-phase6-results
description: "Phase 6 of the MLP-Cortex side project (v6 base = v5 + burst gate/baseline): single-phase burst sweep (T=0) matches 5-step relaxation at 1/3 the training time; multiplicative burst coding fails; spike-count-noise training at T_s<=16 loses 11-16 pt; weight mirror aligns B but the KP decay's real job is regularisation (SUPPORT2 needs decay 1e-2 regardless); SGD cosine no help"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-29T03:36:45.860Z
---

Ran 2026-08-29. v6 base = v5 (NT + dale_fb + bounded + Dale, 256-128, k-WTA 10%, adam_eps 1e-3,
T=5, 30% distance connectivity) + burst_gate + burst_baseline. MNIST 30 epochs x 3 seeds; SUPPORT2
under the report protocol (64-32 core, kwta/homeo off). New code: `CortexNet(sweep, burst_mult,
spike_train, mirror)`, `sweep_errors()`, `mirror_phase()`, `training` flag, cosine eta in run_mnist.

MNIST (acc / noise-1.0 acc / train min / spiking T_s=8):
- v6_base 96.3 / 92.7 / 5.9 / 95.3;  v5_base 96.5 / 93.0 / 5.9 / 95.9
- 6A v6_sweep (T=0: one forward pass + one top-down burst sweep, no settling): 96.5 / 93.3 / 1.8 /
  96.0 -> equal or better on every axis at 1/3 the cost. Relaxation is not needed at all on this
  architecture (T=20 -> 5 -> 0 across phases). NEW DEFAULT.
- 6A burst_mult (plasticity ~ event rate x burst deviation): 78.8 with relaxation, 90.9 with sweep.
  Multiplicative coding starves low-rate units; the relaxation multiplies the distortion each step.
  Gating (a>0) is the right form, multiplication is not.
- 6B spike-count noise in training (Poisson(a*T_s)/T_s on every hidden rate): T_s=8 -> 80.1,
  T_s=16 -> 85.8 (-16 / -11 pt). Those nets do slightly better in the spiking domain than the rate
  domain (they adapt to spike statistics) but the noise budget is too small; train-time spiking
  needs eligibility traces / longer integration windows, not raw count sampling.
- 6C on MNIST: no decay + no mirror 96.3 (KP cos 0.89/0.95), + mirror 96.3 (cos 1.0), + decay 1e-3
  96.1. Accuracy identical; the decay buys noise robustness (92.5-92.7 vs 89.9-90.0 at sigma 1.0).
- 6C on SUPPORT2 (CV AUC): decay 1e-2 (any combination) 0.895; decay 1e-3 0.868; decay 0 0.876;
  decay 0 + mirror 0.878. Mirroring aligns B but does NOT replace the decay: the decay's job on
  SUPPORT2 is weight regularisation over 17k updates, not alignment. So: keep dale_fb + mirror (or
  nothing) for alignment, and treat KP/weight decay as an honest regulariser to set per data set
  (1e-2 tabular, 1e-3 MNIST). The "per-data-set decay" tension is resolved as "it is a weight decay".
- 6D SGD lr 0.03/0.05 + cosine: 93.4 / 94.4, below lr 0.02 constant (95.2). Adam gap stays ~1.3 pt.

**Why:** settles the single-phase question (yes), the burst-coding form (gate, not multiply), the
alignment-vs-regularisation confusion around KP decay, and rules out naive train-time spiking.
**How to apply:** v7 base = v6 + sweep=True (T=0), keep burst gate+baseline, weight decay per data
set, Adam eps 1e-3. Open: train-time spiking with eligibility traces; two-compartment apical/basal
dynamics with STP demultiplexing (the gate/baseline/sweep trio is the abstract version of it);
generative path to the input before any negative phase. Related: [[cortex-phase5-results]],
[[cortex-phase4-results]], [[cortex-v1-results]].
