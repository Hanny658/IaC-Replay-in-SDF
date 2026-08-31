---
name: cortex-phase5-results
description: "Phase 5 of the MLP-Cortex side project on the v5 base (v4 + T=5 + 30% distance connectivity, MNIST): burst-gate and burst-baseline are free; SGD costs ~1.3 pt vs Adam; a weak dream/negative phase costs 1.2 pt and buys no robustness, a strong one is destructive; v5 spiking at T_s=8 = 95.9% for 59 nJ (18x below dense rate)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-28T14:16:36.806Z
---

Ran 2026-08-29. v5 base = NT + dale_fb + bounded + Dale, 256-128, k-WTA 10%, adam_eps 1e-3, T=5
(gamma 0.4), conn_density 0.30 distance-dependent. 30 epochs, 3 seeds, MNIST. New in code:
`CortexNet(burst_gate, burst_baseline, dream_batches, dream_eta)`, `dream()`, `local_update(sign, first)`,
`noise_robustness()` in run_mnist.py (accuracy under Gaussian input noise sigma 0.5/1.0, stored as
`noise_acc` in every phase-5 pickle). Results in `results/bio/mnist/summary.csv` + pickles.

| config | acc | noise 0.5 | noise 1.0 | note |
| mlp_bp_w5 | 97.7 | 97.1 | 93.9 | BP reference, dense |
| v5_base | 96.5 ±0.2 | 95.9 | 93.0 | spiking T_s=8: 95.9% @ 59 nJ (dense rate 1080 nJ) |
| v5_regrow (10% + regrow) | 95.3 | 94.6 | 90.9 | spiking T_s=8: 93.7% @ 20 nJ |
| v5_sgd02 (heavy-ball, lr .02) | 95.2 | 94.5 | 91.5 | SGD lr .01: 93.6 |
| v5_bgate / v5_bbase / v5_burst | 96.3 / 96.2 / 96.3 | 95.5 | 92.7 | both burst ingredients free |
| v5_dream (20 dreams/night, 0.1 eta) | 95.3 | 94.6 | 91.1 | no robustness gain |
| v5_dream_strong (0.3 eta) | 89.3 | 87.8 | 83.5 | destructive |

Readings:
- The v5 base is 1.2 pt under BP at ~1/4 of the training compute and, spiking, ~18x less inference
  energy; robustness gap to BP is ~1 pt at both noise levels (bio constraints did not buy robustness).
- Burst multiplexing ingredients (no event -> no burst; burst-rate baseline) cost nothing, so a
  full two-compartment BurstCCN neuron can be built on them without paying for those properties.
- Dropping Adam for heavy-ball SGD costs ~1.3 pt at lr 0.02 (per-synapse normalisation still
  matters); not tuned beyond two lrs.
- Sleep-as-negative-phase in its simplest form (anti-Hebbian on top-down dreams) is a net loss:
  -1.2 pt accuracy, no robustness gain, and it becomes destructive at 0.3 eta. The FF/wake-sleep
  idea needs a real generative pathway to the input (a feedback B0) and a proper contrastive
  objective before it can be judged.

**Why:** closes the phase-5 questions: which roadmap items are free (burst), which have a known
price (SGD), and which are not yet worth carrying (dreams as implemented).
**How to apply:** v6 base = v5 + burst_gate + burst_baseline (free, more cortical). Next: full
two-compartment neuron with apical/basal integration and STP demultiplexing; train-time spiking;
a generative feedback path to the input before revisiting the negative phase; SGD lr sweep with
a schedule. Related: [[cortex-phase4-results]], [[cortex-v1-results]].
