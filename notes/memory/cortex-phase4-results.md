---
name: cortex-phase4-results
description: "Phase 4 of the MLP-Cortex side project on MNIST (wide core, v4 base): rate-to-spike inference gives ~5x energy at T_s=8 with <0.5 pt loss; PC relaxation needs only T=3-5 steps (T=1 diverges); distance-dependent sparse connectivity beats random at equal synops and structural plasticity adds +1.2 pt at 10% density"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-28T08:55:20.505Z
---

Ran 2026-08-29 on the v4 base (no transport + dale_fb + bounded error + Dale, 256-128, adam_eps 1e-3,
k-WTA 10% unless noted), MNIST, 30 epochs, 3 seeds. Code: `src/models/spiking.py` (4A),
`CortexNet(conn_density, conn_mode, conn_lambda, regrow, input_shape)` + `structural_plasticity()`
(4C), `T`/`gamma` config keys in `src/run_mnist.py` (4B). Results in `results/bio/mnist/summary.csv`
and the per-run pickles (`spike` dict per T_s, `conn_density`).

- 4A spiking inference (Bernoulli input spikes, soft-reset IF, k-WTA as per-step spike budget,
  standardisation + gain folded into W1; spike carries `thr` units; energy 0.9 pJ/AC vs 4.6 pJ/MAC):
  dense rate 97.9% / 1080 nJ -> T_s=8 97.4% / 206 nJ, T_s=16 97.8% / 419 nJ. kwta10: 97.1% ->
  96.0% @ 198 nJ. Below T_s=8 accuracy falls off a cliff (T_s=4: 48-87%). Event counts equal the
  dense MAC count at T_s=8 (input spikes dominate), so the 5x is entirely AC-vs-MAC plus the
  event-driven input; beyond T_s=16 spiking costs MORE events than the rate net.
- 4B relaxation depth (gamma*T = 2 kept): T=1 diverges (18%), T=3 96.8%, T=5 97.0%, T=10 97.2%,
  T=20 97.0%; training time 4.0 / 5.5 / 9.2 / 21 min. Prospective configuration needs a few steps,
  not twenty: T=5 is the new default (4x cheaper training, same accuracy).
- 4C connectivity (masks on W1, W2 and the feedback twins; readout dense): at 30% density
  distance-dependent 96.4% vs random 95.4%; at 10% 94.2% vs 92.5%; 10% + structural plasticity
  (prune weakest 5%/epoch, regrow distance-biased) 95.3%. Synops 56k / 19k vs 187k dense. Locality
  is worth ~1-2 pt at equal cost, and regrowth recovers most of the 10%-density loss.

**Why:** these settle three roadmap questions: spikes pay off only in a narrow T_s window with
event-driven input; deep relaxation is unnecessary; a self-organised (local + plastic) graph is a
real gain, not decoration.
**How to apply:** v5 base = v4 + T=5 + conn_density 0.3 dist (or 0.1 + regrow) ; spiking evaluation
at T_s=8-16 for the energy axis. Open next: train-time spiking (surrogate-free, e.g. spike-based
PC), two-compartment burst multiplexing, sleep phase as FF negative data, SGD instead of Adam.
Related: [[cortex-v1-results]].
