---
name: cortex-phase10-results
description: "Phase 10 (2026-08-31, axes now MNIST + split-CIFAR-10 grayscale): (1) static price of rotation solved by a RELATIVE novelty gate (fast/slow error EMA ratio, ACh-like) -- static 95.0 (= none; rotation on 1-17%) with sequential 90.3-90.9 kept, while unit-level pressure gives only a trade-off curve and absolute thresholds do not transfer across data sets; (2) K=200 gap narrowed 81.8->84.3 by replay noise + pressure bursts (night 86.9 keeps a ~2.6 edge); (3) CIFAR transfers and amplifies v9 (refr 27.1 > BP+ER 25.1 > night 23.8 > unmasked 20.9) and the static axis REVERSES (rotation +2.5..+6.5 when underfitting) but the novelty gate misfires there (turns rotation off on a stream that never becomes predictable); spiking re-measure: v9 net 91.6% @ 119 nJ at T_s=8 (1/20.7 of dense rate)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-31T05:36:55.501Z
---

Ran 2026-08-31 daytime.  New in `run_local`: masks `refr_press` (rest iff per-unit Process-S >=
theta_r, S += firing fraction, rest discharges), `refr_frac` (top-frac_r by S), `refr_ach`
(plain refractory gated by absolute surprise-EMA threshold theta_s), `refr_nov` (gated by
fast-EMA >= beta_nov * slow-EMA of the top-down error); cfg overrides `nrem_gain`,
`replay_noise`; `load_data(cfg)`/`load_cifar10_gray()` (32x32 luminance, `dataset: "cifar"`),
`ach_frac` logged; `tmp/bio/p10_spike.py` re-measures spiking on `run_seq.LAST_NET`.
w512s5, K=1000, bw16, br16, 3 seeds (nov rows 2-3 seeds), split protocol as phase 9.

1. 10A static price: unit-level pressure is a PURE TRADE-OFF (static 91.6->94.0 as theta_r
   0.5->4 while sequential 88.8->80.7; channel 0.43->0.20); refr_frac saturates into "refractory
   with memory" (units with S>0 fewer than the quota at both 25% and 50% -> identical runs,
   seq 91.0 +-0.6 best of grid, static 92.0 unchanged).  Absolute ACh threshold does not
   transfer (static MNIST converged error > any theta the 2-class tasks dip under: ach_frac=1.0
   on static at theta 0.15/0.25 vs 0.019 at 0.40; CIFAR 0.95).  RELATIVE novelty (fast 0.1-EMA
   >= beta * slow 0.002-EMA) works on both MNIST axes with one setting: beta 1.5 -> static
   95.0 +-0.4 (= none 95.0 +-0.2; rotation on 1.4-17%), sequential 90.3 +-0.4 vs refractory
   90.8 +-0.1; beta 1.1: seq 90.9 +-0.6 (best) / static 93.8 +-0.5; beta 1.3: 90.2 / 94.4 (3 seeds).
2. 10B K=200: replay noise sigma 0.2 + pressure bursts (pb16_n02) 84.3 +-1.0 @ 2,763 replay
   batches, forgetting 0.08; gain-1 trickle 83.4 +-0.2; noise alone 82.6; baseline br16
   81.8 +-2.9; night 86.9.  Diversity is the main lever, over-plasticity second; the night
   keeps ~2.6 (offline consolidation still better per unique sample at tiny K).  At K=1000
   noise is harmless (90.4 +-0.2).
3. 10C split-CIFAR-10 (grayscale 1024-d, same protocol): sequential refr_br16 27.1 +-1.7 >
   BP+ER 25.1 +-1.5 > night 23.8 +-1.2 > unmasked 20.9 +-1.8 > no-buffer 16.0-16.5; static
   none 22.8 +-1.3 < refr 25.3 +-1.0 < rp4 29.3 +-1.8: on hard/underfit data the rotation is a
   REGULARISATION GAIN (+2.5..+6.5), the "static price" is a property of the easy-data regime.
   CIFAR task overlap is ~0.9-0.98 (codes barely separate) yet isolation still wins.
   CAVEAT: the novelty gate misfires on CIFAR (nov13 seq 19.1 +-1.1, rotation off because
   fast/slow -> 1 on a stream that never becomes predictable, while rotation there helps
   always).  Default = rotation always on; the novelty gate is the tool for easy/learnable
   streams; a gate keyed to learning PROGRESS rather than error level is the open follow-up.
4. Spiking re-measure (static MNIST, T_s=8): v9 refractory net 91.6% @ 118.8 nJ, control net
   93.7% @ 118.7 nJ, vs dense rate 2461 nJ (20.7x) and event-driven rate 516 nJ (4.3x);
   accuracy now DECREASES beyond T_s=8 on the refractory-trained net (16: 90.4, 32: 89.8),
   unlike phase 4 -- threshold calibration interacts with rotation-trained weights.

**Why:** closes the three v9 open items: the static price has a data-set-independent solution
(relative novelty gating) on learnable streams, the K=200 gap is mostly replay diversity, and
the local-sleep result survives (and grows) on harder data.
**How to apply:** v10 default = refractory local sleep br16 + refr_nov beta 1.5 on streams that
converge, plain refractory on hard/underfit streams; K<=200 add replay_noise 0.2 + pressure
bursts; energy story cites 91.6% @ 119 nJ.  Open: progress-keyed gate, CIFAR with features
(conv front-end or patches) instead of raw luminance, K=200 remaining 2.6.  Related:
[[cortex-phase9-results]], [[cortex-lit-review-2026-08]].
