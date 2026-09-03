---
name: cortex-phase19-results
description: "Phase 19 (2026-09-03, adaptive KP controller): lambda_l = rho*eta*EMA|g_l|/|W_l| (rho=0.25, ONE dimensionless value everywhere; waking-only EMA; W/B share lambda so KP alignment intact; realized lambda logged per layer as kp_eff). Cross-dataset verdict: matches sequential records within noise (MNIST 92.4, CIFAR 28.2, cfeat 41.0) and BEATS fixed lambda on every static axis (MNIST +0.7 -> 95.4, CIFAR +3.5 -> 31.1, cfeat +1.9 -> 43.8); pure controller (no target inflation) = 3.5+-0.4 on raw CIFAR-100, the best local number there (2.5x the manual kp=1e-4+tgt3 rescue). Realized lambda self-differentiates by regime/axis/layer (MNIST seq ~9e-5, static ~9e-4, CIFAR static up to 3.3e-3, c100 ~1.5e-3) -- per-dataset kp tuning is over. ARTIFACT: tgt_scale inflates the gradient estimate -> lambda hits the 1e-2 clip -> starves again; controller and target-scaling must NOT compose (controller supersedes tgt_scale). Design = the project's self-referenced-signal principle (novelty/progress gates) applied at the synapse level"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-03T01:33:39.229Z
---

Implementation (cortex.py KP block in local_update): per-layer scalar EMA (0.02) of mean |gW|
on waking unmasked calls; kp_l = min(0.02, rho * eta * g_ema[l] / mean|W_l|); same kp_l shrinks
W_l and B_{l-1} (alignment preserved); kp_eff EMA logged into run_seq results.  Knob:
cfg kp_adapt=rho (None = fixed kp_decay).  Configs g19_kad_{mnist,cifar,cifarf,c100,c100f}
(+_static, +c100 puretgt variants), s10 substrate, refractory local sleep, SGD.

Verdict table (3 seeds, vs fixed-kp references):
- MNIST:  92.4+-0.5 / 95.4+-0.2  vs 92.7+-0.3 / 94.7+-0.3   (seq tie, static +0.7)
- CIFAR:  28.2+-1.3 / 31.1+-1.1  vs 29.5+-0.8 / 27.6+-0.4   (seq ~tie, static +3.5)
- cfeat:  41.0+-1.6 / 43.8+-0.4  vs 41.7+-1.2 / 41.9+-1.2   (seq tie, static +1.9)
- c100 pure: 3.5+-0.4 vs manual rescue 1.4+-0.5 (and vs floor-grid local ~1) -- partial
  bootstrap by pure lambda adaptation; c100+tgt3 = 1.1 (the overshoot artifact).
Realized lambda: regime/axis/layer-differentiated (see description).  Static-axis wins mean
fixed lambda UNDERREGULARISED thick-signal regimes all along.

**Why:** turns the oldest tuning annoyance of the project (per-dataset kp: 1e-2 tabular /
1e-3 MNIST / 1e-4 c100) into a mechanism -- the correct decay is set by supervision
statistics, and a constant decay-to-drive ratio recovers it everywhere; phase-18's boundary
partially lifts as a corollary.
**How to apply:** candidate new default: kp_adapt=0.25 replacing fixed kp_decay AND
tgt_scale (never compose with target inflation).  Before promoting to manuscript default:
6-seed hardening + tabular (run_bio SUPPORT2, does it recover ~1e-2?) + interaction checks
with night/anchor paths.  Related: [[cortex-phase18-results]], [[cortex-phase17-results]],
[[cortex-phase6-results]].
