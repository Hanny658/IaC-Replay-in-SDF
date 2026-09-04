---
name: cortex-phase22-results
description: "Phase 22 (2026-09-04, is 'stateless SGD' literal? momentum=0 test, 66 runs): momentum 0 with the default replay step (3*eta) is CHANCE at every eta (hidden activity collapses, dim [.,0,0]); probe: waking path learns fine at matched effective step, the collapse comes from the REPLAY path -- the velocity's ~10x noise averaging of 16-sample replay micro-batches is what heavy-ball provides; without it eta=0.2 replay diverges (NaN), eta=0.02 starves layer 2. Rescaling the replay gain (eta=0.2, nrem_gain=0.3) makes pure SGD viable but worse: 89.9+-0.3/90.1+-0.2 (-2.8/-4.6); +controller 91.7/91.4 (-1.0/-3.3); landscape fragile. VERDICT: wording, not a new default -- manuscript 'stateless' -> 'heavy-ball SGD (one per-synapse velocity, no second-moment state, advanced only inside the mask)'; 'the velocity is the least state that keeps guarantee and accuracy'; Table 1 gains a momentum-0 row"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-04T02:40:13.452Z
---

Configs g22_sgd0_e{02,05,1,2,4}[_static] (default replay step; all chance, killed after 4),
g22b_sgd0_e{1,2}_g{01,03,1}[_static] + g22b_sgd0_e2_g03_kad[_static] (36 runs, OMP=4).
Knob: cfg momentum (run_local -> CortexNet(momentum=...)); nrem_gain already a knob.
Waking step in run_local = eta*16/256 (raw), replay step = eta*nrem_gain: heavy-ball
amplifies both ~10x in steady state; momentum 0 needs eta x10 for the waking side but the
replay side must then be cut x10 (gain 0.3) or it diverges.

**Why:** the paper's central optimiser claim was phrased as literal statelessness; the
velocity buffer IS per-synapse state and, mechanistically, is what makes isolated micro-batch
replay survivable.  Now stated precisely.
**How to apply:** never claim 'no state'; if a truly stateless variant is ever wanted, re-tune
eta AND replay gain jointly (they decouple without momentum) and expect -3..-5 points.
Related: [[cortex-phase12-results]], [[cortex-phase14-15-results]],
[[cortex-review-round-2026-09]].
