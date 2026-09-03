---
name: cortex-phase20-results
description: "Phase 20 (2026-09-03, depth): A) ladder d3/d4/d5 MNIST s10 -- fixed lambda CLIFFS where signal thins (static dead at chance from d4, seq bimodal-dead at d5 55.5+-39.3), controller keeps every cell alive (seq 90.6->90.0->87.6, static 95.6->95.3->91.8); depth = phase-18 class-thinning boundary on a 2nd axis; local+ctrl >= same-width BP+ER at all depths with 1/3 the forgetting. B) ResNet-style skip synapses (local product, KP feedback twin, shared kp_l, same pre-or-post-asleep replay mask): d5+skip returns to the d2 record on BOTH axes (91.5+-0.4/96.2+-0.3) and d4+skip static 96.3+-0.2 = NEW PROJECT STATIC BEST; skip alone rescues fixed lambda too (d5 91.3/93.6) -> depth collapse = ERROR-PATH ATTENUATION, not decay mis-tuning; controller margin is static-side (+2.6, seq ties within noise). Two rescue routes: architectural (bypass restores signal) + adaptive (controller stops the decay race); combo best"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-03T07:33:49.777Z
---

20A configs `g20_d{3,4,5}_{kad,fix}[_static]`, `g20_d*_bp_er` (45 runs); ladders (512,256,128),
(+128), (+128,128), af=0.10, SGD .02, refractory cad1 br16, kad = kp_adapt 0.25 (v2 formula).
Full table (seq/static): d3 fix 91.4/93.6, kad 90.6/95.6, bp 88.9; d4 fix 89.3/9.7-DEAD,
kad 90.0/95.3, bp 88.8; d5 fix 55.5+-39.3-BIMODAL/10.1-DEAD, kad 87.6+-1.7/91.8+-0.8,
bp 88.1. Static dies first under fixed lambda (10-way simultaneous supervision = thin;
2-way seq tasks = thick, survive to d5). Realized lambda: starved middle layers 3e-6.
make_bp generalised to any depth.

20B configs `g20b_d{4,5}_skip[_static]`, `g20b_d5_fixskip[_static]` (18 runs). Implementation
(cortex.py): S[l] (sizes[l], sizes[l-2]) for hidden l=3..L-1, dense (bypass axons long-range),
Bs[l] feedback twin learned from the same gated product, shared layer-l kp_l shrink, Dale by
sign[l-2]; forward adds a[l-2]@S[l].T to basal drive; sweep_errors adds Bs-path from eps[l+2];
local_update takes skip_mask (frozen during replay if absent -- isolation-safe default);
run_local builds skip masks by the same pre-or-post-asleep rule. Verified: frozen-default 0.0,
Dale violations 0, drift under strict re-forward test same order as no-skip baseline (the
re-forward test over-rejects: k-WTA re-selection + asleep-bias moves are inherent to the
established system, not a skip leak).

**Why:** answers "does the system survive stacking" -- yes: controller alone makes depth
graceful, skips + controller make it FREE (d5 = d2 record both axes). Fixskip probe splits
the mechanism: seq AND static depth deaths are error-path attenuation (shortcut rescues both
at fixed lambda), so architecture and controller are complementary, not redundant.
**How to apply:** deep variants: always kp_adapt + skip=True. Untried: skip x c100
(class-thinning axis has no architectural rescue candidate yet), skip x conv, anchor/bout x
skip. d4+skip static 96.3 is the number to beat. Related: [[cortex-phase19-results]],
[[cortex-phase18-results]], [[cortex-phase17-results]].
