---
name: cortex-phase13-results
description: "Phase 13 (2026-09-01 evening, bout-committed gate, 3 seeds): refr_block = prog trigger + minimum on-duration M (Saper flip-flop bouts); M=2048 -> 92.1+-0.6 seq / 94.2+-0.4 static, WEAKLY DOMINATES always-on SGD (92.0/93.4) -- the per-batch gate's failure was sparse trigger (12% duty) + flicker, bouts convert it into regime-asymmetric contiguous coverage (94% seq / 63% static); M=512 91.7/94.0 (duty .61/.38), M=4096 refund gone (93.4, duty .94); silent static ceiling 96.0 still out of reach; new sequential default operating point"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-01T14:41:17.036Z
---

Ran 2026-09-01 evening, immediately after the 12b/12c ablation found the per-batch prog gate
weakly dominated under SGD.  Implementation: mask="refr_block" in run_seq.py -- the 11A progress
trigger (beta_nov=1.3, gamma_p=0.5) re-arms a counter `block` (=M batches) on every firing;
rotation runs while the counter is positive (Saper 2005 sleep-wake flip-flop: consolidated
bouts, flicker = pathology).  Configs g13_sgd_block{512,2048,4096}_w512s5 (+static_), SGD
eta=0.02, w512s5, K=1000, br16 cad1.

Results (3 seeds; duty = ach_frac):
- M=512:  seq 91.7 +-0.4 (duty .61) / static 94.0 +-0.5 (duty .38)
- M=2048: seq 92.1 +-0.6 (duty .94) / static 94.2 +-0.4 (duty .63)  <- NEW DEFAULT seq point
- M=4096: seq 92.0 +-0.7 (duty 1.0) / static 93.4 +-0.1 (duty .94)  (= always-on, refund gone)

Reading: the per-batch gate's trigger opens on only 12% of batches (sparse), and closes
mid-task the moment mastery is reached -- the batches where continuous rotation earns its +3.0.
Bout commitment amplifies the same sparse trigger into long contiguous episodes with
regime-asymmetric coverage; M=2048 restores the sequential ceiling AND refunds 0.8 static,
weakly dominating always-on.  The silent static ceiling (96.0) stays out of reach: rotation in
bouts keeps a residual price -- the gate buys regime-adaptivity, not a free lunch.

**Why:** answers the 12b open question ("can any gate make a third operating point under SGD?")
positively, with a mechanism that is MORE biological (bout structure), not less -- and revises
12b's "no third point" to "no third point *per batch*".
13b OFF-side refractory (`block_off`; triggers ignored for M_off batches after a bout expires)
= CLEAN NULL: M_off<=1024 never engages on static (bit-identical trajectories -- a bout expires
only after its trigger has been quiet >=2048 batches, so static re-triggering is not flicker);
M_off=4096 lowers static duty 0.63->0.53 with NO static gain (94.0 +-0.7) and monotone seq
erosion (92.1->91.9->91.7->91.0 at OFF 0/512/1024/4096; long OFF can mask a task switch, seed1
89.8).  With the M=512 cell (duty 0.38 -> 94.0), the residual static price is located in the
rotation of settled coalitions ITSELF, not in trigger frequency -- the flip-flop's functional
half here is the ON side.
**How to apply:** sequential default = refr_block M=2048 + SGD (92.1/94.2), NO off-refractory;
pure static best = SGD+silent (96.0).  Manuscript updated (Table 1 row, reading passage +
OFF-null append, abstract, negative results, Saper 2005).  Open: mastery-annealed rotation
depth or utility-weighted benching (attack the settled-coalition price directly), bout gate on
CIFAR/feature regimes.  Related: [[cortex-phase12-results]], [[cortex-phase11-results]],
[[cortex-phase10-results]].
