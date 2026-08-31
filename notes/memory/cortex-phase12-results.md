---
name: cortex-phase12-results
description: "Phase 12 (2026-08-31 night, reviewer-inspired controls, 3 seeds): (1) mirror isolation (protect-the-past, the null-space direction, same machinery) collapses to 81.3+-2.9 vs ours 90.8 and even below unmasked ER 86.9 -- the protected-direction inversion is a 9.5-pt empirical effect, now in the paper; (2) stateless heavy-ball SGD inside the isolation beats masked Adam: 92.0+-0.7 seq (NEW PROJECT BEST) / 93.4+-0.2 static (unmasked SGD control 84.1+-4.1); SGD gain needs every-batch cadence (sgd+cad2_br64 91.3 = adam); (3) soft rotation (gain halving) loses on both axes (88.4/86.0) -- all-or-none OFF beats turned-down gain, third match to Driessen 2026; external review processed: 11 verified refs added, one hallucinated ref (arXiv 2608.26720) caught and excluded"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-31T13:54:52.977Z
---

Ran 2026-08-31 night after an external novelty review (`MLP-Cortex/report/critic.md`).  Review
verdict accepted: repositioned the paper from "refractory rotation is new" to "concurrent,
exactly isolated replay construction + rotation as channel-widener"; added verified refs
(Kaski-Kohonen 1994, Maeda-Miyajima 1999, Shen AAAI 2024, Abbasi CoLLAs 2022, GPM, Adam-NSCL,
OGD, CLNP, PackNet, Ross 2012 EWMA, Klasson 2023); caught one hallucinated reference in the
review (arXiv:2608.26720 "sLoTh" does not exist; title/description mismatch) -- not cited.

New code (`run_seq.py`): mask="mirror" (waking update barred from synapses whose both endpoints
served the last replay batch; replay unmasked), mask="refr_soft" (suppression 0.5 instead of 1),
cfg opt/eta overrides (SGD path of `_adam` honours masks).

Results (w512s5, K=1000, bw16, br16 cad1 unless noted, 3 seeds):
1. E1 mirror (prior-art direction): seq 81.3 +-2.9 (vs ours 90.8 +-0.1, unmasked ER 86.9 +-1.2),
   static 92.1 +-0.1.  Constraining wake learning hurts the new task while unmasked replay still
   perturbs the live computation -- double loss.  In the paper as "the direction of protection
   matters" (9.5-pt effect from identical machinery).
2. E2 stateless SGD (eta 0.02) inside isolation: seq 92.0 +-0.7 = NEW BEST (masked Adam 90.8;
   Adam cad2_br64 91.3; night 90.7); static 93.4 +-0.2 (> masked-Adam refr 92.0); unmasked SGD
   84.1 +-4.1.  Mask-confined moments are necessary FOR ADAM; no state at all is simpler and
   better.  sgd+cad2_br64 91.3 +-0.3: the SGD gain requires every-batch cadence.
3. E3 soft rotation: seq 88.4 +-0.7, static 86.0 +-1.3 -- worse than hard on both axes;
   all-or-none OFF periods beat gain reduction (matches Driessen 2026 tonic-vs-alternation).

**Why:** the review's strongest suggestions were turned into experiments the same day; two became
paper-strengthening results (direction ablation, stateless optimiser) and one a clean negative
(soft rotation) with a direct biological anchor.
**How to apply:** v12 default = refractory br16 cad1 + SGD (eta 0.02) + prog gate; the paper's
headline is now 92.0 +-0.7 with the stateless variant.  Open: SGD x novelty/prog gate combo, SGD
on CIFAR/feature regimes, eta sensitivity.  Related: [[cortex-phase11-results]],
[[cortex-phase10-results]], [[cortex-lit-review-2026-08]].
