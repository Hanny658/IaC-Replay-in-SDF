---
name: cortex-phase12-results
description: "Phase 12 + 12b ablation table (2026-08-31/09-01, 3 seeds): (1) mirror isolation collapses to 81.3+-2.9 vs ours 90.8 (9.5-pt protected-direction effect); (2) stateless SGD in isolation 92.0+-0.7 NEW BEST / static 93.4; (3) soft rotation loses both axes; 12b: Adam moment LEAK (weights masked, state global) BEATS confined moments 91.9 vs 90.8 -- state confinement serves the exactness proof, hurts accuracy; full system SGD+prog-gate = 89.1/95.0 vs always-on 92.0/93.4 (gate refunds static fully under SGD but costs 2.9 seq -- 'v12 default = SGD+gate' falsified, two operating points); SGD+silent 89.0/96.0 (highest static cell); preprint gained Ablation Study section (2 tables)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-01T11:59:38.273Z
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

12b (2026-09-01, ablation tables for the preprint; 22 new runs, w512s5 K=1000 br16 cad1):
1. Adam moment leak (`leak_moments` in cortex.py `_adam`: weight change masked, m/v advance
   globally): seq 91.9 +-0.4 / static 92.8 +-0.4 -- BEATS confined 90.8/92.0.  Ordering: no state
   92.0 ~ leaked 91.9 > confined 90.8.  State confinement is required by Prop 1 (exactness), but
   for ACCURACY it is Adam's handicap (staleness); statelessness alone keeps both.  Three
   manuscript passages reworded accordingly.
2. Full system SGD+refr+prog05: seq 89.1 +-0.3 / static 95.0 +-0.6 -- WEAKLY DOMINATED by
   SGD+silent (89.0/96.0): seq tie within noise, static a full point worse.  Under SGD the gate
   manufactures no third operating point; the two real ones are always-on (92.0/93.4) and silent
   (89.0/96.0).  "v12 default = refr+SGD+prog gate" FALSIFIED.  12c: the missing Adam-silent
   static cell (static_loc16_silent_br16_w512s5) = 95.0 +-0.3, so under Adam the gate IS an
   undominated compromise (gated 90.4/94.3 vs silent 87.8/95.0: +2.6 seq for -0.7 static) --
   the gate's adaptive value is optimiser-specific.  Manuscript revised accordingly (bold =
   column best; abstract's gate claim scoped to masked Adam; future work: block-wise
   minimum-on-duration gates).
3. SGD+silent (no rotation) 89.0 +-0.5 / static 96.0 +-0.1 = highest static cell (rotation is the
   only statically costly component; SGD lifts silent from Adam's 87.8).  static SGD unmasked
   95.7; no-buffer w512s5: seq 19.5, static 95.1.
**Why:** the ablation table (user-requested) caught an untested assumption (moment confinement)
and falsified the assumed default config -- both now honestly in the paper (Section Ablation
Study, Tables 1-2; substrate ladder from run_mnist summary.csv).
**How to apply:** headline stays 92.0 +-0.7 = SGD always-on; quote the gate as a trade under SGD,
not a free unification (that claim is Adam-only).  Open: SGD on CIFAR/feature regimes, eta
sensitivity, leak at micro-batches.  Related: [[cortex-phase11-results]],
[[cortex-phase10-results]], [[cortex-lit-review-2026-08]].
