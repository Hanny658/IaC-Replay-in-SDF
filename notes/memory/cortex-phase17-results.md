---
name: cortex-phase17-results
description: "Phase 17 (2026-09-02/03, ~160 runs): full re-baseline of every main-text claim onto the 10% record substrate (SGD); appendices keep the 5% historical grids. All orderings preserved, story strengthens: headline 92.7+-0.3 (6 seeds) vs best night 90.7 (gap 1.3->2.0); rotation +3.9, isolation +5.7 (4x variance), mirror -7.7, leak>confined. Key shifts: gates AND anchor neutralised at the record fraction (bout 92.3/94.9 vs always-on 92.7/94.7 mutually non-dominating; anchor = always-on) -> default system simplifies to always-on rotation + SGD; stateless margin over Adam narrows to +0.9; CIFAR static reversal is a 5%-only phenomenon (s10 neutral 27.6 vs 29.5+-2.8); surprise trigger never fires at s10 (0 events); raw-CIFAR ordering widens with night overtaking BP+ER (29.5 > 26.2 > 25.1); feature boundary down to 1.9 (41.7 vs 43.6), matched-substrate local 41.7 > BP 40.8 (d30k10); same-substrate night bimodally unstable at s10 (90.9+-3.2, new open item); br flat to 8 / collapses at 4; cad2 no longer free (91.4)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-02T17:22:55.510Z
---

Re-baseline waves (all detached processes + monitors, resume-safe):
- W1 (54): mechanism grid at s10 -- prog/anchor/none/adam_refr/adam_leak/adam_none/mirror/
  soft/night/ctx_none, seq+static (g17_*_s10).
- W2 (33): 6-seed hardening of bout/silent/prog/anchor/adam_refr/night s10 rows.
- W3 (60): narrative cells -- cad2_br64 91.4+-0.3, br16+night 92.7+-0.4 (night adds nothing),
  br curve (8: 91.8, 64: 92.2, 256: 92.3, 4: 63+-46 collapse), trickle cad8 81.9+-2.9,
  clocked burst 87.3+-0.2, pressure burst 89.1+-1.1 @34%, surprise 19.0 (rep=0!),
  night_nb16 87.3+-3.2; cif_s10 grid (refr 29.5+-0.8 > night 26.2 > BP 25.1 > ER 17.0+-6.1
  ~ none 16.4; static refr 27.6+-0.4 vs none 29.5+-2.8 = reversal GONE at s10);
  cfeat_s10 (refr 41.7+-1.2 > night 35.1 >> ER 20.2+-7.5 > none 17.3).
- BP transplant at record fraction: d30k10 best 40.8+-0.5 (6 seeds) < local 41.7.

Final s10 table (6 seeds headline): always-on 92.7+-0.3/94.7+-0.3 (DEFAULT); bout
92.3+-0.2/94.9+-0.3; prog gate 89.7+-1.4/95.3+-0.2; anchor 92.7+-0.4/94.6+-0.3; silent
88.8+-0.8/95.9+-0.2; SGD-none 87.0+-1.1/96.1+-0.1; no-buffer 19.6/95.8+-0.2; Adam confined
91.8+-0.6/94.6+-0.2; leak 92.6+-0.3/94.3+-0.3; night 90.9+-3.2; mirror 85.0+-2.0/92.5+-0.3;
soft 90.2+-0.4/86.4+-0.3; ER-Adam 89.7+-1.1/95.4+-0.3.  Energy (p16_spike): 93.7% @ 120 nJ
(T_s=8), 94.2% @ 242 (T_s=16), rate 94.5.

Manuscript fully re-anchored (abstract, setup, r1, micro-batch, bursts, rotation price,
CIFAR+decomposition, energy, Table 1, reading section, dial inverted, limits, FW(i) ->
night instability); figures fig_batch/fig_timing/frontier rebuilt at s10 (5% family = grey
historical dots inside the new frontier).

**Why:** the record substrate changed under the paper; every main-text number now comes from
one coherent stack (s10 + SGD), with the 5% mechanism studies honestly preserved as the
regime where gates/anchors mattered.
**How to apply:** default config going forward = w512 s10, refractory always-on, SGD eta 0.02
(eta 0.01 on CIFAR regimes).  Open: night bimodal instability at s10; CIFAR-100 (phase 18,
prepared).  Related: [[cortex-phase16-results]], [[cortex-phase14-15-results]],
[[cortex-phase13-results]].
