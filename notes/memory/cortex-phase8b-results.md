---
name: cortex-phase8b-results
description: "Phase 8B (local sleep, round 2) on split-MNIST K=1000: refractory local sleep (units that fired sit out the next competition, then consolidate) doubles the isolated channel to ~53% and matches the best night (90.1-90.4 vs 90.7) with no night at all; wider/sparser codes raise the isolation gain (+0.9 -> +2.7) and let local sleep beat the night at equal width, but the night itself degrades on wide sparse nets; local+night combo is the best overall (91.4 at 512@5%); unmasked continuous replay raises code overlap, isolation preserves it"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-30T14:15:12.381Z
---

Ran 2026-08-31 (`run_local()` with `mask="refractory"`, `night=True`, `hidden`/`active_frac`
config keys, `task_overlap()` = mean pairwise Jaccard of task-active unit sets; CortexNet.act
honours `net.suppress`). 39 runs, 3 seeds, 5 epochs/task, K=1000 random buffer, NREM gain x3;
local = waking batch 16, one 256-sample replay batch per waking batch (18,760 replay batches vs
the night's 480). Results in `results/bio/seq/summary.csv` and the g8_* pickles.

Final accuracy (mean +-sd) / isolated channel (replay-active & asleep, L1/L2) / task overlap:
- 256-128 @10%: night 88.4 +-4.8 (7A run(): 90.7 +-0.4); local none 86.9; silent 87.8 (ch .19/.31,
  ov .75/.62); refractory 90.1 +-0.6 (ch .52/.55); silent+night 90.6 +-0.5
- 512-256 @5%:  night 84.1; none 86.1; silent 88.8 (ch .21/.32); refractory 90.4 +-0.2 (ch .54/.56);
  silent+night 91.4 +-0.3 (best in the table)
- 1024-512 @2%: night 85.5 +-3.8; none 87.4; silent 88.0 (ch .22/.39, ov .73/.58)
- Overlap: unmasked continuous replay pushes layer-1 overlap to 0.84-0.94; isolation keeps ~0.75;
  night NREM at 1024@2% gives the lowest layer-1 overlap (0.57).

Readings:
1. Direction 1 (stagger) in its refractory form WORKS: forcing what just fired to sit out the next
   competition rotates the active set, the isolated channel doubles (20-30% -> ~53%), and local
   sleep with no night reaches 90.1-90.4 -- level with the best night (90.7). The plain-lag form
   was never viable (a subset of "silent now"); it is the forced rotation that matters, i.e. a
   unit-level use-dependent sleep pressure with a real cost to inference that the cortex absorbs.
2. Direction 2 (orthogonality): wider/sparser codes raise the isolation gain (+0.9 -> +2.7) and
   make local sleep beat the night at the same width, but the night itself gets worse on wide
   sparse nets (84-86 vs 90.7), so the best night is still 256@10%. Overlap moved less than hoped
   (0.75 -> 0.73 at 2%): k-WTA sparsity alone is not strong pattern separation.
3. Combo (daytime isolated replay + night) is the best overall at 512@5% (91.4), +0.7 over the
   best night; at 256@10% it only matches the night. Daytime consolidation is a free add-on, and a
   small net gain when the codes are wider.
4. Compute: local sleep uses 39x the replay batches of a night for equal accuracy; "inference is
   training" is now a matter of per-batch cost, not of feasibility.

**Why:** the user's two proposed directions were both tested against the fallback conclusion
("local replay only as an extra"); the refractory result overturns it -- local sleep can replace
the night at equal accuracy -- while orthogonality is the weaker lever of the two.
**How to apply:** v8 local-sleep default = refractory isolation, waking batch 16, replay every
batch, readout free; keep the night as an option (combo) when compute allows. Next: reduce the
39x replay cost (replay only when the isolated channel is large / sleep-pressure gated), test the
refractory rule on the static axis, and try true pattern separation (DG-like expansion) if
orthogonality is to be pushed further. Related: [[cortex-phase8a-results]], [[cortex-phase7c-results]].
