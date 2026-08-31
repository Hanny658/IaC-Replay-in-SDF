---
name: cortex-phase8a-results
description: "Phase 8A (local sleep: replay during wake confined to units the current input leaves asleep) on split-MNIST: exact isolation is real (0.00 output change) and costs nothing when replay is continuous (87.8 vs 86.9 unmasked @K=1000, bw16, replay every batch), but it never reaches night NREM (90.7), collapses when replay is halved (70.0), fails at K=200, and hurts the static axis (-1.8); the isolated channel is only 20-30% of replay-active units because codes overlap; 'old-memory' units cannot be found by use traces (3% overlap -> no consolidation); readout must stay plastic (-15 pt otherwise)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-29T12:32:24.607Z
---

Ran 2026-08-30 with `run_local()` in `src/run_seq.py` (v7 base). Mechanism: no global night; each
waking batch (16 or 32 samples) is learned normally, and a 256-sample replay batch from the
hippocampus is applied through synapse masks: synapse j->i may update iff pre j or post i is
asleep for the current input (exact under k-WTA+ReLU; verified: masked replay changes the waking
batch's output by 0.00e+00). Masked Adam (`_adam(mask=...)`) advances m/v only inside the mask, so
momentum cannot leak. Sleep sets: silent (zero activity on the waking batch), used (top half of a
recent-use EMA, Krueger), old (long-term-used but not recently: use/long_use below median), none
(unmasked interleaved control); readout free or isolated. Waking step scaled by batch_wake/256.

Results (final acc on 10 classes; K=1000 unless noted; replay batches):
- refs: night NREM 90.7 @480, interleaved ER 86.8 @480
- bw16 / replay every batch (18,760 replay batches): none 86.9, silent 87.8 +-0.9, used 87.4,
  old 19.2, silent + isolated readout 66.9
- bw32 / replay every 2nd batch (4,685): none 85.1, silent 70.0 +-4.4, used 87.5
- K=200 bw32: none 76.0, silent 67.0, used 76.8 (night NREM K=200: 86.9)
- static (i.i.d., 5 epochs): no buffer 94.5, none 95.6, silent 92.8

Readings:
1. Isolation works as claimed and is free ONLY in the continuous regime; the isolated channel is
   narrow (20-30% of the units a replay batch fires are asleep) so it needs replay every batch to
   consolidate, and 39x the replay volume of a night still does not reach the night's accuracy.
2. Codes overlap: 70-80% of the units old memories need are also active for the current input at
   10% k-WTA. "Old-memory carriers" defined by use traces have 2-3% overlap with replay-active
   units -> no consolidation at all. Isolation-based local sleep needs more orthogonal codes
   (stronger pattern separation, sparser k-WTA on wider layers) before it can carry consolidation.
3. Continuous x3 replay over-replays: none @bw16 (86.9) < NREM (90.7) at 39x the volume; at K=200
   it overfits the buffer (76 vs 86.9). Consolidation in a concentrated night beats a background
   trickle at the same plasticity.
4. The readout must stay plastic for old classes (isolating it costs 15 pt).
5. The static axis is hurt by isolated replay (-1.8): in i.i.d. data almost all replay-active
   units are awake, so the masked updates fall on a biased sliver.

**Why:** settles whether "inference is training" via unit-level sleep can replace nights in this
architecture: not yet -- the mechanism is sound and non-interfering, but representation overlap
limits its capacity, and a concentrated night remains the better consolidator per replay sample.
**How to apply:** keep nights (7A/7C) as the default; revisit local sleep only after increasing
code orthogonality (e.g. 2-5% k-WTA on 512+ units, or explicit pattern separation), and then with
replay every batch and a free readout. Related: [[cortex-phase7c-results]], [[cortex-phase7a-results]].
