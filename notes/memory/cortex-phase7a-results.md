---
name: cortex-phase7a-results
description: "Phase 7A (hippocampal buffer + NREM replay) on split-MNIST: night replay at 3x plasticity beats interleaved ER at small buffers (K=200: cortex 86.9 vs BP-ER 70.5), replay VOLUME not buffer size is the bottleneck at large K (K=5000: 87.2 -> 93.6 with 2x replay), surprise-gated writing is worse than random even when class-balanced, BP cannot tolerate the same night replay (unstable); static axis unharmed"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-29T04:01:24.764Z
---

Ran 2026-08-29 with `src/run_seq.py` (v7 base = v6 + sweep T=0, 256-128 wide core; split-MNIST 5
tasks x 2 classes, shared 10-way readout, 5 epochs/task, 3 seeds; static axis = one task with all
10 classes, 5 epochs). Results: `results/bio/seq/summary.csv`, per-run pickles with the accuracy
matrix and buffer class counts.

Final accuracy on all 10 classes (forgetting in brackets):
- BP: none 19.7 [1.00]; ER K=200 70.5 [.37], K=1000 89.3 [.13], K=5000 95.6 [.05]
- BP + night replay (x3 lr, 20 batches/night): K=200 82.2 [.14], K=1000 80.6 +-6.7 (unstable)
- cortex: none 19.5; interleaved ER K=1000 86.8 [.14]
- cortex + random buffer + NREM (x3, 20 batches): K=200 86.9 [.11], K=1000 90.7 [.06], K=5000 87.2
  +-6.6 (one seed collapsed); K=5000 with 40 batches/night 93.6 [.03] stable; gain x2 is worse
  (86.1-86.4) -> the collapses are not from too much plasticity, x2 under-consolidates.
- cortex + surprise-gated buffer (top-30% top-down error, reservoir among them): K=200 80.2,
  K=1000 83.3, K=5000 84.2 -- worse than random at every K; the buffer skews toward hard classes
  (e.g. [166,45,36,155,113,51,39,176,195,24]). Class-balanced surprise gating: K=200 80.5,
  K=1000 89.0 +-5.0 (two seeds 92, one 83). Replaying the hardest samples consolidates worse
  than replaying typical ones.
- static axis (5 epochs, i.i.d.): BP 97.5; cortex 94.5; cortex + surprise buffer + NREM 94.8 ->
  the hippocampus does not hurt i.i.d. learning.

Readings: (1) local single-phase cortex forgets exactly like BP without a buffer; (2) night
replay at high plasticity is the winning ingredient for small buffers, and the cortex tolerates it
where BP+Adam becomes unstable; (3) at large K the fixed nightly replay volume caps consolidation
-> scale replay with buffer occupancy (a natural sleep-pressure signal for 7C); (4) the BTSP
"surprise" write gate as implemented (most surprising samples) is the wrong selector for
consolidation; a plateau signal should gate WHETHER to write, not select the hardest.

**Why:** these numbers define what 7B (generative replay replacing real samples) and 7C
(internal switching keyed to buffer occupancy) have to beat, and rule out naive surprise gating.
**How to apply:** default hippocampus = random reservoir, NREM x3 with replay batches ~ K/128
per night; keep BP + ER at equal K in every table. Related: [[cortex-phase7-roadmap]],
[[cortex-phase6-results]].
