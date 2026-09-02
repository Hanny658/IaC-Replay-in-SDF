---
name: cortex-phase18-results
description: "Phase 18 (2026-09-03, CIFAR-100): the local burst learner FAILS TO BOOTSTRAP at 100-way output (static 2.4% vs 20%+ for a 10-class subset of the same data). Mechanism chain fully localised by probes: per-class signal thins 10x with C -> W2's local gradient loses to the MNIST-tuned KP decay 1e-3 (|W2| follows the pure-decay trajectory; W1 safe since l=1 has no KP) -> a2 activity collapses -> readout gradient = 0 -> total starvation. Partial rescue: kp=1e-4 + target x3 + eta 0.02 (a2 grows again; raw ~4%, features 5.2% and climbing). BUT the whole substrate is floor regime on split CIFAR-100 (BP+ER only 6.1%, chance 1%): a real benchmark needs a stronger front-end / wider net. Infra: cifar100 loaders (gray + V1 feature), 10x10 task table, full class-count generalisation (n_classes/NC through run/run_local/BP/mirror/readout masks), kp_decay + tgt_scale as per-config knobs; grid RESULT: ordering fully REVERSES at the floor (BP+ER 6.2-9.9 > night 2.6-4.8 > no-buffer 2.3-3.1 > ALL local schedules ~1.0-1.4 incl. static) -> the local-sleep machinery presupposes a bootstrappable base task; its overhead (replay interleave x3 gain + rotation during fragile bootstrap) dominates when nothing bootstraps. Floor numbers are ordering/boundary records only"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-02T20:56:09.702Z
---

Dataset choice: CIFAR-100 over Mini-ImageNet (32x32 keeps the distance-sheet geometry and the
V1 patch front; canonical download; standard split benchmark).  Download from Toronto is slow
(~30-60 min); cache at tmp/dataset_cache/cifar100.

Diagnosis probes (all in tmp job dir, reproducible):
1. Smoke: c100 raw/feat sequential 3.2/3.8% -- near chance (1%).
2. BP+ER control 6.1% -> benchmark is floor-regime for everyone at this substrate; NOT only
   our bug.  BUT static i.i.d. 2.4% -> our net genuinely broken at 100-way.
3. 10-class subset of the SAME data: 20-23% -> output dimension is the factor.
4. eta sweep (0.002/.01/.05): all dead.  kp=0 or 1e-4 alone: dead (kp=0 raises error --
   feedback misalignment).  Readout-kwta ablation: irrelevant (predict uses x_L; _err_up does
   not gate at L).  Readout-lr boost x10: no effect (gradient already zero).
5. KEY signatures: |W_L| and |W2| decay EXACTLY as (1-1e-3)^t (zero effective gradient);
   |W1| stable (no KP at l=1); a2 0.018 -> 0.0035.  Frozen-hidden pure delta-rule readout:
   only 3.1% @ 3k batches (features too weak for 100-way even when alive).
6. Combo fix kp=1e-4 + tgt_scale=3 + eta=.02: a2 grows (0.05->0.10), raw 4.0%, features
   5.2% @ 4k batches still climbing.

**Why:** first hard scaling boundary of the project, mechanism fully identified -- the same
KP-as-per-dataset-regulariser phenomenon (v1/v6 records) in its extreme form; class-frequency
of the supervised signal is the hidden variable that KP decay must be tuned against.
**How to apply:** c100 configs carry kp_decay=1e-4, tgt_scale=3.0, eta=0.02 (knobs now exist
per-config).  Do NOT quote CIFAR-100 absolute numbers as competitive -- floor regime; use for
ordering only.  Future: stronger front-end (deeper patch stack / wider net) before CIFAR-100
becomes a headline benchmark; consider error-frequency-normalised KP decay as a principled
fix.  Related: [[cortex-phase17-results]], [[cortex-phase6-results]], [[cortex-v1-results]].
