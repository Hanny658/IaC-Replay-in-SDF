---
name: cortex-v1-results
description: "Side project (not the assignment report): MLP-Cortex v1-v3 = MLP-PC + cortical constraints on NUH/WDBC/SUPPORT2/MNIST. Hard trio (sign-concordant no-transport, bounded error, Dale) ~free; on a wide core with adam_eps=1e-3, k-WTA down to 2% active stays >94%; equaliser homeostasis is destructive, guard-rail homeostasis is harmless; KP decay must be tuned per data set"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-28T05:03:15.947Z
---

Exploration beside the assignment (user's goal: a maximally brain-like network judged on a Pareto of
accuracy vs energy proxy vs brain-likeness, explicitly not SOTA; report/ untouched). Code:
`src/models/cortex.py` (switches transport / bounded / kwta / dale / dale_fb / homeo
{"threshold" v1, "scaling" v2, "sleep" v3 equaliser, "sleep_guard" v3 guard-rail} / opt / adam_eps),
`src/run_bio.py` -> `results/bio/summary.csv`, `src/run_mnist.py` -> `results/bio/mnist/summary.csv`
(784-{64,32 | 256,128}-10, MSE-to-one-hot for BP and PC alike, 30 epochs, 3 seeds). MNIST idx files in
`tmp/dataset_cache/mnist/`. Ran 2026-08-27..29.

Findings (all with an equivalence anchor: act=tanh + switches off == MLP-PC bit-for-bit):
- Hard trio: no weight transport (Kolen-Pollack feedback synapses + `dale_fb` sign-concordance) + burst-
  bounded error + Dale. Tabular: free (SUPPORT2 0.895 = best MLP). MNIST wide core, adam_eps 1e-3:
  base3_w 97.1 vs control_w 97.8 vs BP 97.7. `dale_fb` is free everywhere and is what lets the
  no-transport feedback align (cos 1.0) under Dale's law; on 784-d input it is required (KP alone
  crushes weights or diverges).
- KP decay is a per-data-set hyperparameter, not a constant: SUPPORT2 needs 1e-2/step (1e-3 costs
  0.027 AUC), MNIST needs 1e-3 (1e-2 crushes W1). This is a real tension in the design.
- Adam runaway on wide k-WTA: after thousands of small steps v is tiny; a discontinuous k-WTA winner
  flip then gives g/sqrt(v) steps 10x too large -> |W| doubles in one epoch, then the layer dies
  (epoch ~15, all seeds). Fix: adam_eps=1e-3 (97.2 vs 82.6 same run); SGD+momentum also stable (96.1).
  adam_eps 1e-3 also lifts control_w from 97.2 to 97.8 and helps SUPPORT2 (+0.016).
- Sparsity on the wide core with eps 1e-3, no homeostasis: 25% 97.3, 10% 97.0, 5% 95.9, 2% 94.2
  (5/256 + 3/128 units active). On the 64/32 core 10% collapsed to 87%: sparsity is a property of
  wide layers. Event-driven synops only ~15-21% below dense because the dense input layer dominates.
- Homeostasis: per-step threshold (v1) and per-step scaling (v2) degrade with #updates; offline
  equaliser scaling ("sleep") collapsed 2/3 MNIST seeds (positive feedback with k-WTA); guard-rail
  form ("sleep_guard", tolerance band [0.25,4]x set point fixed on the first night) is harmless at 25%
  (95.7 = no-homeo) and +0.1..0.2 at 5% (96.1), -1 at 2% (93.2). Homeostasis is a guard, not a
  learning signal.

**Why:** these are the measured prices of each cortical constraint on the same core; they decide the
base for spiking/graph/SOM layers and say which mechanisms are settled (trio, k-WTA on wide layers)
and which are not (homeostasis, decay scaling, Adam vs SGD).
**How to apply:** v4 base = NT + dale_fb + bounded + Dale, wide layers, kwta 5-10%, adam_eps 1e-3 (or
SGD momentum for cleaner biology), KP decay 1e-2 tabular / 1e-3 MNIST, homeostasis guard-only.
Next mechanisms in the roadmap: two-compartment neurons + burst multiplexing (BurstCCN), rate->spike,
SOM topology as the GNN graph, sleep phase as FF negative data. Related: [[assignment2-design-decisions]],
[[hibernate-breaks-subprocesses]].
