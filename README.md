# MLP-Cortex

An experiment ladder toward a maximally brain-like neural network — judged on a Pareto of
accuracy × energy proxy × brain-likeness, explicitly *not* SOTA-chasing. Built during (and beside)
an NTU AI6124 assignment; the course pipeline lives in a separate repository, and the shared
helpers (`data / evaluate / metrics / preprocess / models/base / models/torch_utils`) are copied
here so this repo is self-contained.

**Headline findings (v9, 2026-08-31)** — see [`docs/mlp-cortex.md`](docs/mlp-cortex.md) (Chinese)
for the full v1–v9 record:

1. **Use-dependent local sleep replaces the sleep night.** Units that just fired sit out the next
   competition (refractory rotation); hippocampal replay is written only into synapses that are
   "asleep" for the current input (exact isolation under k-WTA + ReLU, masked Adam). Split-MNIST
   class-IL: 91.3±0.3 with *no* offline night vs 90.7±0.4 for the best nightly-replay schedule.
2. **Isolation stabilises micro-batch replay.** Masked replay is insensitive to replay batch size
   down to ~8 samples, while unmasked ER and the offline night both degrade sharply at the same
   size. "Inference-is-training" therefore costs ≈1.2× the night's replay samples, not 39×.
3. **The waking-replay trigger must be homeostatic, not novelty.** Unit-level sleep-pressure
   (Process S) triggered bursts of 8–16 replay batches reach 89.9±0.1 at 15% of the replay events;
   surprise-triggered replay fails at any volume (≈19%) because it fires only after task switches.
4. The cortical constraint stack is nearly free and ~18× cheaper at inference when spiking:
   no weight transport (Kolen–Pollack + sign-concordant feedback), bounded burst errors, Dale's
   law, k-WTA on wide layers, single-phase T=0 learning, 30% distance-dependent connectivity,
   rate→spike conversion at T_s=8.

5. **A relative novelty gate removes the i.i.d. price of rotation.** Gating the refractory
   rotation by a fast/slow EMA ratio of the top-down error (ACh-like adaptation) keeps the
   sequential result (90.3-90.9) while restoring static accuracy to the no-rotation level
   (95.0 = 95.0); absolute thresholds provably do not transfer across data sets. On split
   CIFAR-10 the v9 ordering transfers and grows (refractory 27.1 > BP+ER 25.1 > night 23.8),
   and the static price REVERSES into a +2.5..+6.5 regularisation gain when underfitting.

Falsified along the way (kept for the record): input-level DG pattern separation, generative (REM)
replay as a substitute for episodic memory, reverse learning, surprise-gated memory writing,
multiplicative burst coding, naive dream negative phases, deep iterative relaxation.

## Layout

```
src/models/cortex.py    CortexNet: every mechanism behind a switch; CorticalPC (tabular wrapper)
src/models/spiking.py   rate->spike conversion + energy accounting (0.9 pJ/AC vs 4.6 pJ/MAC)
src/run_seq.py          sequential axis: split-MNIST, hippocampus, NREM/REM, internal switching,
                        local sleep (8A/8B), DG front-end + gated/burst replay (phase 9)
src/run_mnist.py        static axis: MNIST ladder (v1-v6)
src/run_bio.py          tabular ladder under the course CV protocol
scripts/p9_table.py     phase-9 aggregate table (accuracy / replay / channel / overlap / dims)
results/bio/            all summaries + per-run checkpoints (the evidence for every number)
docs/mlp-cortex.md      full experimental record v1-v9 (Chinese)
```

## Reproduce

```bash
python src/run_seq.py --configs all --seeds 0 1 2 --resume   # sequential axis (overnight-scale)
python src/run_seq.py --summary                              # -> results/bio/seq/summary.csv
python src/run_mnist.py --configs all --seeds 0 1 2          # static axis
python scripts/p9_table.py                                   # phase-9 diagnostics table
```

Key configs: `g9a_w512s5_cad2_br64` (best, no night), `g9a_w512s5_br8` (cheapest),
`g9a_w512s5_pburst16_t64` (pressure-triggered bursts), `ctx_nrem_rand_1000` (night reference).

Data notes: MNIST idx `.gz` files go in `tmp/dataset_cache/mnist/`; SUPPORT2 auto-downloads from
the UCI URL in `src/data.py`; the NUH ovarian dataset is private and is not included (only
`run_bio.py` needs it). Python 3.13 + torch (CPU), pandas, scikit-learn.
