# Replay in the Silent Degrees of Freedom

**Continual learning without an offline phase.** Code, results and manuscripts for the paper
*Replay in the Silent Degrees of Freedom: Continual Learning Without an Offline Phase*
(under double-blind review).

Replay-based continual learning usually consolidates in one of two ways: in an offline phase
during which the learner stops acting, or by interleaving replayed samples with the live
stream, where they perturb the computation serving the current input. Brains have a third
option, *local sleep*: brief, use-dependent off-periods of individual cortical circuits during
wakefulness. This repository asks whether a learner can consolidate the same way, writing old
memories into the degrees of freedom that the current input leaves unused, so that no offline
phase is needed and the live computation is not disturbed.

## The mechanism

- **Isolated replay.** In a network with *k*-winner-take-all hidden layers, a replay update may
  touch a synapse only if its presynaptic unit is silent for the current batch or its
  postsynaptic unit is inactive. The mask covers the whole parameter change, optimiser state
  and weight decay included. The hidden computation on the current batch is then provably
  unchanged for silent and suppressed units, and in practice for all but 0.3% of waking samples.
- **Refractory rotation.** Units that fired on the current batch sit out the next competition.
  This doubles the width of the consolidable channel and carries most of the accuracy.
- **Internal triggers** (for sparse replay budgets). A unit-level homeostatic pressure decides
  when replay bursts fire, and a relative-novelty gate decides when rotation runs.

The default learner is a two-hidden-layer network trained end to end by local rules: learned
feedback weights instead of weight transport, a bounded burst-like error signal, Dale's law,
sparse distance-dependent wiring, and a single forward and feedback sweep per batch.

## Results at a glance

Class-incremental split-MNIST, buffer of 1,000 samples, evaluated on a held-out tenth of the
training data that took part in no selection; mean ± s.d. over six seeds.

| Method | Five epochs per task | i.i.d. stream | Single pass |
|---|---|---|---|
| **Isolated replay + rotation, no offline phase** | **91.6 ± 0.3** | 94.1 ± 0.2 | **91.8 ± 0.3** |
| Unmasked interleaved replay, same learner | 85.4 ± 3.5 | 93.8 ± 5.0 | 83.0 ± 14.9 |
| Offline rehearsal ("night"), same learner | 89.0 ± 3.5 | | 76.9 ± 3.4 |
| Backprop + experience replay | 88.8 ± 0.3 | 97.4 ± 0.1 | 88.6 ± 0.2 |
| Backprop + DER++ | 91.5 ± 0.5 | | 90.1 ± 0.7 |
| Backprop + ER-ACE | 90.2 ± 0.4 | | 88.5 ± 0.3 |
| Backprop + A-GEM | 68.5 ± 7.0 | | 46.8 ± 5.0 |
| Backprop + *k*-WTA, isolated replay + rotation, same schedule | 92.1 ± 0.3 | 96.6 ± 0.1 | 92.8 ± 0.2 |

- Rotation carries most of the gain. Isolation adds the invariance guarantee at no cost on top
  of it, and at two-sample replay batches it avoids the severe failures that unmasked replay
  shows in half the seeds.
- The advantage is largest at small buffers (83.0 against 77.1 for DER++ at 200 samples) and
  gives way to the backprop references at large ones (93.8 against 95.7 at 5,000).
- On split CIFAR-10 the system leads offline rehearsal and experience replay (28.4 against 25.1
  and 24.6) but trails ER-ACE and DER++ (31.6 and 30.7).
- The mechanism is not tied to the local rule: the last row runs the same schedule on a
  backprop network with *k*-WTA layers, where rotation helps and isolation is again free.

The papers give the full picture, including the controls, the second held-out split, the
replay-cost accounting, spiking-inference energy and a decay controller for deeper networks.

## Repository layout

```
src/run_seq.py              every continual-learning experiment: the local learner under all
                            replay schedules, the offline night, the backprop baselines, the
                            backprop k-WTA transfer test; each configuration is one CONFIGS entry
src/models/cortex.py        the local learner (CortexNet), every mechanism behind a switch
src/models/spiking.py       rate-to-spike conversion and the energy proxy
src/run_mnist.py            MNIST loader and the static ladder that fixed the substrate
scripts/queue_runs.py       run a list of (config, seed, flags) jobs on N single-thread workers
scripts/agg_val.py          mean ± s.d. and seed-paired bootstrap margins from the checkpoints
scripts/table_numbers.py    the paper's table cells, printed from the checkpoints
scripts/val_spike.py        spiking-inference energy under the held-out protocol
scripts/build_arxiv_bundle.py  arXiv source bundle from the preprint
report/preprint.tex         full-length preprint (25 pages)
report/workshop/            8-page workshop version
report/make_fig*.py         every figure, regenerated from the checkpoints
results/bio/seq/            runs.csv, summary.csv and one checkpoint per run (parts/)
docs/mlp-cortex.md          the complete lab notebook, phase by phase (in Chinese)
notes/memory/               per-phase result notes
```

## Reproducing the paper

**Environment.** Python 3.13, CPU only; `pip install -r requirements.txt` pins the versions the
results were produced with. Runs are deterministic at a fixed CPU thread count. A different
thread count changes the reduction order, which *k*-WTA's discontinuity turns into a different
trajectory, so treat it like a different seed. Each checkpoint records its thread count
(`omp_threads`); nearly all held-out runs used one thread, and `scripts/queue_runs.py` always
does.

**Data.** Nothing is downloaded automatically. Place the files under `tmp/dataset_cache/`
(gitignored); derived caches are built there on first use.

```
tmp/dataset_cache/mnist/     train-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz
                             t10k-images-idx3-ubyte.gz   t10k-labels-idx1-ubyte.gz
tmp/dataset_cache/cifar10/   cifar-10-python.tar.gz
tmp/dataset_cache/cifar100/  cifar-100-python.tar.gz     (decay-controller appendix only)
```

**Protocol.** Configurations were selected on the official test split, which served as the
development set, and then frozen. `--val` re-trains a configuration with a stratified tenth
of the training data removed from the stream and evaluates on that tenth; every number in the
papers comes from such runs. `--val --val-seed 4321` draws the second held-out split.
Checkpoints are written to `results/bio/seq/parts/` as `val_<config>_s<seed>.pkl`
(`val4321_` for the second split, no prefix for development runs).

**Re-running a cell.** For example, the headline row on both axes:

```bash
python src/run_seq.py --configs g16_sgd_refr_s10_w512 g16_sgd_refr_s10_w512_static \
                      --seeds 0 1 2 3 4 5 --val
```

On one thread the headline configuration takes about 6 to 9 minutes per seed, a CIFAR-10 run
about 11, and the slowest controls (mirror isolation, soft rotation) up to half an hour.
`--resume` skips cells whose checkpoint already exists, and the repository ships every
checkpoint, so drop it (or delete the checkpoint) to retrain. For many jobs, write one
`<config> <seed> [flags]` per line and run `python scripts/queue_runs.py jobs.txt --workers 5`;
each local-learner worker needs about 1 GB of memory, a CIFAR-10 worker about 1.3 GB.

| Paper row | Configuration |
|---|---|
| Isolated replay + rotation (the system) | `g16_sgd_refr_s10_w512` |
| Without rotation / without isolation | `g16_sgd_silent_s10_w512` / `g26_ctrl_rot_noiso` |
| Unmasked interleaved replay | `g17_sgd_none_s10` |
| Offline night, same / narrow learner | `g17_night_s10` / `ctx_nrem_rand_1000` |
| Backprop + ER / DER++ / ER-ACE / A-GEM | `bp_er_1000` / `bp_derpp_a0.03_b1.0_w512_ce` / `bp_erace_1000` / `bp_agem_1000_w512` |
| Backprop + *k*-WTA transfer test | `bpk_{er,iso,er_rot,iso_rot}_lr*` |
| Buffer of 200 / 5,000 | `g27_refr_K200` / `g27_refr_K5000` |
| Split CIFAR-10 | `cif_s10_sgd_refr` |
| Decay controller / five hidden layers with skips | `g19b_mnist` / `g20b_d5_skip` |

The static (i.i.d.) and single-pass variant of every cell in the main table, the buffer axis
and the CIFAR-10 table is listed in `scripts/table_numbers.py`.

**Tables and figures without retraining.** Everything below reads the shipped checkpoints.

```bash
python scripts/table_numbers.py                      # table cells, held-out split
python scripts/table_numbers.py val4321_             # second held-out split
python scripts/agg_val.py --ref g16_sgd_refr_s10_w512 bp_derpp_a0.03_b1.0_w512_ce
python src/run_seq.py --summary                      # rebuild results/bio/seq/*.csv
MLPC_VAL=1 python report/make_figs.py                # fig_batch, fig_timing, fig_frontier, ...
MLPC_VAL=1 python report/make_fig_ksweep.py          # likewise make_fig_isomargin.py,
                                                     # make_fig_depth.py, make_fig_appendix.py
```

**Manuscripts.** `latexmk -pdf preprint.tex` in `report/` and `latexmk -pdf cl4fmagents.tex` in
`report/workshop/`.

## Notes

`src/run_bio.py`, `src/data.py`, `src/evaluate.py`, `src/metrics.py` and `src/preprocess.py`
belong to tabular experiments from the start of the project; the paper does not use them.
`scripts/p9_table.py` and `scripts/p1*_spike.py` produced earlier-phase diagnostics that the lab
notebook refers to.
