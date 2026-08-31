"""Phase 7A: a hippocampal buffer with NREM replay, judged on a sequential axis beside the static one.

    python src/run_seq.py --configs all --seeds 0 1 2          # fit + checkpoint
    python src/run_seq.py --summary

Sequential axis: split-MNIST, five tasks of two classes each ((0,1), (2,3), ...), presented one
after another, one shared ten-way readout (class-incremental, the hard version).  Metrics after
the last task: accuracy on the whole test set, per-task accuracy, and forgetting = mean over the
first four tasks of (accuracy right after learning the task - accuracy at the end).
Static axis: the same code path with a single task holding all ten classes, so that the buffer can
be shown not to hurt i.i.d. learning.

Hippocampus: a buffer of K real (x, y) pairs.  Write policies:
    random    reservoir sampling over everything seen (uniform over the stream)
    surprise  a sample can be written only if its top-down prediction error is in the top 30% of
              its batch (the BTSP plateau analogue: one-shot write gated by a dendritic event),
              reservoir sampling among those
Replay:
    nrem      after every waking epoch the cortex is shown R buffer batches at `nrem_gain` times
              the waking learning rate (high-plasticity consolidation), same local rule
    er        standard experience replay: every waking batch is concatenated with a buffer batch
              (the ML baseline, used for BP and as a control for the cortex)

Baselines that must be in every table: BP with no buffer, BP + experience replay at the same K,
cortex with no buffer, and cortex + interleaved ER (separates "night replay" from "cortex").
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from models.base import seed_everything  # noqa: E402
from models.cortex import CortexNet  # noqa: E402
from models.torch_utils import to_t  # noqa: E402
import run_mnist as RM  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "bio", "seq")
PARTS = os.path.join(OUT, "parts")

TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
V7 = dict(RM.V6, sweep=True)  # v6 + single-phase burst sweep
for k in ("T", "gamma", "spike", "hidden", "eta_override", "cosine"):  # run-loop keys, not net kwargs
    V7.pop(k, None)

CONFIGS = {
    # ---- sequential axis
    "bp_none": dict(model="bp"),
    "bp_er_200": dict(model="bp", buffer=200, policy="random", replay="er"),
    "bp_er_1000": dict(model="bp", buffer=1000, policy="random", replay="er"),
    "bp_er_5000": dict(model="bp", buffer=5000, policy="random", replay="er"),
    "ctx_none": dict(model="ctx"),
    "ctx_er_1000": dict(model="ctx", buffer=1000, policy="random", replay="er"),
    "ctx_nrem_rand_200": dict(model="ctx", buffer=200, policy="random", replay="nrem"),
    "ctx_nrem_rand_1000": dict(model="ctx", buffer=1000, policy="random", replay="nrem"),
    "ctx_nrem_rand_5000": dict(model="ctx", buffer=5000, policy="random", replay="nrem"),
    "ctx_nrem_surp_200": dict(model="ctx", buffer=200, policy="surprise", replay="nrem"),
    "ctx_nrem_surp_1000": dict(model="ctx", buffer=1000, policy="surprise", replay="nrem"),
    "ctx_nrem_surp_5000": dict(model="ctx", buffer=5000, policy="surprise", replay="nrem"),
    # ---- follow-ups from the first ladder
    # surprise gating skewed the buffer toward hard classes: class-balanced surprise gating
    "ctx_nrem_surpbal_200": dict(model="ctx", buffer=200, policy="surprise_bal", replay="nrem"),
    "ctx_nrem_surpbal_1000": dict(model="ctx", buffer=1000, policy="surprise_bal", replay="nrem"),
    # K=5000 did not beat K=1000 and one seed collapsed: replay volume scaled with K, gentler gain
    "ctx_nrem_rand_5000_R40": dict(model="ctx", buffer=5000, policy="random", replay="nrem", nrem_batches=40),
    "ctx_nrem_rand_5000_g2": dict(model="ctx", buffer=5000, policy="random", replay="nrem", nrem_gain=2.0),
    "ctx_nrem_rand_1000_g2": dict(model="ctx", buffer=1000, policy="random", replay="nrem", nrem_gain=2.0),
    # is it the night or the cortex?  BP with the same night replay
    "bp_nrem_rand_200": dict(model="bp", buffer=200, policy="random", replay="nrem"),
    "bp_nrem_rand_1000": dict(model="bp", buffer=1000, policy="random", replay="nrem"),
    # ---- 7B REM: generative rehearsal (dreams as positives) and reverse learning with a margin
    "b0_only_200": dict(model="ctx", buffer=200, policy="random", replay="nrem", decoder=True),   # decoder learned, unused
    "gen_only_0": dict(model="ctx", rem_gen=20),                                                  # no real buffer at all
    "gen_200": dict(model="ctx", buffer=200, policy="random", replay="nrem", rem_gen=20),
    "gen_1000": dict(model="ctx", buffer=1000, policy="random", replay="nrem", rem_gen=20),
    "neg_200": dict(model="ctx", buffer=200, policy="random", replay="nrem", rem_neg=0.1),
    "neg_1000": dict(model="ctx", buffer=1000, policy="random", replay="nrem", rem_neg=0.1),
    "gen_neg_200": dict(model="ctx", buffer=200, policy="random", replay="nrem", rem_gen=20, rem_neg=0.1),
    "gen_200_R40": dict(model="ctx", buffer=200, policy="random", replay="nrem", rem_gen=40),
    "neg_200_w": dict(model="ctx", buffer=200, policy="random", replay="nrem", rem_neg=0.02),
    "neg_1000_w": dict(model="ctx", buffer=1000, policy="random", replay="nrem", rem_neg=0.02),
    "gen_neg_200_w": dict(model="ctx", buffer=200, policy="random", replay="nrem", rem_gen=20, rem_neg=0.02),
    "static_ctx_gen": dict(model="ctx", static=True, rem_gen=20),
    "static_ctx_neg": dict(model="ctx", static=True, rem_neg=0.02),
    # ---- 7C internal-signal switching (stream without visible task boundaries)
    **{f"sch_{name}_{K}": dict(model="ctx", schedule="internal", buffer=K, policy="random", **kw)
       for K in (200, 1000)
       for name, kw in {
           "count_fixed": dict(pressure="count", duration="fixed"),                  # = fixed schedule, control
           "count_err": dict(pressure="count", duration="error"),                    # timing fixed, duration internal
           "count_R11": dict(pressure="count", duration="fixed", R=11),             # fixed timing at the replay volume the internal controller chose (~275)
           "surp10_fixed": dict(pressure="surprise", theta=10.0, duration="fixed"),  # timing internal
           "surp20_fixed": dict(pressure="surprise", theta=20.0, duration="fixed"),
           "surp10_err": dict(pressure="surprise", theta=10.0, duration="error"),    # both internal
           "surp20_err": dict(pressure="surprise", theta=20.0, duration="error"),
           "surp40_err": dict(pressure="surprise", theta=40.0, duration="error"),
           "surp10_err_rem": dict(pressure="surprise", theta=10.0, duration="error", rem="internal"),
       }.items()},
    # ---- 8A local sleep: replay during wake, confined to the units the current input leaves asleep
    **{f"loc_{name}_{K}": dict(model="ctx", schedule="local", buffer=K, policy="random", **kw)
       for K in (200, 1000)
       for name, kw in {
           "none": dict(mask="none"),                    # interleaved replay at the same cadence (control)
           "silent": dict(mask="silent"),                # asleep = silent for the current batch
           "idle": dict(mask="idle"),                    # asleep = the less-used half
           "used": dict(mask="used"),                    # asleep = the more-used half (Krueger)
           "old": dict(mask="old"),                      # asleep = long-used but not recently used (old-memory carriers)
           "old_iso": dict(mask="old", readout="isolated"),
           "silent_iso": dict(mask="silent", readout="isolated"),
       }.items()},
    "static_loc_none_1000": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random", mask="none"),
    "static_loc_silent_1000": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random", mask="silent"),
    # ---- 8A ladder proper: continuous background consolidation (small waking batch, replay every
    # batch or every other batch); the bw64/cadence-9 entries above were too sparse to consolidate
    **{f"loc16_{name}_1000": dict(model="ctx", schedule="local", buffer=1000, policy="random", batch_wake=16, cadence=1, **kw)
       for name, kw in {"none": dict(mask="none"), "silent": dict(mask="silent"), "used": dict(mask="used"),
                        "old": dict(mask="old"), "silent_iso": dict(mask="silent", readout="isolated")}.items()},
    **{f"loc32_{name}_{K}": dict(model="ctx", schedule="local", buffer=K, policy="random", batch_wake=32, cadence=2, **kw)
       for K in (200, 1000)
       for name, kw in {"none": dict(mask="none"), "silent": dict(mask="silent"), "used": dict(mask="used")}.items()},
    "static_loc32_none_1000": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random", batch_wake=32, cadence=2, mask="none"),
    "static_loc32_silent_1000": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random", batch_wake=32, cadence=2, mask="silent"),
    # ---- 8B: (1) code orthogonality grid (width x sparsity) for night NREM and for local silent replay,
    # with the task-overlap diagnostic; (2) refractory local sleep (what fired sits out the next round);
    # (3) local + night combination
    **{f"g8_{tag}_{kind}": dict(model="ctx", buffer=1000, policy="random", hidden=h, active_frac=a,
                                **({"replay": "nrem"} if kind == "night" else
                                   dict(schedule="local", batch_wake=16, cadence=1, mask=kind.split("_")[1])))
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05), ("w1024s2", (1024, 512), 0.02))
       for kind in ("night", "loc_silent", "loc_none")},
    "g8_w256s10_loc_refr": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16, cadence=1, mask="refractory"),
    "g8_w512s5_loc_refr": dict(model="ctx", buffer=1000, policy="random", hidden=(512, 256), active_frac=0.05,
                               schedule="local", batch_wake=16, cadence=1, mask="refractory"),
    "g8_w256s10_combo": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16, cadence=1, mask="silent", night=True),
    "g8_w512s5_combo": dict(model="ctx", buffer=1000, policy="random", hidden=(512, 256), active_frac=0.05,
                            schedule="local", batch_wake=16, cadence=1, mask="silent", night=True),
    # ---- 9B: the refractory rule on the static axis (does forced rotation cost i.i.d. accuracy?)
    **{f"static_loc16_{kind}_{tag}": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random",
                                           batch_wake=16, cadence=1, mask=kind, hidden=h, active_frac=a)
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05))
       for kind in ("none", "refractory")},
    "static_ctx_none_w512s5": dict(model="ctx", static=True, hidden=(512, 256), active_frac=0.05),
    # ---- 9C: a dentate-gyrus front-end (Cayco-Gajic & Silver 2019 checklist): fixed random
    # expansion 784 -> 2025 units, each sampling `syn` inputs (dist: a local patch; random: anywhere;
    # 784 = dense control), ReLU, k-WTA at `frac`, divisive normalisation.  The hippocampus and the
    # cortex both see the DG code.  Judged by code dimensionality (participation ratio), task
    # overlap, the isolated channel, and accuracy under the night and under refractory local sleep.
    **{f"g9_{dgk}_{tag}_{kind}": dict(model="ctx", buffer=1000, policy="random", hidden=h, active_frac=a, dg=dg,
                                       **({"replay": "nrem"} if kind == "night" else
                                          dict(schedule="local", batch_wake=16, cadence=1,
                                               mask="refractory" if kind == "refr" else "silent",
                                               **({"night": True} if kind == "combo" else {}))))
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05))
       for dgk, dg in (("dg8", dict(n=2025, syn=8)), ("dg32", dict(n=2025, syn=32)),
                       ("dgdense", dict(n=2025, syn=784)), ("dg8r", dict(n=2025, syn=8, mode="random")))
       for kind in ("night", "refr", "combo")},
    **{f"static_g9_{dgk}_{tag}": dict(model="ctx", static=True, hidden=h, active_frac=a, dg=dg)
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05))
       for dgk, dg in (("dg8", dict(n=2025, syn=8)), ("dg32", dict(n=2025, syn=32)),
                       ("dgdense", dict(n=2025, syn=784)), ("dg8r", dict(n=2025, syn=8, mode="random")))},
    # ---- 9A: what decides WHEN a replay batch is consolidated during wake.  Fixed cadence (every
    # c-th waking batch) is the cost curve; "pressure" replays when the units that are asleep for the
    # current input carry enough accumulated use (unit-level Process S, discharged by the sleep they
    # get); "surprise" replays when the waking batch's top-down error is high (7C's signal at batch
    # level).  All on refractory local sleep, K=1000, no night; judged by accuracy vs replay batches.
    **{f"g9a_{sub}_{name}": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16,
                                 mask="refractory", hidden=(512, 256), active_frac=0.05, **dgkw, **kw)
       for sub, dgkw in (("w512s5", {}), ("dg8_w512s5", dict(dg=dict(n=2025, syn=8))))
       for name, kw in {
           "cad2": dict(cadence=2), "cad4": dict(cadence=4), "cad8": dict(cadence=8), "cad16": dict(cadence=16),
           "press02": dict(cadence=1, gate="pressure", theta=0.2), "press04": dict(cadence=1, gate="pressure", theta=0.4),
           "press08": dict(cadence=1, gate="pressure", theta=0.8), "press16": dict(cadence=1, gate="pressure", theta=1.6),
           "surp03": dict(cadence=1, gate="surprise", theta=0.3), "surp05": dict(cadence=1, gate="surprise", theta=0.5),
       }.items()},
    # ---- 9C probes: after a sparse expansion the biology is DENSE fan-in (parallel fibres onto a
    # Purkinje cell, KC onto MBON); the cortex's 30% distance-dependent L1 connectivity sees only
    # ~2 active DG units per sample.  d = dense L1/L2 connectivity; fNN = DG active fraction NN%.
    **{f"g9d_{dgk}_{tag}_{kind}": dict(model="ctx", buffer=1000, policy="random", hidden=h, active_frac=a, dg=dg, conn_density=1.0,
                                        **({"replay": "nrem"} if kind == "night" else
                                           dict(schedule="local", batch_wake=16, cadence=1,
                                                mask="refractory" if kind == "refr" else "silent",
                                                **({"night": True} if kind == "combo" else {}))))
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05))
       for dgk, dg in (("dg8", dict(n=2025, syn=8)), ("dg8f10", dict(n=2025, syn=8, frac=0.10)),
                       ("dg32", dict(n=2025, syn=32)), ("dg32f10", dict(n=2025, syn=32, frac=0.10)),
                       ("dg8r", dict(n=2025, syn=8, mode="random")), ("dgdense", dict(n=2025, syn=784)))
       for kind in ("night", "refr", "combo")},
    "g9_dg8f10_w256s10_night": dict(model="ctx", buffer=1000, policy="random", replay="nrem", dg=dict(n=2025, syn=8, frac=0.10)),
    "g9d_nodg_w256s10_night": dict(model="ctx", buffer=1000, policy="random", replay="nrem", conn_density=1.0),
    "g9d_nodg_w512s5_night": dict(model="ctx", buffer=1000, policy="random", replay="nrem", hidden=(512, 256), active_frac=0.05, conn_density=1.0),
    **{f"static_g9d_{dgk}_{tag}": dict(model="ctx", static=True, hidden=h, active_frac=a, dg=dg, conn_density=1.0)
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05))
       for dgk, dg in (("dg8", dict(n=2025, syn=8)), ("dg8f10", dict(n=2025, syn=8, frac=0.10)),
                       ("dg32", dict(n=2025, syn=32)), ("dg8r", dict(n=2025, syn=8, mode="random")), ("dgdense", dict(n=2025, syn=784)))},
    # 9C follow-up: a denser DG code (10% active) under refractory local sleep
    **{f"g9_dg8f10_{tag}_refr": dict(model="ctx", buffer=1000, policy="random", hidden=h, active_frac=a,
                                     dg=dict(n=2025, syn=8, frac=0.10), schedule="local", batch_wake=16, cadence=1, mask="refractory")
       for tag, h, a in (("w256s10", (256, 128), 0.10), ("w512s5", (512, 256), 0.05))},
    # 9A follow-up: the cost of a replay event is samples x synapses, so (a) smaller replay batches at
    # every waking batch; (b) the same replay FRACTION delivered as bursts (n consecutive replay
    # batches every c waking batches: the sharp-wave-ripple pattern of quiet wakefulness) instead of
    # an even trickle -- does consolidation need temporal concentration?
    **{f"g9a_w512s5_{name}": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16,
                                  mask="refractory", hidden=(512, 256), active_frac=0.05, **kw)
       for name, kw in {
           "br64": dict(cadence=1, batch_replay=64), "br32": dict(cadence=1, batch_replay=32), "br16": dict(cadence=1, batch_replay=16),
           "burst8_64": dict(cadence=64, burst=8), "burst16_128": dict(cadence=128, burst=16),
           "burst4_32": dict(cadence=32, burst=4), "burst32_256": dict(cadence=256, burst=32),
       }.items()},
    # 9A follow-up 2: local-sleep EPISODES.  A burst of replay batches is triggered by an internal
    # signal (unit-level pressure of the asleep units, or the waking batch's surprise) instead of a
    # clock; and rarer, longer clocked bursts that approach naps (the night is 480 batches).
    **{f"g9a_w512s5_{name}": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16,
                                  mask="refractory", hidden=(512, 256), active_frac=0.05, **kw)
       for name, kw in {
           "pburst8_t32": dict(cadence=1, gate="pressure", theta=3.2, burst=8),
           "pburst8_t64": dict(cadence=1, gate="pressure", theta=6.4, burst=8),
           "pburst16_t64": dict(cadence=1, gate="pressure", theta=6.4, burst=16),
           "sburst8_03": dict(cadence=1, gate="surprise", theta=0.3, burst=8),
           "sburst8_05": dict(cadence=1, gate="surprise", theta=0.5, burst=8),
           "sburst16_05": dict(cadence=1, gate="surprise", theta=0.5, burst=16),
           "burst64_1024": dict(cadence=1024, burst=64), "burst128_2048": dict(cadence=2048, burst=128),
           "burst64_512": dict(cadence=512, burst=64),
       }.items()},
    # 9A follow-up 3: bursts of SMALL replay batches -- episodes x sample efficiency.  Replay cost
    # in samples: cad1 = 18,760 x 256 = 4.8M; the night = 480 x 256 = 123k.
    **{f"g9a_w512s5_{name}": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16,
                                  mask="refractory", hidden=(512, 256), active_frac=0.05, **kw)
       for name, kw in {
           "pburst8_t32_br64": dict(cadence=1, gate="pressure", theta=3.2, burst=8, batch_replay=64),
           "pburst16_t64_br64": dict(cadence=1, gate="pressure", theta=6.4, burst=16, batch_replay=64),
           "burst16_128_br64": dict(cadence=128, burst=16, batch_replay=64),
           "pburst8_t32_br32": dict(cadence=1, gate="pressure", theta=3.2, burst=8, batch_replay=32),
           "cad2_br64": dict(cadence=2, batch_replay=64),
           "cad4_br64": dict(cadence=4, batch_replay=64),
       }.items()},
    # 9A follow-up 4: the cost floor (replay batch 8 / 4; bursts of batch-16 replay), K=200
    # generality, local + night on the cheap local, and the static price at batch 16.
    **{f"g9a_w512s5_{name}": dict(model="ctx", buffer=K, policy="random", schedule="local", batch_wake=16,
                                  mask="refractory", hidden=(512, 256), active_frac=0.05, **kw)
       for name, K, kw in (
           ("br8", 1000, dict(cadence=1, batch_replay=8)), ("br4", 1000, dict(cadence=1, batch_replay=4)),
           ("pburst16_t64_br16", 1000, dict(cadence=1, gate="pressure", theta=6.4, burst=16, batch_replay=16)),
           ("burst16_128_br16", 1000, dict(cadence=128, burst=16, batch_replay=16)),
           ("br16_night", 1000, dict(cadence=1, batch_replay=16, night=True)),
           ("br16_K200", 200, dict(cadence=1, batch_replay=16)),
           ("pburst16_t64_br16_K200", 200, dict(cadence=1, gate="pressure", theta=6.4, burst=16, batch_replay=16)),
       )},
    "static_loc16_refractory_br16_w512s5": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random",
                                                batch_wake=16, cadence=1, batch_replay=16, mask="refractory", hidden=(512, 256), active_frac=0.05),
    "static_loc16_none_br16_w512s5": dict(model="ctx", schedule="local", static=True, buffer=1000, policy="random",
                                          batch_wake=16, cadence=1, batch_replay=16, mask="none", hidden=(512, 256), active_frac=0.05),
    # 9A follow-up 5: the batch-size floor (2, 1 replay samples per waking batch) and the control that
    # separates isolation from batch size (unmasked / silent replay at the same tiny batch).
    **{f"g9a_w512s5_{name}": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16,
                                  hidden=(512, 256), active_frac=0.05, cadence=1, **kw)
       for name, kw in (("br2", dict(mask="refractory", batch_replay=2)), ("br1", dict(mask="refractory", batch_replay=1)),
                        ("none_br16", dict(mask="none", batch_replay=16)), ("none_br8", dict(mask="none", batch_replay=8)),
                        ("silent_br16", dict(mask="silent", batch_replay=16)))},
    # 9A fairness control: does the NIGHT also tolerate tiny replay batches?  480 x 16 = 7.7k samples.
    "ctx_nrem_rand_1000_nb16": dict(model="ctx", buffer=1000, policy="random", replay="nrem", hidden=(512, 256), active_frac=0.05, night_batch=16),
    "ctx_nrem_rand_1000_nb64": dict(model="ctx", buffer=1000, policy="random", replay="nrem", hidden=(512, 256), active_frac=0.05, night_batch=64),
    "ctx_nrem_rand_1000_w512": dict(model="ctx", buffer=1000, policy="random", replay="nrem", hidden=(512, 256), active_frac=0.05),
    # 9A: tiny-batch local sleep on the w256s10 substrate (whose night is the strong one)
    **{f"g9a_w256s10_{name}": dict(model="ctx", buffer=1000, policy="random", schedule="local", batch_wake=16,
                                   mask="refractory", cadence=1, **kw)
       for name, kw in (("br16", dict(batch_replay=16)), ("br8", dict(batch_replay=8)))},
    # ---- phase 10A: homeostatic rotation (does the static price of forced rest disappear when
    # rest is gated by accumulated per-unit pressure instead of the last batch?)
    **{f"{pre}g10_{name}_w512s5": dict(model="ctx", schedule="local", buffer=1000, policy="random",
                                       batch_wake=16, cadence=1, batch_replay=16, hidden=(512, 256), active_frac=0.05,
                                       **({"static": True} if pre else {}), **kw)
       for pre in ("", "static_")
       for name, kw in {"rp05": dict(mask="refr_press", theta_r=0.5), "rp1": dict(mask="refr_press", theta_r=1.0),
                        "rp2": dict(mask="refr_press", theta_r=2.0), "rp4": dict(mask="refr_press", theta_r=4.0),
                        "rp8": dict(mask="refr_press", theta_r=8.0), "rf25": dict(mask="refr_frac", frac_r=0.25),
                        "rf50": dict(mask="refr_frac", frac_r=0.50)}.items()},
    # ---- phase 10A2: ACh-gated rotation -- plain refractory, but only while the stream is novel
    **{f"{pre}g10_ach{tag}_w512s5": dict(model="ctx", schedule="local", buffer=1000, policy="random",
                                         batch_wake=16, cadence=1, batch_replay=16, hidden=(512, 256), active_frac=0.05,
                                         mask="refr_ach", theta_s=th, **({"static": True} if pre else {}))
       for pre in ("", "static_")
       for tag, th in (("15", 0.15), ("25", 0.25), ("40", 0.40))},
    "cif_ach25_br16": dict(model="ctx", dataset="cifar", buffer=1000, policy="random", schedule="local",
                           batch_wake=16, cadence=1, batch_replay=16, mask="refr_ach", theta_s=0.25,
                           hidden=(512, 256), active_frac=0.05),
    # ---- phase 10A3: novelty as fast-over-slow error ratio (adaptation), data-set independent
    **{f"{pre}g10_nov{tag}_w512s5": dict(model="ctx", schedule="local", buffer=1000, policy="random",
                                         batch_wake=16, cadence=1, batch_replay=16, hidden=(512, 256), active_frac=0.05,
                                         mask="refr_nov", beta_nov=b, **({"static": True} if pre else {}))
       for pre in ("", "static_")
       for tag, b in (("11", 1.1), ("13", 1.3), ("15", 1.5))},
    "cif_nov13_br16": dict(model="ctx", dataset="cifar", buffer=1000, policy="random", schedule="local",
                           batch_wake=16, cadence=1, batch_replay=16, mask="refr_nov", beta_nov=1.3,
                           hidden=(512, 256), active_frac=0.05),
    # ---- phase 10B: the K=200 gap -- over-plasticity (gain 1) and replay correlation (noise)
    **{f"g10_k200_{name}": dict(model="ctx", schedule="local", buffer=200, policy="random", batch_wake=16,
                                mask="refractory", cadence=1, batch_replay=16, hidden=(512, 256), active_frac=0.05, **kw)
       for name, kw in {"g1": dict(nrem_gain=1.0), "g2": dict(nrem_gain=2.0), "n02": dict(replay_noise=0.2),
                        "g1_n02": dict(nrem_gain=1.0, replay_noise=0.2),
                        "pb16_g1": dict(gate="pressure", theta=6.4, burst=16, nrem_gain=1.0),
                        "pb16_n02": dict(gate="pressure", theta=6.4, burst=16, replay_noise=0.2)}.items()},
    "g10_n02_w512s5": dict(model="ctx", schedule="local", buffer=1000, policy="random", batch_wake=16,
                           mask="refractory", cadence=1, batch_replay=16, replay_noise=0.2, hidden=(512, 256), active_frac=0.05),
    # ---- phase 10C: split CIFAR-10 (grayscale) -- does the v9 conclusion transfer?
    **{f"cif_{name}": dict(model="ctx", dataset="cifar", buffer=1000, policy="random",
                           hidden=(512, 256), active_frac=0.05, **kw)
       for name, kw in {
           "night": dict(replay="nrem"),
           "refr_br16": dict(schedule="local", batch_wake=16, cadence=1, batch_replay=16, mask="refractory"),
           "none_br16": dict(schedule="local", batch_wake=16, cadence=1, batch_replay=16, mask="none"),
           "rp4_br16": dict(schedule="local", batch_wake=16, cadence=1, batch_replay=16, mask="refr_press", theta_r=4.0),
       }.items()},
    "cif_ctx_none": dict(model="ctx", dataset="cifar"),
    "cif_bp_none": dict(model="bp", dataset="cifar"),
    "cif_bp_er_1000": dict(model="bp", dataset="cifar", buffer=1000, policy="random", replay="er"),
    "cif_static_none": dict(model="ctx", dataset="cifar", static=True, schedule="local", buffer=1000, policy="random",
                            batch_wake=16, cadence=1, batch_replay=16, mask="none", hidden=(512, 256), active_frac=0.05),
    "cif_static_refr": dict(model="ctx", dataset="cifar", static=True, schedule="local", buffer=1000, policy="random",
                            batch_wake=16, cadence=1, batch_replay=16, mask="refractory", hidden=(512, 256), active_frac=0.05),
    "cif_static_rp4": dict(model="ctx", dataset="cifar", static=True, schedule="local", buffer=1000, policy="random",
                           batch_wake=16, cadence=1, batch_replay=16, mask="refr_press", theta_r=4.0, hidden=(512, 256), active_frac=0.05),
    # ---- static axis: the buffer must not hurt i.i.d. learning
    "static_bp_none": dict(model="bp", static=True),
    "static_ctx_none": dict(model="ctx", static=True),
    "static_ctx_nrem_surp_1000": dict(model="ctx", static=True, buffer=1000, policy="surprise", replay="nrem"),
}


# ------------------------------------------------------------------ data (phase 10: CIFAR)
CIFAR_DIR = os.path.join(ROOT, "tmp", "dataset_cache", "cifar10")


def load_cifar10_gray():
    """CIFAR-10 as 32x32 luminance, per-pixel standardised on the training set.  Grayscale keeps
    the input a single sheet, so distance-dependent connectivity keeps its geometry."""
    cache = os.path.join(CIFAR_DIR, "gray.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        Xtr, ytr, Xte, yte = z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    else:
        import pickle as pk
        import tarfile
        with tarfile.open(os.path.join(CIFAR_DIR, "cifar-10-python.tar.gz")) as tf:
            def batch(name):
                d = pk.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
                return d[b"data"], np.array(d[b"labels"], dtype=np.int64)
            parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
            Xtr, ytr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
            Xte, yte = batch("test_batch")

        def gray(X):
            X = X.reshape(-1, 3, 1024).astype(np.float32) / 255
            return 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
        Xtr, Xte = gray(Xtr), gray(Xte)
        os.makedirs(CIFAR_DIR, exist_ok=True)
        np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte


def load_data(cfg):
    if cfg.get("dataset") == "cifar":
        X, y, Xe, ye = load_cifar10_gray()
        return X, y, Xe, ye, (32, 32)
    X, y, Xe, ye = RM.load_mnist()
    return X, y, Xe, ye, (28, 28)


# ------------------------------------------------------------------ dentate gyrus (9C)
class DG:
    """Fixed sparse random expansion in front of the cortex, after the pattern-separation checklist
    of Cayco-Gajic & Silver (2019): divergence (784 -> n), few synaptic inputs per unit (`syn`, the
    granule cell's ~4 mossy fibres / the Kenyon cell's ~7 claws), a threshold nonlinearity, k-WTA
    (feedback inhibition) and divisive gain control.  Excitatory, equal, frozen synapses, as the
    fly's PN->KC projection.  mode "dist" samples a unit's inputs from a local patch around its
    position on the sheet (the granule cell's short dendrites), "random" from anywhere."""

    def __init__(self, n_in, n_dg, n_syn, frac, seed, input_shape=(28, 28), lam=0.1, mode="dist"):
        g = torch.Generator().manual_seed(seed + 977)
        self.n_dg, self.n_syn, self.k = n_dg, min(n_syn, n_in), max(1, int(round(frac * n_dg)))
        in_c, dg_c = CortexNet._sheet(n_in, input_shape, g), CortexNet._sheet(n_dg, None, g)
        W = torch.zeros(n_dg, n_in)
        if n_syn >= n_in:
            W = torch.rand(n_dg, n_in, generator=g)  # dense control: every input, random positive weight
        else:
            p = torch.exp(-torch.cdist(dg_c, in_c) / lam) if mode == "dist" else torch.ones(n_dg, n_in)
            for i in range(n_dg):
                W[i, torch.multinomial(p[i] / p[i].sum(), n_syn, replacement=False, generator=g)] = 1.0
        self.W = W / np.sqrt(self.n_syn)
        self.cost = 0.0

    def __call__(self, X):
        with torch.no_grad():
            z = torch.relu(X @ self.W.T)
            thr = z.topk(self.k, dim=1).values[:, -1:]
            z = torch.where(z >= thr, z, torch.zeros_like(z))
            z = z / (z.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)  # divisive normalisation
            self.cost = float((X != 0).float().sum(1).mean()) * float((self.W != 0).float().sum(0).mean())
        return z


def make_front(cfg, seed):
    dg = cfg.get("dg")
    if not dg:
        return None
    return DG(784, dg["n"], dg["syn"], dg.get("frac", 0.03), seed, mode=dg.get("mode", "dist"))


def code_dim(net, Xte, n=3000):
    """Participation-ratio dimensionality of the population code at the cortex input (the raw or DG
    code) and at every hidden layer, on n test samples: (sum lambda)^2 / sum lambda^2."""
    net.training = False
    with torch.no_grad():
        _, a = net.forward(to_t(Xte[:n]))
    out = []
    for l in range(0, net.L):
        A = a[l] - a[l].mean(0)
        lam = torch.linalg.svdvals(A.double()).pow(2) / (A.shape[0] - 1)
        out.append(float(lam.sum() ** 2 / (lam.pow(2).sum() + 1e-12)))
    return out


# ------------------------------------------------------------------ hippocampus
class Buffer:
    def __init__(self, K, policy, g, top_frac=0.3):
        self.K, self.policy, self.g, self.top_frac = K, policy, g, top_frac
        self.X, self.Y, self.seen = [], [], 0

    def offer(self, Xb, yb, surprise):
        """One-shot writes.  Reservoir sampling keeps the buffer uniform over the eligible stream.

        surprise_bal: the surprise gate is applied within each class of the batch and the buffer
        holds a per-class quota of K/10, so the write rule cannot skew the buffer toward the
        classes that happen to be hard (which plain surprise gating did)."""
        if self.K == 0:
            return
        if self.policy == "surprise_bal":
            if not hasattr(self, "seen_c"):
                self.seen_c, self.slots = [0] * 10, {c: [] for c in range(10)}
            quota = max(1, self.K // 10)
            for c in yb.unique().tolist():
                m = (yb == c).nonzero().squeeze(1)
                k = max(1, int(self.top_frac * len(m)))
                for i in m[torch.topk(surprise[m], k).indices].tolist():
                    self.seen_c[c] += 1
                    if len(self.slots[c]) < quota:
                        self.slots[c].append(len(self.X)); self.X.append(Xb[i]); self.Y.append(c)
                    else:
                        j = int(torch.randint(0, self.seen_c[c], (1,), generator=self.g))
                        if j < quota:
                            self.X[self.slots[c][j]] = Xb[i]
            return
        if self.policy == "surprise":
            k = max(1, int(self.top_frac * len(yb)))
            elig = torch.topk(surprise, k).indices
        else:
            elig = torch.arange(len(yb))
        for i in elig.tolist():
            self.seen += 1
            if len(self.X) < self.K:
                self.X.append(Xb[i]); self.Y.append(int(yb[i]))
            else:
                j = int(torch.randint(0, self.seen, (1,), generator=self.g))
                if j < self.K:
                    self.X[j], self.Y[j] = Xb[i], int(yb[i])

    def sample(self, n):
        if not self.X:
            return None, None
        idx = torch.randint(0, len(self.X), (min(n, len(self.X)),), generator=self.g)
        return torch.stack([self.X[i] for i in idx]), torch.as_tensor([self.Y[i] for i in idx])

    def class_counts(self):
        return np.bincount(np.array(self.Y, dtype=int), minlength=10).tolist() if self.Y else [0] * 10


# ------------------------------------------------------------------ models
def make_bp(seed, hidden, n_in=784):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden[0]), torch.nn.ReLU(),
                              torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(),
                              torch.nn.Linear(hidden[1], 10))
    return net, torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)


def bp_step(net, opt, Xb, Yb):
    opt.zero_grad()
    out = net(Xb)
    loss = ((out - Yb) ** 2).sum(1).mean()
    loss.backward()
    opt.step()
    return ((out.detach() - Yb) ** 2).sum(1)  # per-sample surprise


def ctx_step(net, Xb, Yb, eta):
    x, a, eps = net.relax(Xb, Yb, 0, 0.0)  # sweep: T and gamma unused
    net.local_update(x, a, eps, eta, 1e-3)
    return (eps[net.L] ** 2).sum(1)


def predict(model, net, X):
    with torch.no_grad():
        if model == "bp":
            return net(to_t(X)).numpy()
        net.training = False
        return net.forward(to_t(X))[0][-1].numpy()


# ------------------------------------------------------------------ protocol
def run(config, seed, epochs_per_task, batch, nrem_batches, nrem_gain):
    cfg = CONFIGS[config]
    model, K = cfg["model"], cfg.get("buffer", 0)
    policy, replay, static = cfg.get("policy", "random"), cfg.get("replay", None), cfg.get("static", False)
    nrem_batches, nrem_gain = cfg.get("nrem_batches", nrem_batches), cfg.get("nrem_gain", nrem_gain)
    # 7B: REM.  rem_gen = dream batches replayed as positives per night; rem_neg = reverse-learning
    # weight (0 = off); both need the decoder.
    rem_gen, rem_neg = cfg.get("rem_gen", 0), cfg.get("rem_neg", 0.0)
    decoder = bool(cfg.get("decoder", False) or rem_gen or rem_neg)
    Xtr, ytr, Xte, yte, ishape = load_data(cfg)
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    hidden = RM.WIDE
    hidden = tuple(cfg.get("hidden", RM.WIDE))
    if model == "bp":
        net, opt = make_bp(seed, hidden, Xtr.shape[1])
    else:
        front = make_front(cfg, seed)
        net = CortexNet([front.n_dg if front else Xtr.shape[1], *hidden, 10], seed=seed, input_shape=None if front else ishape,
                        decoder=decoder, rem_neg=rem_neg, **dict(V7, active_frac=cfg.get("active_frac", V7["active_frac"]),
                                                                 conn_density=cfg.get("conn_density", V7["conn_density"])))
        net.front = front
    buf = Buffer(K, policy, g)
    tasks = [tuple(range(10))] if static else TASKS
    onehot = lambda y: torch.nn.functional.one_hot(torch.as_tensor(y), 10).float()
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    t0 = time.time()
    for ti, classes in enumerate(tasks):
        m = np.isin(ytr, classes)
        Xt, yt = to_t(Xtr[m]), torch.as_tensor(ytr[m])
        n = len(yt)
        for ep in range(epochs_per_task):
            if model == "ctx":
                net.training = True
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                Xb, yb = Xt[idx], yt[idx]
                if replay == "er" and K:
                    Xr, yr = buf.sample(batch)
                    if Xr is not None:
                        Xb_all, yb_all = torch.cat([Xb, Xr]), torch.cat([yb, yr])
                    else:
                        Xb_all, yb_all = Xb, yb
                else:
                    Xb_all, yb_all = Xb, yb
                if model == "bp":
                    surprise = bp_step(net, opt, Xb_all, onehot(yb_all))
                else:
                    surprise = ctx_step(net, Xb_all, onehot(yb_all), 1e-3)
                buf.offer(Xb, yb, surprise[:len(yb)])  # only the fresh samples can be written
            if replay == "nrem" and K:  # night: high-plasticity replay from the hippocampus
                if model == "ctx":
                    net.training = False
                for _ in range(nrem_batches):
                    Xr, yr = buf.sample(cfg.get("night_batch", batch))
                    if Xr is None:
                        break
                    if model == "bp":
                        for pg in opt.param_groups:
                            pg["lr"] = 1e-3 * nrem_gain
                        bp_step(net, opt, Xr, onehot(yr))
                        for pg in opt.param_groups:
                            pg["lr"] = 1e-3
                    else:
                        ctx_step(net, Xr, onehot(yr), 1e-3 * nrem_gain)
            if model == "ctx" and (rem_gen or rem_neg):  # REM: dreams as rehearsal and/or reverse learning
                net.training = False
                for _ in range(max(rem_gen, 20 if rem_neg else 0)):
                    X_hat, y_hat = net.rem_generate(batch, g, 5, 0.4)
                    if rem_gen:
                        ctx_step(net, X_hat, onehot(y_hat), 1e-3)
                    if rem_neg:
                        net.rem_negative(X_hat, 1e-3)
        S = predict(model, net, Xte)
        pred = S.argmax(1)
        for tj, cj in enumerate(tasks[:ti + 1]):
            mt = np.isin(yte, cj)
            acc_matrix[ti, tj] = float((pred[mt] == yte[mt]).mean())
        print(f"  {config:26s} seed {seed} after task {ti + 1}: per-task acc "
              f"{[round(float(v), 3) for v in acc_matrix[ti, :ti + 1]]}  buffer {buf.class_counts()}", flush=True)
    S = predict(model, net, Xte)
    final_acc = float((S.argmax(1) == yte).mean())
    T = len(tasks)
    forgetting = float(np.mean([acc_matrix[j, j] - acc_matrix[T - 1, j] for j in range(T - 1)])) if T > 1 else 0.0
    out = dict(config=config, seed=seed, model=model, buffer=K, policy=policy if K else None, replay=replay if K else None,
               static=static, final_acc=final_acc, forgetting=forgetting, acc_matrix=acc_matrix.tolist(),
               buffer_classes=buf.class_counts(), fit_s=time.time() - t0)
    globals()["LAST_NET"] = net
    if model == "ctx":
        out["overlap"] = task_overlap(net, Xte, yte, tasks)
        out["overlap_in"] = task_overlap(net, Xte, yte, tasks, layers=(0,))[0]
        out["dim"] = code_dim(net, Xte)
        out["hidden"], out["active_frac"] = hidden, cfg.get("active_frac", V7["active_frac"])
        out["synops"] = net.last_cost["synops"]
        print(f"  {config:26s} seed {seed}: task-overlap in {out['overlap_in']:.2f} hidden {[round(v, 2) for v in out['overlap']]}  "
              f"dim {[round(v, 1) for v in out['dim']]}  synops/sample {out['synops']:.0f}", flush=True)
    if model == "ctx" and decoder:  # dream quality: an independent judge, the net's own readout, reconstruction
        out.update(dream_quality(net, g, Xtr, ytr, Xte, yte, seed))
        print(f"  {config:26s} seed {seed}: dreams -> judge acc {out['dream_judge_acc']:.3f}  self acc {out['dream_self_acc']:.3f}  "
              f"recon mse {out['recon_mse']:.3f}", flush=True)
    print(f"  {config:26s} seed {seed}: FINAL acc {final_acc:.4f}  forgetting {forgetting:.4f}  ({out['fit_s'] / 60:.1f} min)", flush=True)
    return out


def dream_quality(net, g, Xtr, ytr, Xte, yte, seed, n=2000):
    """How real are the dreams?  judge = a BP net trained on the whole (i.i.d.) training set, so it
    knows every class; self = the dreaming net's own readout; recon = decoder error on test input."""
    judge, opt = make_bp(seed + 100, RM.WIDE)
    Xt, Yt = to_t(Xtr), torch.nn.functional.one_hot(torch.as_tensor(ytr), 10).float()
    gj = torch.Generator().manual_seed(seed + 100)
    for _ in range(3):
        perm = torch.randperm(len(Xt), generator=gj)
        for i in range(0, len(Xt), 256):
            idx = perm[i:i + 256]
            bp_step(judge, opt, Xt[idx], Yt[idx])
    with torch.no_grad():
        X_hat, y_hat = net.rem_generate(n, g, 5, 0.4)
        judge_acc = float((judge(X_hat).argmax(1) == y_hat).float().mean())
        net.training = False
        self_acc = float((net.forward(X_hat)[0][-1].argmax(1) == y_hat).float().mean())
        Xte_t = to_t(Xte[:n])
        recon = net.decode(net.forward(Xte_t)[1][1])
        recon_mse = float(((recon - Xte_t) ** 2).mean())
        judge_test = float((judge(to_t(Xte)).argmax(1) == torch.as_tensor(yte)).float().mean())
    return dict(dream_judge_acc=judge_acc, dream_self_acc=self_acc, recon_mse=recon_mse, judge_test_acc=judge_test)


def run_stream(config, seed, epochs_per_task, batch, nrem_gain, replay_cap=500):
    """7C: the same split-MNIST stream, but wake/sleep switching is driven by internal signals.

    The agent sees a continuous stream of batches (each task's data passes epochs_per_task times,
    so the waking compute equals the fixed schedule's) and never a task boundary.
      pressure  "count":    S += 1 per batch; sleep at S >= theta (theta = batches/epoch reproduces
                            the fixed schedule and is the control)
                "surprise": S += mean top-down error of the batch; a task switch raises the error
                            and brings sleep forward, steady learning postpones it
      duration  "fixed":    R replay batches per night
                "error":    replay until the mean error of the last 5 replay batches < eps_stop
                            (or R_max), minimum 5: wake when the buffer has been consolidated
      rem       "off" | "internal": after NREM, dream batches replayed as positives until the
                            net's own readout agrees with the dream labels (>= 0.95) or 20 batches
    Total replay is capped at replay_cap batches (= the fixed schedule's 5 tasks x 5 nights x 20).
    """
    cfg = CONFIGS[config]
    K, policy = cfg.get("buffer", 0), cfg.get("policy", "random")
    pressure, theta = cfg.get("pressure", "count"), cfg.get("theta", None)
    duration, R, eps_stop, R_max = cfg.get("duration", "fixed"), cfg.get("R", 20), cfg.get("eps_stop", 0.15), cfg.get("R_max", 40)
    rem = cfg.get("rem", "off")
    Xtr, ytr, Xte, yte = RM.load_mnist()
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    net = CortexNet([784, *RM.WIDE, 10], seed=seed, input_shape=(28, 28), decoder=(rem != "off"), **V7)
    buf = Buffer(K, policy, g)
    onehot = lambda y: torch.nn.functional.one_hot(torch.as_tensor(y), 10).float()
    # build the stream: task after task, each shuffled epochs_per_task times
    stream, boundaries = [], []
    for classes in TASKS:
        m = np.isin(ytr, classes)
        Xt, yt = to_t(Xtr[m]), torch.as_tensor(ytr[m])
        boundaries.append(len(stream))
        for _ in range(epochs_per_task):
            perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(yt), batch):
                idx = perm[i:i + batch]
                stream.append((Xt[idx], yt[idx]))
    batches_per_epoch = (boundaries[1] - boundaries[0]) // epochs_per_task
    if theta is None:
        theta = batches_per_epoch  # count pressure: one epoch, the fixed schedule
    S, replay_used, sleeps, sleep_log, t0 = 0.0, 0, 0, [], time.time()
    acc_matrix = np.full((len(TASKS), len(TASKS)), np.nan)
    next_eval = 1
    for bi, (Xb, yb) in enumerate(stream):
        net.training = True
        surprise = ctx_step(net, Xb, onehot(yb), 1e-3)
        buf.offer(Xb, yb, surprise)
        S += 1.0 if pressure == "count" else float(surprise.mean())
        last = bi + 1 == len(stream)  # the fixed schedule always ends with a night: so does this
        if (S >= theta or last) and K and replay_used < replay_cap:
            # ---- night
            sleeps += 1
            net.training = False
            used, recent = 0, []
            limit = R if duration == "fixed" else R_max
            while used < limit and replay_used < replay_cap:
                Xr, yr = buf.sample(batch)
                if Xr is None:
                    break
                err = float(ctx_step(net, Xr, onehot(yr), 1e-3 * nrem_gain).mean())
                used += 1; replay_used += 1; recent.append(err)
                if duration == "error" and used >= 5 and np.mean(recent[-5:]) < eps_stop:
                    break
            rem_used = 0
            if rem == "internal" and hasattr(net, "G"):
                for _ in range(20):
                    X_hat, y_hat = net.rem_generate(batch, g, 5, 0.4)
                    with torch.no_grad():
                        agree = float((net.forward(X_hat)[0][-1].argmax(1) == y_hat).float().mean())
                    if agree >= 0.95:
                        break
                    ctx_step(net, X_hat, onehot(y_hat), 1e-3); rem_used += 1
            sleep_log.append(dict(batch=bi, nrem=used, rem=rem_used, pressure=S))
            S = 0.0
        # evaluate at the end of each task's stream segment (same points as the fixed schedule)
        if next_eval < len(boundaries) and bi + 1 == boundaries[next_eval] or bi + 1 == len(stream):
            ti = next_eval - 1 if bi + 1 != len(stream) else len(TASKS) - 1
            S_te = predict("ctx", net, Xte); pred = S_te.argmax(1)
            for tj, cj in enumerate(TASKS[:ti + 1]):
                mt = np.isin(yte, cj)
                acc_matrix[ti, tj] = float((pred[mt] == yte[mt]).mean())
            next_eval += 1
    S_te = predict("ctx", net, Xte)
    final_acc = float((S_te.argmax(1) == yte).mean())
    T = len(TASKS)
    forgetting = float(np.mean([acc_matrix[j, j] - acc_matrix[T - 1, j] for j in range(T - 1)]))
    # how many sleeps fell inside the first epoch after a task switch (novelty-triggered sleep)
    early = sum(1 for s in sleep_log for b0 in boundaries[1:] if b0 <= s["batch"] < b0 + batches_per_epoch)
    out = dict(config=config, seed=seed, model="ctx", buffer=K, policy=policy, replay="nrem", static=False,
               schedule="internal", final_acc=final_acc, forgetting=forgetting, acc_matrix=acc_matrix.tolist(),
               buffer_classes=buf.class_counts(), sleeps=sleeps, replay_used=replay_used,
               rem_used=sum(s["rem"] for s in sleep_log), early_sleeps=early, sleep_log=sleep_log, fit_s=time.time() - t0)
    print(f"  {config:26s} seed {seed}: FINAL acc {final_acc:.4f}  forgetting {forgetting:.4f}  sleeps {sleeps}  "
          f"replay {replay_used}  early-after-switch {early}  ({out['fit_s'] / 60:.1f} min)", flush=True)
    return out


def run_local(config, seed, epochs_per_task, batch_wake, batch_replay, nrem_gain, cadence):
    """8A: local sleep -- no global night.  While the waking batch is being processed, the units
    it leaves silent consolidate a replay batch from the hippocampus, at NREM plasticity.

    Isolation is exact under k-WTA + ReLU: a synapse j->i may take the replay update iff the
    presynaptic unit j or the postsynaptic unit i is asleep for the current input, because then
    the change is invisible to the current inference (a_j = 0 now, or unit i emits nothing now).
    Sleep-mask policies for the hidden layers:
        silent  units with zero activity across the current waking batch (batch_wake small so the
                silent set is not empty)
        idle    units whose recent use (EMA of firing) is below the layer median: idle resources
                consolidate
        used    units whose recent use is above the median: use-dependent local sleep (Krueger)
        none    no mask: interleaved replay at the same cadence (the control)
    readout "free": the readout rows are always plastic (old classes must stay calibrated);
    readout "isolated": readout synapses obey the same rule (only from sleeping hidden units).
    Replay volume: one replay batch every `cadence` waking batches, sized to match the NREM budget.
    """
    cfg = CONFIGS[config]
    K, policy, static = cfg.get("buffer", 0), cfg.get("policy", "random"), cfg.get("static", False)
    mask_policy, readout = cfg.get("mask", "silent"), cfg.get("readout", "free")
    hidden, af, night = tuple(cfg.get("hidden", RM.WIDE)), cfg.get("active_frac", V7["active_frac"]), cfg.get("night", False)
    gate, theta, delta = cfg.get("gate"), cfg.get("theta", 0.0), cfg.get("delta", 1.0)  # 9A
    batch_replay, burst = cfg.get("batch_replay", batch_replay), cfg.get("burst", 1)
    nrem_gain = cfg.get("nrem_gain", nrem_gain)          # 10B: gentler consolidation for tiny buffers
    replay_noise = cfg.get("replay_noise", 0.0)          # 10B: replay is variable, not verbatim
    theta_r, frac_r = cfg.get("theta_r", 4.0), cfg.get("frac_r", 0.25)  # 10A homeostatic rotation
    theta_s = cfg.get("theta_s", 0.25)  # 10A2: ACh gate -- rotation only while novelty is high
    beta_nov = cfg.get("beta_nov", 1.3)  # 10A3: novelty = fast error EMA above its own slow baseline
    surprise_ema, surprise_slow, ach_on = None, None, []
    S = [None] + [torch.zeros(s) for s in hidden]  # unit-level sleep pressure: use since last consolidation
    gate_log = []
    Xtr, ytr, Xte, yte, ishape = load_data(cfg)
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    front = make_front(cfg, seed)
    net = CortexNet([front.n_dg if front else Xtr.shape[1], *hidden, 10], seed=seed,
                    input_shape=None if front else ishape,
                    **dict(V7, active_frac=af, conn_density=cfg.get("conn_density", V7["conn_density"])))
    net.front = front
    buf = Buffer(K, policy, g)
    tasks = [tuple(range(10))] if static else TASKS
    onehot = lambda y: torch.nn.functional.one_hot(torch.as_tensor(y), 10).float()
    use = [None] + [torch.full((s,), 0.5) for s in hidden]       # recent-use trace per hidden unit
    long_use = [None] + [torch.full((s,), 0.5) for s in hidden]  # long-term use trace
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan)
    replay_used, asleep_frac, eff_frac, t0 = 0, [], [], time.time()
    net.suppress = None
    gstep = 0  # waking batches since the start of the stream (cadence must not reset per epoch)
    Sr = [None] + [torch.zeros(s) for s in hidden]  # 10A: per-unit rest pressure (discharged by rest)
    for ti, classes in enumerate(tasks):
        m = np.isin(ytr, classes)
        Xt, yt = to_t(Xtr[m]), torch.as_tensor(ytr[m])
        n = len(yt)
        for ep in range(epochs_per_task):
            perm = torch.randperm(n, generator=g)
            for step, i in enumerate(range(0, n, batch_wake)):
                idx = perm[i:i + batch_wake]
                Xb, yb = Xt[idx], yt[idx]
                net.training = True
                x, a, eps = net.relax(Xb, onehot(yb), 0, 0.0)
                if mask_policy == "refractory":  # what fired now sits out the next competition
                    net.suppress = [None] + [(a[l] > 0).float().mean(0).gt(0).float() for l in range(1, net.L)]
                elif mask_policy == "refr_nov":
                    # 10A3: the absolute threshold cannot compare across data sets (the static
                    # stream's converged error sits above any theta the sequential stream dips
                    # under), so novelty is the FAST error EMA rising above its own SLOW baseline:
                    # adaptation, the way a neuromodulator actually measures surprise.
                    sb = float((eps[net.L] ** 2).sum(1).mean())
                    surprise_ema = sb if surprise_ema is None else 0.9 * surprise_ema + 0.1 * sb
                    surprise_slow = sb if surprise_slow is None else 0.998 * surprise_slow + 0.002 * sb
                    if surprise_ema >= beta_nov * surprise_slow:
                        net.suppress = [None] + [(a[l] > 0).float().mean(0).gt(0).float() for l in range(1, net.L)]
                        ach_on.append(1.0)
                    else:
                        net.suppress = None
                        ach_on.append(0.0)
                elif mask_policy == "refr_ach":
                    # 10A2: the rotation itself is the plain refractory rule, but a global ACh-like
                    # novelty signal decides WHEN it runs (Hasselmo): while the top-down error of the
                    # waking stream is high the system is in encoding mode and rotates; once the
                    # stream is predictable, rotation (and its i.i.d. price) switches off.
                    sb = float((eps[net.L] ** 2).sum(1).mean())
                    surprise_ema = sb if surprise_ema is None else 0.95 * surprise_ema + 0.05 * sb
                    if surprise_ema >= theta_s:
                        net.suppress = [None] + [(a[l] > 0).float().mean(0).gt(0).float() for l in range(1, net.L)]
                        ach_on.append(1.0)
                    else:
                        net.suppress = None
                        ach_on.append(0.0)
                elif mask_policy in ("refr_press", "refr_frac"):
                    # 10A: rest is forced by ACCUMULATED use (a per-unit Process S discharged by the
                    # rest itself, Vyazovskiy 2011), not by the last batch alone.  On i.i.d. data
                    # pressure spreads over many units, so far fewer are benched at once.
                    sup = [None]
                    for l in range(1, net.L):
                        fired_now = (a[l] > 0).float().mean(0)  # fraction of the batch's samples
                        Sr[l] = Sr[l] + fired_now
                        if mask_policy == "refr_press":
                            s_l = (Sr[l] >= theta_r).float()
                        else:
                            k_r = max(1, int(round(frac_r * Sr[l].numel())))
                            thr_r = Sr[l].topk(k_r).values[-1]
                            s_l = ((Sr[l] >= thr_r) & (Sr[l] > 0)).float()
                        Sr[l] = Sr[l] * (1.0 - s_l)  # the rest they are about to take discharges S
                        sup.append(s_l)
                    net.suppress = sup
                # Adam moves ~eta per step: with a batch of 64 there are 4x the steps of the
                # batch-256 protocol, so the waking step is scaled to keep the drift per epoch equal
                net.local_update(x, a, eps, 1e-3 * batch_wake / 256, 1e-3)
                buf.offer(Xb, yb, (eps[net.L] ** 2).sum(1))
                # who is awake for this input?
                awake = [None]
                for l in range(1, net.L):
                    fired = (a[l] > 0).float().mean(0)
                    S[l] = S[l] + fired                             # Process S: rises with use
                    use[l] = 0.9 * use[l] + 0.1 * fired            # recent use, ~10 batches
                    long_use[l] = 0.995 * long_use[l] + 0.005 * fired  # long-term use, ~200 batches
                    if mask_policy in ("silent", "refractory", "refr_press", "refr_frac", "refr_ach", "refr_nov"):
                        awake.append((fired > 0).float())
                    elif mask_policy == "idle":
                        awake.append((use[l] >= use[l].median()).float())   # asleep = idle half
                    elif mask_policy == "used":
                        awake.append((use[l] < use[l].median()).float())    # asleep = used half
                    elif mask_policy == "old":
                        # asleep = units that were used over the long run but not recently:
                        # the carriers of older memories, which the current input leaves alone
                        ratio = use[l] / (long_use[l] + 1e-6)
                        awake.append((ratio >= ratio.median()).float())
                    else:
                        awake.append(torch.zeros_like(fired))               # none: everyone may learn
                gstep += 1
                do_replay = bool(K) and gstep % cadence == 0
                if gate == "pressure":  # replay iff the units asleep for this input are under enough pressure
                    asleep_S = torch.cat([S[l][awake[l] == 0] for l in range(1, net.L)])
                    p = float(asleep_S.mean()) if asleep_S.numel() else 0.0
                    do_replay = bool(K) and p >= theta
                    gate_log.append(p)
                elif gate == "surprise":  # replay iff the waking batch was surprising
                    surp = float((eps[net.L] ** 2).sum(1).mean())
                    do_replay = bool(K) and surp >= theta
                    gate_log.append(surp)
                for _burst_i in range(burst if do_replay else 0):
                    Xr, yr = buf.sample(batch_replay)
                    if Xr is not None:
                        if replay_noise:
                            Xr = Xr + replay_noise * torch.randn(Xr.shape, generator=g)
                        # synapse masks: allowed iff pre asleep or post asleep
                        pre_awake = [torch.ones(net.sizes[0])] + [awake[l] for l in range(1, net.L)]  # input is always "awake"
                        post_awake = [None] + [awake[l] for l in range(1, net.L)] + [torch.ones(10)]
                        syn, bias = [None], [None]
                        for l in range(1, net.L + 1):
                            if l == net.L and readout == "free":
                                syn.append(torch.ones(10, hidden[-1])); bias.append(torch.ones(10))
                            else:
                                allowed = 1.0 - post_awake[l][:, None] * pre_awake[l - 1][None, :]
                                syn.append(allowed); bias.append(1.0 - post_awake[l])
                        asleep_frac.append([float(1 - awake[l].mean()) for l in range(1, net.L)])
                        net.training = False
                        sup_saved, net.suppress = net.suppress, None  # replay competes freely
                        xr, ar, er = net.relax(Xr, onehot(yr), 0, 0.0)
                        net.suppress = sup_saved
                        # effective consolidation capacity: of the units the replay batch fires,
                        # what fraction is asleep (and so allowed to learn from it)?
                        eff_frac.append([float(((1 - awake[l]) * ((ar[l] > 0).float().mean(0) > 0).float()).sum()
                                               / max(1.0, float(((ar[l] > 0).float().mean(0) > 0).float().sum())))
                                         for l in range(1, net.L)])
                        net.local_update(xr, ar, er, 1e-3 * nrem_gain, 1e-3, syn_mask=syn, bias_mask=bias)
                        replay_used += 1
                        if gate == "pressure":  # the sleep the asleep units just had discharges their pressure
                            for l in range(1, net.L):
                                S[l] = S[l] * (1.0 - delta * (1.0 - awake[l]))
            if night and K:  # 8B combo: the concentrated night on top of the daytime trickle
                net.training = False
                net.suppress = None
                for _ in range(20):
                    Xr, yr = buf.sample(256)
                    if Xr is None:
                        break
                    ctx_step(net, Xr, onehot(yr), 1e-3 * nrem_gain)
                    replay_used += 1
        net.suppress = None
        S_te = predict("ctx", net, Xte)
        pred = S_te.argmax(1)
        for tj, cj in enumerate(tasks[:ti + 1]):
            mt = np.isin(yte, cj)
            acc_matrix[ti, tj] = float((pred[mt] == yte[mt]).mean())
    S_te = predict("ctx", net, Xte)
    final_acc = float((S_te.argmax(1) == yte).mean())
    T = len(tasks)
    forgetting = float(np.mean([acc_matrix[j, j] - acc_matrix[T - 1, j] for j in range(T - 1)])) if T > 1 else 0.0
    globals()["LAST_NET"] = net
    af = np.mean(asleep_frac, axis=0).tolist() if asleep_frac else None
    ef = np.mean(eff_frac, axis=0).tolist() if eff_frac else None
    out = dict(config=config, seed=seed, model="ctx", buffer=K, policy=policy, replay="local", static=static,
               schedule="local", final_acc=final_acc, forgetting=forgetting, acc_matrix=acc_matrix.tolist(),
               buffer_classes=buf.class_counts(), replay_used=replay_used, asleep_frac=af, eff_frac=ef,
               hidden=hidden, active_frac=af_cfg(cfg), overlap=task_overlap(net, Xte, yte, tasks),
               overlap_in=task_overlap(net, Xte, yte, tasks, layers=(0,))[0], dim=code_dim(net, Xte),
               synops=net.last_cost["synops"], gate=gate, theta=theta, rest_frac=(np.mean(asleep_frac, axis=0).tolist() if asleep_frac else None),
               ach_frac=(float(np.mean(ach_on)) if ach_on else None),
               gate_signal=(float(np.mean(gate_log)) if gate_log else None), fit_s=time.time() - t0)
    print(f"  {config:26s} seed {seed}: overlap-in {out['overlap_in']:.2f}  dim {[round(v, 1) for v in out['dim']]}  "
          f"synops/sample {out['synops']:.0f}", flush=True)
    print(f"  {config:26s} seed {seed}: FINAL acc {final_acc:.4f}  forgetting {forgetting:.4f}  replay {replay_used}  "
          f"asleep {None if af is None else [round(v, 2) for v in af]}  replay-active&asleep {None if ef is None else [round(v, 2) for v in ef]}  "
          f"task-overlap {[round(v, 2) for v in out['overlap']]}  ({out['fit_s'] / 60:.1f} min)", flush=True)
    return out


def af_cfg(cfg):
    return cfg.get("active_frac", V7["active_frac"])


def task_overlap(net, Xte, yte, tasks, thresh=0.05, layers=None):
    """Code orthogonality: per hidden layer (or the given layers; 0 = the cortex input, i.e. the DG
    code), the mean Jaccard overlap between the sets of units that are active (fire on > thresh of
    the task's test samples) for different tasks."""
    layers = list(range(1, net.L)) if layers is None else list(layers)
    net.training = False
    sets = []
    with torch.no_grad():
        for classes in tasks:
            mt = np.isin(yte, classes)
            _, a = net.forward(to_t(Xte[mt]))
            sets.append([((a[l] > 0).float().mean(0) > thresh) for l in layers])
    out = []
    for l in range(len(layers)):
        js = []
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                inter = (sets[i][l] & sets[j][l]).sum().item()
                union = (sets[i][l] | sets[j][l]).sum().item()
                js.append(inter / max(union, 1))
        out.append(float(np.mean(js)) if js else float("nan"))
    return out


def part_path(config, seed):
    return os.path.join(PARTS, f"{config}_s{seed}.pkl")


def summary():
    rows = []
    for f in sorted(os.listdir(PARTS)) if os.path.isdir(PARTS) else []:
        with open(os.path.join(PARTS, f), "rb") as fh:
            rows.append(pickle.load(fh))
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows)
    agg = df.groupby("config").agg(seeds=("seed", "count"), acc=("final_acc", "mean"), acc_sd=("final_acc", "std"),
                                   forgetting=("forgetting", "mean"), fit_min=("fit_s", lambda s: s.mean() / 60))
    agg = agg.reindex([c for c in CONFIGS if c in agg.index])
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    os.makedirs(OUT, exist_ok=True)
    df.drop(columns="acc_matrix").to_csv(os.path.join(OUT, "runs.csv"), index=False)
    agg.to_csv(os.path.join(OUT, "summary.csv"))
    print(f"\nwrote {OUT}/summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs-per-task", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--nrem-batches", type=int, default=20)
    ap.add_argument("--nrem-gain", type=float, default=3.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return
    os.makedirs(PARTS, exist_ok=True)
    configs = list(CONFIGS) if args.configs == ["all"] else args.configs
    for seed in args.seeds:
        for config in configs:
            if args.resume and os.path.exists(part_path(config, seed)):
                print(f"  skip {config} seed {seed}", flush=True)
                continue
            if CONFIGS[config].get("schedule") == "internal":
                out = run_stream(config, seed, args.epochs_per_task, args.batch, args.nrem_gain)
            elif CONFIGS[config].get("schedule") == "local":
                c = CONFIGS[config]
                out = run_local(config, seed, args.epochs_per_task, c.get("batch_wake", 64), args.batch, args.nrem_gain,
                                cadence=c.get("cadence", 9))
            else:
                out = run(config, seed, args.epochs_per_task, args.batch, args.nrem_batches, args.nrem_gain)
            tmp = part_path(config, seed) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(out, f)
            os.replace(tmp, part_path(config, seed))


if __name__ == "__main__":
    main()
