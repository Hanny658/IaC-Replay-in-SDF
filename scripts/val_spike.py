"""Spiking-inference energy under the held-out protocol: train the static default (and the
rotation-free control) on the 90% split, calibrate thresholds on training images, and evaluate
the rate net and its spiking conversion on the held-out tenth (raw intensities).

Energy convention as in scripts/p16_spike.py: 0.9 pJ per synaptic event (AC) vs 4.6 pJ per MAC."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_seq as RS  # noqa: E402
import run_mnist as RM  # noqa: E402
from models.spiking import calibrate_thresholds, spike_eval  # noqa: E402

AC_PJ, MAC_PJ = 0.9, 4.6
RS.VAL = True  # train on the 90% split, evaluate on the held-out tenth

CONFIGS = tuple(sys.argv[1:]) or ("g16_sgd_refr_s10_w512_static", "g16_sgd_silent_s10_w512_static")

for config in CONFIGS:
    print(f"=== {config} (held-out protocol)", flush=True)
    cfg = RS.CONFIGS[config]
    out = RS.run_local(config, 0, 5, 16, cfg.get("batch_replay", 256), 3.0, cadence=cfg.get("cadence", 1))
    net = RS.LAST_NET
    Xtr_std, ytr, _, _ = RM.load_mnist()  # full-train standardisation, as used by run_seq
    mu, sd = RM._MU, RM._SD
    raw = RM._idx(os.path.join(RM.DATA, "train-images-idx3-ubyte.gz")).reshape(-1, 784).astype(np.float32) / 255
    _, _, Xva_raw, yva = RS._val_split(raw, ytr)        # same stratified split (fixed seed)
    Xtr90_std, _, _, _ = RS._val_split(Xtr_std, ytr)
    dense_macs = sum(net.sizes[i] * net.sizes[i + 1] for i in range(len(net.sizes) - 1))
    print(f"  rate net (held-out): acc {out['final_acc']:.4f}  synops/sample {out['synops']:.0f}  "
          f"dense {dense_macs} MACs = {dense_macs * MAC_PJ / 1e3:.0f} nJ  "
          f"event-driven rate = {out['synops'] * MAC_PJ / 1e3:.0f} nJ", flush=True)
    thr = calibrate_thresholds(net, RS.to_t(Xtr90_std[:2000]))  # thresholds from TRAINING images
    for T_s in (4, 8, 16, 32):
        acc, spikes, events = spike_eval(net, Xva_raw, mu, sd, yva, T_s, thr, seed=0)
        print(f"  T_s={T_s:3d}: acc {acc:.4f}  spikes/sample {sum(spikes):.0f}  events {events:.0f}  "
              f"energy {events * AC_PJ / 1e3:.1f} nJ", flush=True)
