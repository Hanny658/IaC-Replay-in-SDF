"""Phase 10C-b: re-measure spiking inference energy on the v9 net (w512s5, refractory local sleep,
replay batch 16) against the same net trained without local sleep, on static MNIST.

Energy convention: 0.9 pJ per synaptic event (AC) vs 4.6 pJ per MAC for the dense rate net."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_seq as RS  # noqa: E402
import run_mnist as RM  # noqa: E402
from models.spiking import calibrate_thresholds, spike_eval  # noqa: E402

AC_PJ, MAC_PJ = 0.9, 4.6

for config in ("static_loc16_refractory_br16_w512s5", "static_loc16_none_br16_w512s5"):
    print(f"=== {config}", flush=True)
    out = RS.run_local(config, 0, 5, 16, RS.CONFIGS[config].get("batch_replay", 256), 3.0,
                       cadence=RS.CONFIGS[config].get("cadence", 1))
    net = RS.LAST_NET
    Xtr, ytr, Xte, yte = RM.load_mnist()
    mu, sd, Xte_raw = RM._MU, RM._SD, RM._XTE_RAW
    dense_macs = sum(net.sizes[i] * net.sizes[i + 1] for i in range(len(net.sizes) - 1))
    print(f"  rate net: acc {out['final_acc']:.4f}  synops/sample {out['synops']:.0f}  "
          f"dense {dense_macs} MACs = {dense_macs * MAC_PJ / 1e3:.0f} nJ  "
          f"event-driven rate = {out['synops'] * MAC_PJ / 1e3:.0f} nJ", flush=True)
    thr = calibrate_thresholds(net, RS.to_t(Xtr[:2000]))  # thresholds from TRAINING images
    for T_s in (4, 8, 16, 32):
        acc, spikes, events = spike_eval(net, Xte_raw, mu, sd, yte, T_s, thr, seed=0)
        print(f"  T_s={T_s:3d}: acc {acc:.4f}  spikes/sample {sum(spikes):.0f}  events {events:.0f}  "
              f"energy {events * AC_PJ / 1e3:.1f} nJ", flush=True)
