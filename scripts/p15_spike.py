"""Phase 15E: re-measure spiking inference energy on the CURRENT default stack (stateless SGD,
bout-committed gate M=2048) and its controls, static MNIST -- the paper's Energy numbers were
still from the v10 Adam/novelty-gate stack.

Energy convention: 0.9 pJ per synaptic event (AC) vs 4.6 pJ per MAC for the dense rate net."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import run_seq as RS  # noqa: E402
import run_mnist as RM  # noqa: E402
from models.spiking import calibrate_thresholds, spike_eval  # noqa: E402

AC_PJ, MAC_PJ = 0.9, 4.6

CONFIGS = (
    "static_g13_sgd_block2048_w512s5",  # new default: SGD + bout gate
    "static_g12_sgd_silent_w512s5",     # rotation-free control (static-side default)
    "static_g12_sgd_refr_w512s5",       # always-on rotation (calibration-interaction check)
)

for config in CONFIGS:
    print(f"=== {config}", flush=True)
    cfg = RS.CONFIGS[config]
    out = RS.run_local(config, 0, 5, 16, cfg.get("batch_replay", 256), 3.0,
                       cadence=cfg.get("cadence", 1))
    net = RS.LAST_NET
    Xtr, ytr, Xte, yte = RM.load_mnist()
    mu, sd, Xte_raw = RM._MU, RM._SD, RM._XTE_RAW
    dense_macs = sum(net.sizes[i] * net.sizes[i + 1] for i in range(len(net.sizes) - 1))
    print(f"  rate net: acc {out['final_acc']:.4f}  synops/sample {out['synops']:.0f}  "
          f"dense {dense_macs} MACs = {dense_macs * MAC_PJ / 1e3:.0f} nJ  "
          f"event-driven rate = {out['synops'] * MAC_PJ / 1e3:.0f} nJ", flush=True)
    thr = calibrate_thresholds(net, RS.to_t(Xte[:2000]))
    for T_s in (4, 8, 16, 32):
        acc, spikes, events = spike_eval(net, Xte_raw, mu, sd, yte, T_s, thr, seed=0)
        print(f"  T_s={T_s:3d}: acc {acc:.4f}  spikes/sample {sum(spikes):.0f}  events {events:.0f}  "
              f"energy {events * AC_PJ / 1e3:.1f} nJ", flush=True)
