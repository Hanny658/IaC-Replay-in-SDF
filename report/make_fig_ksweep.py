"""Buffer-size axis on split-MNIST (held-out split): sequential accuracy against K for the proposed
system and the replay references.  Config names per series are listed in SERIES; run after the
K = 200 / 5000 held-out runs exist.  Usage: MLPC_VAL=1 python report/make_fig_ksweep.py
"""
import glob
import os
import pickle
import re
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight", "figure.dpi": 150})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "results", "bio", "seq", "parts")
FIGS = os.path.join(ROOT, "report", "figs")
PREFIX = "val_" if os.environ.get("MLPC_VAL") else ""
KS = [200, 1000, 5000]
SERIES = [  # label, colour, marker, configs at K = 200 / 1000 / 5000
    ("isolated replay + rotation (ours)", "C0", "o",
     ["g27_refr_K200", "g16_sgd_refr_s10_w512", "g27_refr_K5000"]),
    ("offline rehearsal (narrow learner)", "C3", "s",
     ["ctx_nrem_rand_200", "ctx_nrem_rand_1000", "ctx_nrem_rand_5000"]),
    ("BP + ER", "C7", "^", ["bp_er_200", "bp_er_1000", "bp_er_5000"]),
    ("BP + DER++", "C2", "D", [os.environ.get("DERPP_K200", "bp_derpp_K200"),
                              os.environ.get("DERPP_K1000", "bp_derpp_K1000"),
                              os.environ.get("DERPP_K5000", "bp_derpp_K5000")]),
]


def stat(cfg):
    accs = []
    for f in glob.glob(os.path.join(PARTS, f"{PREFIX}{cfg}_s[0-9]*.pkl")):
        seed = int(re.search(r"_s(\d+)\.pkl$", f).group(1))
        if seed >= 100:
            continue
        with open(f, "rb") as fh:
            accs.append(100 * pickle.load(fh)["final_acc"])
    if not accs:
        return np.nan, np.nan, 0
    return float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, len(accs)


fig, ax = plt.subplots(figsize=(3.6, 2.7))
for label, colour, marker, cfgs in SERIES:
    stats = [stat(c) for c in cfgs]
    means = np.array([s[0] for s in stats]); sds = np.array([s[1] for s in stats])
    if np.all(np.isnan(means)):
        print("skip", label, "(no runs)"); continue
    print(f"{label:38s}", "  ".join(f"K={k}: {m:5.1f}+-{s:4.1f} (n={n})" for k, (m, s, n) in zip(KS, stats)))
    ax.errorbar(KS, means, yerr=sds, color=colour, marker=marker, ms=4, lw=1.3, capsize=2, label=label)
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.tick_params(axis="x", which="minor", length=0)
ax.set_xlabel("buffer size $K$ (samples)"); ax.set_ylabel("final accuracy (%)")
ax.legend(fontsize=7, loc="lower right", frameon=True)
out = os.path.join(FIGS, "fig_ksweep.pdf")
fig.savefig(out); print("wrote", out)
