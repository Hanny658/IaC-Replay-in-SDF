"""Continual evaluation during the stream: the retention curve and the churn of the served
prediction.  Usage: MLPC_VAL=1 python report/make_fig_stability.py
"""
import glob
import os
import pickle
import re

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

# label, config, batch (to put every schedule on a samples axis), colour, style
CURVES = [
    ("isolated replay + rotation (ours)", "g29_stab_ours", 16, "C0", "-"),
    ("rotation, unmasked replay", "g29_stab_rot_noiso", 16, "C4", "--"),
    ("isolated replay, no rotation", "g29_stab_silent", 16, "C7", "-."),
    ("BP + DER++ (same schedule)", "g29_stab_derpp", 16, "C2", "-"),
    ("offline rehearsal", "g29_stab_night", 256, "C3", "-"),
]
BARS = [
    ("iso\n+ rot", "g29_stab_ours", "C0"),
    ("rot\nonly", "g29_stab_rot_noiso", "C4"),
    ("iso\nonly", "g29_stab_silent", "C7"),
    ("neither", "g29_stab_unmasked", "C1"),
]


def runs(cfg):
    out = []
    for f in sorted(glob.glob(os.path.join(PARTS, f"{PREFIX}{cfg}_s[0-9].pkl"))):
        with open(f, "rb") as fh:
            out.append(pickle.load(fh))
    return out


fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [2.05, 1]})

for label, cfg, batch, colour, ls in CURVES:
    rs = [d for d in runs(cfg) if d.get("eval_trace")]
    if not rs:
        print("skip", cfg)
        continue
    n = min(len(d["eval_trace"]) for d in rs)
    x = np.array([r[0] for r in rs[0]["eval_trace"][:n]], dtype=float) * batch / 1000.0
    y = np.stack([[r[2] * 100 for r in d["eval_trace"][:n]] for d in rs])
    ax.plot(x, y.mean(0), color=colour, lw=1.2, ls=ls, label=label)
    ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=colour, alpha=0.15, lw=0)
    print(f"{label:34s} anytime {y.mean():5.2f}  worst {y.min():5.2f}  ({len(rs)} seeds)")

# task switches, in thousands of waking samples (nine tenths of the training set, five tasks)
switches = [d for d in (np.array([r[0] for r in runs(CURVES[0][1])[0]["eval_trace"]])
                        [np.diff([r[1] for r in runs(CURVES[0][1])[0]["eval_trace"]], prepend=0) > 0] * 16 / 1000.0)]
for sdx in switches:
    ax.axvline(sdx, color="black", lw=0.6, alpha=0.35)
ax.set_xlabel("waking samples seen (thousands)")
ax.set_ylabel("accuracy on classes seen (%)")
ax.set_ylim(15, 101)
ax.legend(fontsize=6.5, loc="lower center", ncol=2, frameon=True, framealpha=0.92)

for i, (label, cfg, colour) in enumerate(BARS):
    vals = [100 * d["drift"][1] for d in runs(cfg) if d.get("drift")]
    if not vals:
        continue
    bx.bar(i, np.mean(vals), 0.62, color=colour, yerr=(np.std(vals, ddof=1) if len(vals) > 1 else 0),
           capsize=3, error_kw={"lw": 1})
    bx.scatter([i] * len(vals), vals, s=5, color="black", zorder=3)
    print(f"{label.replace(chr(10), ' '):34s} churn {np.mean(vals):5.2f}%  ({len(vals)} seeds)")
bx.set_xticks(range(len(BARS)))
bx.set_xticklabels([b[0] for b in BARS], fontsize=7)
bx.set_ylabel("served predictions changed\nper replay update (%)", fontsize=8)
bx.set_yscale("log")
bx.set_yticks([0.3, 0.5, 1, 2, 5])
bx.set_yticklabels(["0.3", "0.5", "1", "2", "5"])
bx.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
bx.tick_params(axis="y", which="minor", length=0)

fig.tight_layout(w_pad=1.6)
out = os.path.join(FIGS, "fig_stability.pdf")
fig.savefig(out)
print("wrote", out)
