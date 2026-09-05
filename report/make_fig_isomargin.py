"""Appendix figure: what isolation adds on top of rotation as the replay micro-batch shrinks.
Left: split-MNIST accuracy vs replay batch size for the full system (isolated replay) and for
rotation with unmasked replay, every seed shown, mean +- sd where the seeds agree; unmasked
seeds below the axis floor are drawn at the floor with their values. Right: the seed-paired
margin (isolated minus unmasked) with a 95% paired-bootstrap CI at the batch sizes where the
unmasked run does not collapse. Drawn from the per-run checkpoints; MLPC_VAL=1 selects the
held-out-split pickles."""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(os.path.dirname(HERE), "results", "bio", "seq", "parts")
FIGS = os.path.join(HERE, "figs")
PREFIX = "val_" if os.environ.get("MLPC_VAL") else ""
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4, "legend.frameon": False})
C = {"refr": "#0072B2", "press": "#CC79A7"}
FLOOR = 72.0  # left-panel axis floor; collapsed seeds are drawn at the floor and labelled

ISO = {2: "g17_sgd_br2_s10", 4: "g17_sgd_br4_s10", 8: "g17_sgd_br8_s10", 16: "g16_sgd_refr_s10_w512",
       64: "g17_sgd_br64_s10", 256: "g17_sgd_br256_s10"}
NOISO = {2: "g26_ctrl_rot_noiso_br2", 4: "g26_ctrl_rot_noiso_br4", 8: "g26_ctrl_rot_noiso_br8",
         16: "g26_ctrl_rot_noiso"}


def seeds(cfg, n=6):
    out = {}
    for s in range(n):
        p = os.path.join(PARTS, f"{PREFIX}{cfg}_s{s}.pkl")
        if os.path.exists(p):
            out[s] = 100 * pickle.load(open(p, "rb"))["final_acc"]
    return out


def paired(a, b, B=20000, seed=0):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)]
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(d)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.2, 1]})
rng = np.random.RandomState(1)
for name, table, col, mk, ls in (("isolated replay (full system)", ISO, C["refr"], "o", "-"),
                                 ("rotation, unmasked replay", NOISO, C["press"], "v", "--")):
    xs, ys, es = [], [], []
    for b in sorted(table):
        v = seeds(table[b])
        if not v:
            continue
        vals = np.array(list(v.values()))
        jit = b * np.exp(rng.uniform(-0.08, 0.08, len(vals)))
        below = vals < FLOOR
        ax.scatter(jit[~below], vals[~below], s=7, color=col, alpha=0.4, lw=0, zorder=2)
        if below.any():
            ax.scatter(jit[below], np.full(below.sum(), FLOOR + 0.4), s=22, marker="v", color=col, lw=0, zorder=4)
            ax.annotate(", ".join(f"{x:.0f}" for x in sorted(vals[below])) + " (off axis)",
                        (b, FLOOR + 0.4), textcoords="offset points", xytext=(8, 4), fontsize=6.2, color=col)
        if below.any():
            continue  # a mean over collapsed and surviving seeds carries no information
        xs.append(b); ys.append(vals.mean()); es.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    ax.errorbar(xs, ys, es, marker=mk, ms=3.5, lw=1.2, ls=ls, color=col, label=name, zorder=3, capsize=2)
ax.set_ylim(FLOOR, 93.2)
ax.set_xscale("log", base=2); ax.set_xlabel("replay batch size"); ax.set_ylabel("final accuracy (%)")
ax.set_xticks([2, 4, 8, 16, 64, 256]); ax.set_xticklabels(["2", "4", "8", "16", "64", "256"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=6.8, loc="center right")

bs, ms, lo, hi = [], [], [], []
for b in sorted(NOISO):
    a, c = seeds(ISO[b]), seeds(NOISO[b])
    if not a or not c:
        continue
    if min(c.values()) < FLOOR:  # collapse regime: the paired mean is driven by the failures
        m, l, h, n = paired(a, c)
        ax2.text(0.04, 0.97, f"batch {b}: unmasked replay collapses in\n{sum(v < FLOOR for v in c.values())} of {n} seeds; "
                 f"paired mean {m:+.0f} [{l:+.1f}, {h:+.1f}]\n(off axis, not a margin)",
                 transform=ax2.transAxes, fontsize=6.2, ha="left", va="top", color=C["press"])
        ax2.scatter([b], [1.45], marker="^", s=22, color=C["press"], lw=0, zorder=4, clip_on=False)
        continue
    m, l, h, n = paired(a, c)
    bs.append(b); ms.append(m); lo.append(m - l); hi.append(h - m)
    ax2.annotate(f"{m:+.1f} (n={n})", (b, m), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=6.5)
ax2.axhline(0, color="#666666", lw=0.8)
ax2.errorbar(bs, ms, [lo, hi], marker="o", ms=4, lw=1.2, color=C["refr"], capsize=3)
ax2.set_xscale("log", base=2); ax2.set_xlabel("replay batch size")
ax2.set_ylabel("isolation margin (pts, seed-paired)")
ax2.set_xticks([2, 4, 8, 16]); ax2.set_xticklabels(["2", "4", "8", "16"])
ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.set_xlim(1.4, 24); ax2.set_ylim(-0.9, 1.6)
fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(FIGS, "fig_isomargin.pdf")); plt.close(fig)
print("fig_isomargin.pdf written; margins:", {b: round(m, 2) for b, m in zip(bs, ms)})
