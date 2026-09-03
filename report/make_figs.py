"""Result figures for the preprint, drawn live from the per-run checkpoints."""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(os.path.dirname(HERE), "results", "bio", "seq", "parts")
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4, "legend.frameon": False})
C = {"refr": "#0072B2", "none": "#E69F00", "night": "#009E73", "press": "#CC79A7",
     "burst": "#56B4E9", "surp": "#D55E00", "grey": "#666666"}


def stat(cfg):
    accs, reps = [], []
    for s in range(6):
        p = os.path.join(PARTS, f"{cfg}_s{s}.pkl")
        if os.path.exists(p):
            r = pickle.load(open(p, "rb"))
            accs.append(r["final_acc"]); reps.append(r.get("replay_used", np.nan))
    if not accs:
        raise KeyError(cfg)
    return 100 * np.mean(accs), 100 * (np.std(accs, ddof=1) if len(accs) > 1 else 0.0), np.nanmean(reps)


def series(cfgs):
    out = [stat(c) for c in cfgs]
    return [o[2] for o in out], [o[0] for o in out], [o[1] for o in out]


# ---------------- Fig: replay timing (s10 record substrate, stateless SGD) ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
PTS = [  # label, cfg, marker, color
    ("even trickle (cad 8)", "g17_sgd_cad8_br16_s10", "o", C["none"]),
    ("clocked burst 16/128", "g17_sgd_burst16_128_s10", "s", C["burst"]),
    ("pressure bursts", "g17_sgd_pburst16_s10", "*", C["refr"]),
    ("surprise bursts", "g17_sgd_sburst16_s10", "x", C["surp"]),
    ("every batch (default)", "g16_sgd_refr_s10_w512", "D", C["refr"]),
]
for lab, cfg, mk, col in PTS:
    m, e, r = stat(cfg)
    ax.errorbar(r if r == r else 18760, m, e, marker=mk, ms=6 if mk == "*" else 4,
                lw=0, elinewidth=1.0, capsize=2, color=col, label=lab)
ax.axhline(stat("ctx_nrem_rand_1000")[0], color=C["night"], lw=1, ls="--")
ax.text(1500, stat("ctx_nrem_rand_1000")[0] + 0.5, "offline night (480 batches)", color=C["night"], fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("replay batches during wake"); ax.set_ylabel("final accuracy (%)")
ax.legend(fontsize=6.3, loc="lower right")
fig.savefig(os.path.join(FIGS, "fig_timing.pdf")); plt.close(fig)

# ---------------- Fig: replay batch size (s10 record substrate, stateless SGD) ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bs = [4, 8, 16, 64, 256]
cfgs = ["g17_sgd_br4_s10", "g17_sgd_br8_s10", "g16_sgd_refr_s10_w512", "g17_sgd_br64_s10", "g17_sgd_br256_s10"]
_, y, e = series(cfgs)
ax.errorbar(bs, y, e, marker="o", ms=3.5, lw=1.2, color=C["refr"], label="refractory local sleep")
m, e1, _ = stat("g17_sgd_none_s10")
ax.errorbar([16], [m], [e1], marker="s", ms=4, lw=0, elinewidth=1.0, capsize=2, color=C["none"], label="unmasked interleaved (batch 16)")
m, e1, _ = stat("g16_sgd_silent_s10_w512")
ax.errorbar([16], [m], [e1], marker="^", ms=4, lw=0, elinewidth=1.0, capsize=2, color=C["grey"], label="silent mask (batch 16)")
_, y, e = series(["g17_night_nb16_s10", "g17_night_s10"])
ax.errorbar([16, 256], y, e, marker="D", ms=3.5, lw=1.2, color=C["night"], label="offline night")
ax.set_xscale("log", base=2); ax.set_xlabel("replay batch size"); ax.set_ylabel("final accuracy (%)")
ax.legend(fontsize=6.3, loc="lower right")
fig.savefig(os.path.join(FIGS, "fig_batch.pdf")); plt.close(fig)

# ---------------- Fig: rotation price/gain and the novelty gate ----------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax = axes[0]
names = ["no rotation", "refractory", "pressure\n($\\theta_r{=}4$)", "novelty gate\n($\\beta{=}1.5$)"]
cfgs = ["static_loc16_none_w512s5", "static_loc16_refractory_br16_w512s5", "static_g10_rp4_w512s5", "static_g10_nov15_w512s5"]
y, e = zip(*[stat(c)[:2] for c in cfgs])
ax.bar(range(4), y, yerr=e, color=[C["grey"], C["refr"], C["press"], C["night"]], width=0.62)
ax.set_xticks(range(4), names, fontsize=6.3); ax.set_ylim(88, 96.5); ax.set_ylabel("accuracy (%)")
ax.set_title("MNIST i.i.d. (static)", fontsize=8)
ax = axes[1]
cfgs = ["g9a_w512s5_none_br16", "g9a_w512s5_br16", "g10_rp4_w512s5", "g10_nov15_w512s5"]
y, e = zip(*[stat(c)[:2] for c in cfgs])
ax.bar(range(4), y, yerr=e, color=[C["grey"], C["refr"], C["press"], C["night"]], width=0.62)
ax.set_xticks(range(4), names, fontsize=6.3); ax.set_ylim(75, 93)
ax.set_title("split-MNIST (sequential)", fontsize=8)
ax = axes[2]
cfgs = ["cif_static_none", "cif_static_refr", "cif_static_rp4"]
y, e = zip(*[stat(c)[:2] for c in cfgs])
ax.bar(np.arange(3) - 0.19, y, 0.36, yerr=e, color="#88bbdd", label="static")
cfgs = ["cif_none_br16", "cif_refr_br16", "cif_rp4_br16"]
y, e = zip(*[stat(c)[:2] for c in cfgs])
ax.bar(np.arange(3) + 0.19, y, 0.36, yerr=e, color="#114477", label="sequential")
ax.set_xticks(range(3), ["no rotation", "refractory", "pressure"], fontsize=6.3)
ax.set_title("split CIFAR-10 (grayscale)", fontsize=8); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_rotation.pdf")); plt.close(fig)

# ---------------- Fig: K=200 mechanism ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.3))
rows = [("offline night", "g11_k200_night", C["night"]),
        ("night + replay noise", "g11_k200_night_n02", C["night"]),
        ("local + night (noisy)", "g11_k200_combo_n02", C["refr"]),
        ("pressure bursts + noise", "g10_k200_pb16_n02", C["refr"]),
        ("local trickle + noise", "g10_k200_n02", C["refr"]),
        ("local trickle (plain)", "g9a_w512s5_br16_K200", C["refr"]),
        ("unmasked trickle + noise", "g11_k200_none_g1_n02", C["none"])]
labels, ys, es, cols = [], [], [], []
for lab, cfg, col in rows:
    try:
        m, s, _ = stat(cfg)
    except KeyError:
        continue
    labels.append(lab); ys.append(m); es.append(s); cols.append(col)
ypos = np.arange(len(labels))[::-1]
ax.barh(ypos, ys, xerr=es, color=cols, height=0.62)
ax.set_yticks(ypos, labels, fontsize=7); ax.set_xlim(65, 90); ax.set_xlabel("final accuracy (%)")
ax.set_title("K = 200: what the night still buys", fontsize=8)
fig.savefig(os.path.join(FIGS, "fig_k200.pdf")); plt.close(fig)

print("figures written to", FIGS)


# ---------------- Fig: the operating frontier (phases 12-15, stateless optimiser) ----------------
def stat6(cfg):
    accs = []
    for s in range(6):
        p = os.path.join(PARTS, f"{cfg}_s{s}.pkl")
        if os.path.exists(p):
            accs.append(pickle.load(open(p, "rb"))["final_acc"])
    return 100 * np.mean(accs), 100 * (np.std(accs, ddof=1) if len(accs) > 1 else 0.0)


FRONTIER = [  # s10 record substrate: label, seq cfg, static cfg, non-dominated, offset
    ("always-on", "g16_sgd_refr_s10_w512", "g16_sgd_refr_s10_w512_static", True, (5, -3)),
    ("bout gate", "g16_sgd_block2048_s10_w512", "g16_sgd_block2048_s10_w512_static", True, (-40, 5)),
    ("per-batch gate", "g17_sgd_prog05_s10", "static_g17_sgd_prog05_s10", False, (6, -3)),
    ("silent", "g16_sgd_silent_s10_w512", "g16_sgd_silent_s10_w512_static", False, (-24, -12)),
    ("$-$isolation", "g17_sgd_none_s10", "static_g17_sgd_none_s10", True, (5, -3)),
    # phase 19: the adaptive decay controller retires silent and the per-batch gate
    ("adaptive $\\lambda$ (grad)", "g19_kad_mnist", "g19_kad_mnist_static", True, (6, -2)),
    ("adaptive $\\lambda$ (drive)", "g19b_mnist", "g19b_mnist_static", True, (-64, -3)),
    ("$+$anchor", "g17_sgd_anchor_s10", "static_g17_sgd_anchor_s10", False, (5, -9)),
    ("masked Adam", "g17_adam_refr_s10", "static_g17_adam_refr_s10", False, (-56, -5)),
    ("Adam, leak", "g17_adam_leak_s10", "static_g17_adam_leak_s10", False, (5, -3)),
]
HIST5 = [  # the historical 5% family (faded)
    ("g12_sgd_refr_w512s5", "static_g12_sgd_refr_w512s5"),
    ("g13_sgd_block2048_w512s5", "static_g13_sgd_block2048_w512s5"),
    ("g12_sgd_silent_w512s5", "static_g12_sgd_silent_w512s5"),
    ("g12_sgd_prog05_w512s5", "static_g12_sgd_prog05_w512s5"),
]
fig, ax = plt.subplots(figsize=(3.8, 2.9))
for cs, ct in HIST5:
    (xm, _), (ym, _) = stat6(cs), stat6(ct)
    ax.plot(xm, ym, marker="o", ms=3, color=C["grey"], alpha=0.45, lw=0)
ax.annotate("historical $5\%$ family", (89.2, 93.6), fontsize=6.5, color=C["grey"], alpha=0.8)
front = []
for name, cs, ct, onf, off in FRONTIER:
    (xm, xs), (ym, ys) = stat6(cs), stat6(ct)
    ax.errorbar(xm, ym, xerr=xs, yerr=ys, marker="o", ms=4,
                color=C["refr"] if onf else C["press"], lw=0, elinewidth=0.8, capsize=1.5)
    ax.annotate(name, (xm, ym), textcoords="offset points", xytext=off, fontsize=7,
                color=C["refr"] if onf else C["press"])
    if onf:
        front.append((xm, ym))
front.sort()
ax.plot([p[0] for p in front], [p[1] for p in front], color=C["refr"], lw=0.8, alpha=0.45, zorder=0)
ax.set_xlabel("sequential (split-MNIST) accuracy (%)")
ax.set_ylabel("i.i.d. (static) accuracy (%)")
fig.savefig(os.path.join(FIGS, "fig_frontier.pdf")); plt.close(fig)
print("frontier written")
