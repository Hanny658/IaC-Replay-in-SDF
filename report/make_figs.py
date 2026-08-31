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
    for s in range(3):
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


# ---------------- Fig: replay timing / cost (accuracy vs replay events) ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
x, y, e = series(["g9a_w512s5_cad16", "g9a_w512s5_cad8", "g9a_w512s5_cad4", "g9a_w512s5_cad2"])
x, y, e = x + [18760], y + [stat("g8_w512s5_loc_refr")[0]], e + [stat("g8_w512s5_loc_refr")[1]]
ax.errorbar(x, y, e, marker="o", ms=3.5, lw=1.2, color=C["none"], label="clocked, even (cad $c$)")
x, y, e = series(["g9a_w512s5_burst4_32", "g9a_w512s5_burst8_64", "g9a_w512s5_burst16_128", "g9a_w512s5_burst32_256"])
ax.errorbar(x, y, e, marker="s", ms=3.5, lw=1.2, color=C["burst"], label="clocked bursts")
x, y, e = series(["g9a_w512s5_press16", "g9a_w512s5_press08", "g9a_w512s5_press04", "g9a_w512s5_press02"])
ax.errorbar(x, y, e, marker="^", ms=3.5, lw=1.2, color=C["press"], label="pressure, even")
x, y, e = series(["g9a_w512s5_pburst8_t32", "g9a_w512s5_pburst16_t64"])
ax.errorbar(x, y, e, marker="*", ms=7, lw=1.2, color=C["refr"], label="pressure bursts")
x, y, e = series(["g9a_w512s5_surp05", "g9a_w512s5_surp03", "g9a_w512s5_sburst8_03"])
ax.errorbar(x, y, e, marker="x", ms=4, lw=1.2, color=C["surp"], label="surprise-triggered")
ax.axhline(stat("ctx_nrem_rand_1000")[0], color=C["night"], lw=1, ls="--")
ax.text(220, stat("ctx_nrem_rand_1000")[0] + 0.8, "offline night (480 batches)", color=C["night"], fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("replay batches during wake"); ax.set_ylabel("final accuracy (%)")
ax.set_ylim(15, 95); ax.legend(fontsize=6.5, loc="lower right")
fig.savefig(os.path.join(FIGS, "fig_timing.pdf")); plt.close(fig)

# ---------------- Fig: replay batch size (isolation stabilises micro-batches) ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bs = [1, 2, 4, 8, 16, 32, 64, 256]
cfgs = ["g9a_w512s5_br1", "g9a_w512s5_br2", "g9a_w512s5_br4", "g9a_w512s5_br8",
        "g9a_w512s5_br16", "g9a_w512s5_br32", "g9a_w512s5_br64", "g8_w512s5_loc_refr"]
_, y, e = series(cfgs)
ax.errorbar(bs, y, e, marker="o", ms=3.5, lw=1.2, color=C["refr"], label="refractory local sleep")
_, y, e = series(["g9a_w512s5_none_br8", "g9a_w512s5_none_br16", "g8_w512s5_loc_none"])
ax.errorbar([8, 16, 256], y, e, marker="s", ms=3.5, lw=1.2, color=C["none"], label="unmasked interleaved ER")
_, y, e = series(["g9a_w512s5_silent_br16", "g8_w512s5_loc_silent"])
ax.errorbar([16, 256], y, e, marker="^", ms=3.5, lw=1.2, color=C["grey"], label="silent-mask local sleep")
_, y, e = series(["ctx_nrem_rand_1000_nb16", "ctx_nrem_rand_1000_nb64", "ctx_nrem_rand_1000_w512"])
ax.errorbar([16, 64, 256], y, e, marker="D", ms=3.5, lw=1.2, color=C["night"], label="offline night")
ax.set_xscale("log", base=2); ax.set_xlabel("replay batch size"); ax.set_ylabel("final accuracy (%)")
ax.set_ylim(78, 93); ax.legend(fontsize=6.5, loc="lower right")
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
