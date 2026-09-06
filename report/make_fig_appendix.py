"""Appendix figures for the workshop version: (a) substrate-decomposition waterfall on feature
CIFAR-10, (b) energy-accuracy Pareto of spiking inference (idealised arithmetic proxy)."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(os.path.dirname(HERE), "results", "bio", "seq", "parts")
FIGS = os.path.join(HERE, "figs")
PREFIX = "val_" if os.environ.get("MLPC_VAL") else ""  # held-out protocol pickles
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # outline (TrueType) fonts, no Type 3
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def stat(cfg, n=6):
    a = []
    for s in range(n):
        p = os.path.join(PARTS, f"{PREFIX}{cfg}_s{s}.pkl")
        if os.path.exists(p):
            a.append(100 * pickle.load(open(p, "rb"))["final_acc"])
    return np.mean(a), (np.std(a, ddof=1) if len(a) > 1 else 0.0), len(a)


# ---- (a) waterfall: BP+ER narrow -> wide -> +distance wiring -> +kWTA 10% ; local learner
steps = [("BP+ER\ndense\n$256$--$128$", "cfeat_bp_er_lr3e4", "#777777"),
         ("BP+ER\ndense\n$512$--$256$", "cfeat_bp_er_w512_lr3e4", "#777777"),
         ("+ $30\\%$\ndistance\nwiring", "cfeat_bp_er_w512_d30_lr1e3", "#777777"),
         ("+ $k$-WTA\n$10\\%$", "cfeat_bp_er_w512_d30k10_lr1e3", "#777777"),
         ("local\nlearner", "cfeat_sgd_refr_s10", "#1f6fb3")]
fig, ax = plt.subplots(figsize=(3.6, 2.5))
ms, ss = zip(*[stat(c)[:2] for _, c, _ in steps])
x = np.arange(len(steps))
for i, ((lab, cfg, col), m, s) in enumerate(zip(steps, ms, ss)):
    ax.bar(i, m, yerr=s, color=col, width=0.62, capsize=2, alpha=0.9 if i == 4 else 0.75)
    ax.text(i, m + s + 0.25, f"{m:.1f}", ha="center", fontsize=6.5)
for i in range(3):
    ax.annotate("", xy=(i + 1, ms[i + 1]), xytext=(i, ms[i]),
                arrowprops=dict(arrowstyle="->", color="#b3477a", lw=0.8, shrinkA=8, shrinkB=8))
    ax.text(i + 0.5, (ms[i] + ms[i + 1]) / 2 + 1.6, f"{ms[i+1]-ms[i]:+.1f}", ha="center", fontsize=6.5, color="#b3477a")
ax.set_xticks(x, [s[0] for s in steps], fontsize=6.0)
ax.set_ylim(36, 47); ax.set_ylabel("sequential accuracy (%)")
ax.set_title("feature CIFAR-10: the local substrate transplanted into BP+ER", fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_waterfall.pdf")); plt.close(fig)

# ---- (b) energy-accuracy Pareto (numbers from the spiking re-measure, train-calibrated thresholds)
series = {
    "always-on rotation, $10\\%$ (default)": ("#1f6fb3", [(57.5, 68.64, 4), (116.6, 92.73, 8), (236.0, 93.95, 16), (475.8, 94.02, 32)], (2461, 94.17), (526, 94.17)),
    "rotation-free (silent mask), $10\\%$": ("#b3477a", [(58.0, 56.94, 4), (119.6, 94.18, 8), (245.3, 94.30, 16), (498.0, 93.93, 32)], (2461, 95.22), (526, 95.22)),
}
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for lab, (col, pts, dense, evr) in series.items():
    xs, ys, ts = zip(*pts)
    ax.plot(xs, ys, marker="o", ms=3.5, lw=1.1, color=col, label=lab)
    if "default" in lab:
        for xx, yy, tt in pts:
            ax.annotate(f"$T_s{{=}}{tt}$", (xx, yy), textcoords="offset points", xytext=(3, -9), fontsize=6, color=col)
    ax.plot([dense[0]], [dense[1]], marker="s", ms=4, color=col, lw=0)
    ax.plot([evr[0]], [evr[1]], marker="D", ms=3.5, color=col, lw=0)
ax.plot([], [], marker="s", ms=4, color="black", lw=0, label="dense rate inference")
ax.plot([], [], marker="D", ms=3.5, color="black", lw=0, label="event-driven rate inference")
ax.set_xscale("log"); ax.set_xlabel("energy proxy per sample (nJ, 45-nm FP32 arithmetic)"); ax.set_ylabel("i.i.d. MNIST accuracy (%)")
ax.set_ylim(55, 97.5); ax.legend(loc="lower right", frameon=False)
ax.set_title("spiking inference: accuracy against energy proxy")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_pareto.pdf")); plt.close(fig)
print("appendix figures written")
