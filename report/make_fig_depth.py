"""Controller x depth figure (workshop version): accuracy vs hidden depth for fixed lambda,
the adaptive controller, and controller + skips, on both axes, plus the realised per-layer
lambda under the controller.  Reads the phase-20 checkpoints."""
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
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"fix": "#b3477a", "kad": "#1f6fb3", "skip": "#2a9d5c", "bp": "#777777"}


def load(cfg, n=6):
    out = []
    for s in range(n):
        p = os.path.join(PARTS, f"{PREFIX}{cfg}_s{s}.pkl")
        if os.path.exists(p):
            out.append(pickle.load(open(p, "rb")))
    return out


def stat(cfg):
    r = load(cfg)
    a = [100 * x["final_acc"] for x in r]
    return (np.mean(a), np.std(a, ddof=1) if len(a) > 1 else 0.0) if a else (np.nan, 0.0)


SERIES = {  # label -> (color, {depth: (seq cfg, static cfg)})
    "fixed $\\lambda$": (C["fix"], {2: ("g16_sgd_refr_s10_w512", "g16_sgd_refr_s10_w512_static"),
                                    3: ("g20_d3_fix", "g20_d3_fix_static"), 4: ("g20_d4_fix", "g20_d4_fix_static"),
                                    5: ("g20_d5_fix", "g20_d5_fix_static")}),
    "adaptive $\\lambda$": (C["kad"], {2: ("g19b_mnist", "g19b_mnist_static"), 3: ("g20_d3_kad", "g20_d3_kad_static"),
                                       4: ("g20_d4_kad", "g20_d4_kad_static"), 5: ("g20_d5_kad", "g20_d5_kad_static")}),
    "adaptive $\\lambda$ + skips": (C["skip"], {4: ("g20b_d4_skip", "g20b_d4_skip_static"),
                                                5: ("g20b_d5_skip", "g20b_d5_skip_static")}),
    "BP + ER": (C["bp"], {2: ("bp_er_1000", None), 3: ("g20_d3_bp_er", None), 4: ("g20_d4_bp_er", None),
                          5: ("g20_d5_bp_er", None)}),
}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for ax, axis_i, title in ((axes[0], 0, "sequential (split-MNIST)"), (axes[1], 1, "i.i.d. (static)")):
    for label, (col, cells) in SERIES.items():
        xs, ms, ss = [], [], []
        for d, cfgs in sorted(cells.items()):
            cfg = cfgs[axis_i]
            if cfg is None:
                continue
            m, s = stat(cfg)
            if not np.isnan(m):
                xs.append(d); ms.append(m); ss.append(s)
        if xs:
            ax.errorbar(xs, ms, yerr=ss, marker="o", ms=3.5, lw=1.2, capsize=2, color=col, label=label)
    ax.set_xlabel("hidden layers"); ax.set_xticks([2, 3, 4, 5]); ax.set_title(title)
    ax.axhline(10, color="black", lw=0.6, ls=":", alpha=0.6)
    ax.text(2.05, 11.5, "chance", fontsize=6, color="black", alpha=0.7)
axes[0].set_ylabel("final accuracy (%)")
axes[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.42), frameon=False)

# realised lambda per layer under the controller (sequential axis)
ax = axes[2]
for d, cfg, col in ((2, "g19b_mnist", "#9ecae1"), (3, "g20_d3_kad", "#6baed6"), (4, "g20_d4_kad", "#3182bd"), (5, "g20_d5_kad", "#08519c")):
    rs = load(cfg)
    kps = [[v for v in r["kp_eff"] if v is not None] for r in rs if r.get("kp_eff")]
    if kps:
        mean = np.mean(kps, axis=0)
        ax.semilogy(np.arange(2, 2 + len(mean)), mean, marker="s", ms=3, lw=1.1, color=col, label=f"{d} hidden")
ax.set_xlabel("layer $\\ell$ (decay applies from $\\ell=2$)"); ax.set_ylabel("realised $\\lambda_\\ell$")
ax.axhline(1e-3, color=C["fix"], lw=0.8, ls="--"); ax.text(2.1, 1.25e-3, "fixed $\\lambda=10^{-3}$", fontsize=6, color=C["fix"])
ax.set_ylim(1e-6, 3e-3); ax.set_title("controller's per-layer decay"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), frameon=False, ncol=2)
fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(FIGS, "fig_depth.pdf")); fig.savefig(os.path.join(FIGS, "fig_depth.png"), dpi=200)
print("fig_depth written")
