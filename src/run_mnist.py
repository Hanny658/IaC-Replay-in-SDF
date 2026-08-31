"""MLP-Cortex on MNIST: the same ladder, ten classes, an input with structure.

    python src/run_mnist.py --configs all --seeds 0 1 2 --epochs 30      # fit + checkpoint
    python src/run_mnist.py --summary                                     # table

Why a separate runner: the report protocol is binary ROC/EER on given splits; MNIST is ten-way and
comes with its own 60k/10k split, so the metric is test accuracy (plus macro one-vs-rest AUC from
the output node values), over several seeds instead of CV.  Everything lands in results/bio/mnist/.

Network: 784-64-32-10, the report's hidden sizes kept on purpose so that the cortical constraints
are measured on the same core.  Targets are one-hot and the PC energy is the same squared error as
in the binary case, so MLP-BP is trained on that same objective (MSE to one-hot, Adam, same
epochs) rather than on cross-entropy, to keep the comparison about the learning rule only.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from models.base import seed_everything  # noqa: E402
from models.cortex import CortexNet  # noqa: E402
from models.torch_utils import HIDDEN, to_t  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tmp", "dataset_cache", "mnist")
OUT = os.path.join(ROOT, "results", "bio", "mnist")
PARTS = os.path.join(OUT, "parts")

ALL_OFF = dict(transport=True, bounded=False, kwta=False, dale=False, homeo=False)
# On 784-d input the Kolen-Pollack decay that worked on the tabular sets (1e-2/step) crushes the
# weights (|W| ~ 0.01, 80% at 2 epochs), and without it the random feedback never aligns and
# learning diverges.  Sign-concordant feedback rescues it: with dale_fb the sign of every feedback
# synapse is fixed by cell type, which is local, and Liao et al. (2016) / Xiao et al. (2018) showed
# sign symmetry alone scales to ImageNet.  So the no-transport rows here use dale_fb + weak decay.
NT = dict(transport=False, dale=True, dale_fb=True, kp_decay=1e-3)
WIDE = (256, 128)  # v3: sparsity is a property of wide layers, so the sweep is run on these
CONFIGS = {
    "mlp_bp": dict(hidden=HIDDEN),                          # autograd reference, same objective
    "pc_tanh": dict(ALL_OFF, act="tanh"),                   # == MLP-PC of the report
    "control": ALL_OFF,                                     # ReLU rates, no constraints
    "bd_transport": dict(ALL_OFF, bounded=True, dale=True), # bounded + Dale, transport allowed
    "base3": dict(NT, bounded=True, kwta=False, homeo=False),   # + no transport (sign-concordant)
    "v2_full": dict(NT, bounded=True),                      # + k-WTA 25% + scaling homeostasis
    "v2_s10": dict(NT, bounded=True, active_frac=0.10),
    "kwta_only": dict(NT, bounded=True, homeo=False),      # splits v2_full's cost: k-WTA vs homeostasis
    # ---- v3: sleep-phase homeostasis on the narrow core; sparsity sweep on a wide core
    "v3_sleep25": dict(NT, bounded=True, homeo="sleep"),
    "mlp_bp_w": dict(hidden=WIDE),
    "control_w": dict(ALL_OFF, hidden=WIDE),
    "base3_w": dict(NT, bounded=True, kwta=False, homeo=False, hidden=WIDE),
    "kwta25_w": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.25),
    "kwta10_w": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.10),
    "kwta05_w": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.05),
    "kwta02_w": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.02),
    "sleep05_w": dict(NT, bounded=True, homeo="sleep", hidden=WIDE, active_frac=0.05),
    "sleep02_w": dict(NT, bounded=True, homeo="sleep", hidden=WIDE, active_frac=0.02),
    # ---- v3b: the equaliser-style sleep scaling collapsed 2/3 seeds; guard-rail form instead
    "guard25": dict(NT, bounded=True, homeo="sleep_guard"),
    "guard05_w": dict(NT, bounded=True, homeo="sleep_guard", hidden=WIDE, active_frac=0.05),
    "guard02_w": dict(NT, bounded=True, homeo="sleep_guard", hidden=WIDE, active_frac=0.02),
    # ---- v3c: the wide k-WTA runs above collapsed at epoch ~15 through an Adam runaway (tiny
    # second moments after thousands of small steps, then a discontinuous k-WTA gradient); with
    # adam_eps=1e-3 the same run reaches 97.2%.  The wide ladder is therefore re-run with it.
    "control_w_e3": dict(ALL_OFF, hidden=WIDE, adam_eps=1e-3),
    "base3_w_e3": dict(NT, bounded=True, kwta=False, homeo=False, hidden=WIDE, adam_eps=1e-3),
    "kwta25_w_e3": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.25, adam_eps=1e-3),
    "kwta10_w_e3": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.10, adam_eps=1e-3),
    "kwta05_w_e3": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.05, adam_eps=1e-3),
    "kwta02_w_e3": dict(NT, bounded=True, homeo=False, hidden=WIDE, active_frac=0.02, adam_eps=1e-3),
    "guard05_w_e3": dict(NT, bounded=True, homeo="sleep_guard", hidden=WIDE, active_frac=0.05, adam_eps=1e-3),
    "guard02_w_e3": dict(NT, bounded=True, homeo="sleep_guard", hidden=WIDE, active_frac=0.02, adam_eps=1e-3),
}

# ---- phase 4: on the v4 base (NT + bounded + Dale, wide core, adam_eps 1e-3)
V4 = dict(NT, bounded=True, homeo=False, hidden=WIDE, adam_eps=1e-3)
CONFIGS.update({
    # 4A spiking inference: the fitted rate network run as integrate-and-fire neurons
    "s4_dense": dict(ALL_OFF, hidden=WIDE, adam_eps=1e-3, spike=True),
    "s4_kwta25": dict(V4, active_frac=0.25, spike=True),
    "s4_kwta10": dict(V4, active_frac=0.10, spike=True),
    "s4_kwta05": dict(V4, active_frac=0.05, spike=True),
    # 4B relaxation depth: is a 20-step settling needed, or does one local pass do?
    # gamma is scaled so that gamma*T stays 2 (the T=20, gamma=0.1 product used everywhere else)
    "r4_T1": dict(V4, active_frac=0.10, T=1, gamma=1.0),
    "r4_T3": dict(V4, active_frac=0.10, T=3, gamma=0.67),
    "r4_T5": dict(V4, active_frac=0.10, T=5, gamma=0.4),
    "r4_T10": dict(V4, active_frac=0.10, T=10, gamma=0.2),
    # 4C self-organised connectivity: distance-dependent sparse synapses vs random at equal density
    "c4_d30_dist": dict(V4, active_frac=0.10, conn_density=0.30, conn_mode="dist"),
    "c4_d30_rand": dict(V4, active_frac=0.10, conn_density=0.30, conn_mode="random"),
    "c4_d10_dist": dict(V4, active_frac=0.10, conn_density=0.10, conn_mode="dist"),
    "c4_d10_rand": dict(V4, active_frac=0.10, conn_density=0.10, conn_mode="random"),
    "c4_d10_dist_regrow": dict(V4, active_frac=0.10, conn_density=0.10, conn_mode="dist", regrow=0.05),
})

# ---- phase 5: the v5 base = v4 + T=5 relaxation + 30% distance-dependent connectivity
V5 = dict(V4, active_frac=0.10, T=5, gamma=0.4, conn_density=0.30, conn_mode="dist")
CONFIGS.update({
    "v5_base": dict(V5, spike=True),
    "v5_regrow": dict(V5, conn_density=0.10, regrow=0.05, spike=True),
    "mlp_bp_w5": dict(hidden=WIDE),                       # BP reference re-run for the robustness metric
    # 5A: no per-synapse optimiser state -- heavy-ball SGD instead of Adam
    "v5_sgd01": dict(V5, opt="sgd", eta_override=0.01),
    "v5_sgd02": dict(V5, opt="sgd", eta_override=0.02),
    # 5B: burst-multiplexing ingredients
    "v5_bgate": dict(V5, burst_gate=True),
    "v5_bbase": dict(V5, burst_baseline=True),
    "v5_burst": dict(V5, burst_gate=True, burst_baseline=True),
    # 5C: sleep as the negative phase (dreams per night x anti-Hebbian rate)
    "v5_dream": dict(V5, dream_batches=20, dream_eta=0.1),
    "v5_dream_strong": dict(V5, dream_batches=20, dream_eta=0.3),
})

# ---- phase 6: the v6 base = v5 + the two free burst ingredients
V6 = dict(V5, burst_gate=True, burst_baseline=True)
CONFIGS.update({
    "v6_base": dict(V6, spike=True),
    # 6A single-phase burst propagation (no settling) and multiplicative burst coding
    "v6_sweep": dict(V6, sweep=True, spike=True),
    "v6_mult": dict(V6, burst_mult=True),
    "v6_sweep_mult": dict(V6, sweep=True, burst_mult=True),
    # 6B spike-count noise during training; spiking inference then measured on the same net
    "v6_spiketrain8": dict(V6, spike_train=8, spike=True),
    "v6_spiketrain16": dict(V6, spike_train=16, spike=True),
    # 6C weight mirroring instead of Kolen-Pollack decay (kp_decay 0 and 1e-3)
    "v6_mirror": dict(V6, mirror=20, kp_decay=0.0),
    "v6_mirror_d3": dict(V6, mirror=20, kp_decay=1e-3),
    "v6_nokp": dict(V6, kp_decay=0.0),                      # control: no decay, no mirror
    # 6D SGD with a larger step and cosine decay
    "v6_sgd03_cos": dict(V6, opt="sgd", eta_override=0.03, cosine=True),
    "v6_sgd05_cos": dict(V6, opt="sgd", eta_override=0.05, cosine=True),
})


# ------------------------------------------------------------------ data
def _idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        dims = [int.from_bytes(f.read(4), "big") for _ in range(magic & 0xFF)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist():
    Xtr = _idx(os.path.join(DATA, "train-images-idx3-ubyte.gz")).reshape(-1, 784).astype(np.float32) / 255
    ytr = _idx(os.path.join(DATA, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    Xte = _idx(os.path.join(DATA, "t10k-images-idx3-ubyte.gz")).reshape(-1, 784).astype(np.float32) / 255
    yte = _idx(os.path.join(DATA, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    # per-pixel standardisation fit on the training set (constant pixels keep scale 1)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    global _MU, _SD, _XTE_RAW
    _MU, _SD, _XTE_RAW = mu, sd, Xte  # kept for the spiking evaluation, which needs raw intensities
    return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte


# ------------------------------------------------------------------ models
def fit_cortex(Xtr, ytr, seed, epochs, batch, cfg, T=20, gamma=0.1, eta=1e-3, weight_decay=1e-3, log=None):
    cfg = dict(cfg)
    hidden = cfg.pop("hidden", HIDDEN)
    T, gamma = cfg.pop("T", T), cfg.pop("gamma", gamma)   # 4B: relaxation depth ladder
    eta = cfg.pop("eta_override", eta)                     # 5A: SGD needs its own step size
    cosine = cfg.pop("cosine", False)                      # 6D: cosine decay of eta to 10%
    cfg.pop("spike", None)                                 # 4A flag, handled by run()
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)                                # 6B: torch.poisson uses the global RNG
    Xt = to_t(Xtr)
    Yt = torch.nn.functional.one_hot(torch.as_tensor(ytr), 10).float()
    net = CortexNet([784, *hidden, 10], seed=seed, input_shape=(28, 28), **cfg)
    n, hist = len(Xt), []
    with torch.no_grad():
        for ep in range(epochs):
            net.training = True
            eta_ep = eta * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * ep / epochs))) if cosine else eta
            t0, perm, tot = time.time(), torch.randperm(n, generator=g), 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                x, a, eps = net.relax(Xt[idx], Yt[idx], T, gamma)
                net.local_update(x, a, eps, eta_ep, weight_decay)
                tot += (eps[net.L] ** 2).sum().item()
            hist.append(tot / n)
            net.training = False
            if net.homeo in ("sleep", "sleep_guard"):
                net.sleep(Xt[perm])
            net.structural_plasticity(g)
            for _ in range(net.dream_batches):  # 5C: the negative phase happens at night
                net.dream(batch, g, T, gamma, eta_ep, weight_decay)
            for _ in range(net.mirror):         # 6C: the mirror phase happens at night too
                net.mirror_phase(batch, g)
            if log:
                log(ep, hist[-1], time.time() - t0)
    net.training = False
    return net, hist


def noise_robustness(predict, Xte, yte, sigmas=(0.5, 1.0), seed=0):
    """Accuracy under additive Gaussian noise on the standardised input (same noise for every model)."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in sigmas:
        Xn = Xte + rng.normal(0, s, Xte.shape).astype(np.float32)
        out[s] = float((predict(Xn).argmax(1) == yte).mean())
    return out


def predict_cortex(net, X):
    net.training = False
    with torch.no_grad():
        x, _ = net.forward(to_t(X))
    return x[-1].numpy()


def fit_bp(Xtr, ytr, seed, epochs, batch, hidden=HIDDEN, eta=1e-3, weight_decay=1e-3, log=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(784, hidden[0]), torch.nn.ReLU(),
                              torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(),
                              torch.nn.Linear(hidden[1], 10))
    opt = torch.optim.Adam(net.parameters(), lr=eta, weight_decay=weight_decay)
    Xt = to_t(Xtr)
    Yt = torch.nn.functional.one_hot(torch.as_tensor(ytr), 10).float()
    n, hist = len(Xt), []
    for ep in range(epochs):
        t0, perm, tot = time.time(), torch.randperm(n, generator=g), 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).sum(1).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot / n)
        if log:
            log(ep, hist[-1], time.time() - t0)
    return net, hist


def predict_bp(net, X):
    with torch.no_grad():
        return net(to_t(X)).numpy()


# ------------------------------------------------------------------ metrics
def macro_auc(y, S):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score((y == c).astype(int), S[:, c]) for c in range(10)]))


def run(config: str, seed: int, epochs: int, batch: int) -> dict:
    Xtr, ytr, Xte, yte = load_mnist()
    seed_everything(seed)
    t0 = time.time()

    def log(ep, loss, dt):
        print(f"    {config:10s} seed {seed} epoch {ep + 1:3d}/{epochs}  loss {loss:.4f}  {dt:5.1f}s", flush=True)

    if config.startswith("mlp_bp"):
        h = CONFIGS[config]["hidden"]
        net, hist = fit_bp(Xtr, ytr, seed, epochs, batch, hidden=h, log=log)
        S = predict_bp(net, Xte)
        cost = {"dense_macs": 784 * h[0] + h[0] * h[1] + h[1] * 10, "synops": None, "active": None}
        kp = None
    else:
        net, hist = fit_cortex(Xtr, ytr, seed, epochs, batch, CONFIGS[config], log=log)
        S = predict_cortex(net, Xte)
        cost = net.last_cost
        kp = None if net.transport else [
            float(torch.nn.functional.cosine_similarity(net.B[l].flatten(), net.W[l + 1].flatten(), dim=0))
            for l in range(1, net.L)]
    acc = float((S.argmax(1) == yte).mean())
    out = dict(config=config, seed=seed, epochs=epochs, test_acc=acc, test_macro_auc=macro_auc(yte, S),
               final_loss=hist[-1], fit_s=time.time() - t0, dense_macs=cost["dense_macs"],
               synops=cost["synops"], active=cost["active"], kp_cos=kp, hist=hist)
    out["noise_acc"] = noise_robustness(
        (lambda X: predict_bp(net, X)) if config.startswith("mlp_bp") else (lambda X: predict_cortex(net, X)),
        Xte, yte, seed=seed)
    print(f"    noise robustness: {[f'sigma {s}: {v:.4f}' for s, v in out['noise_acc'].items()]}", flush=True)
    if not config.startswith("mlp_bp"):
        out["conn_density"] = [None if m is None else float(m.mean()) for m in net.mask[1:net.L]]
        if CONFIGS[config].get("spike"):  # 4A: the same fitted network as integrate-and-fire neurons
            from models.spiking import calibrate_thresholds, spike_eval
            thr = calibrate_thresholds(net, to_t(Xtr[:5000]))
            out["spike"] = {}
            for T_s in (4, 8, 16, 32, 64):
                s_acc, s_counts, s_events = spike_eval(net, _XTE_RAW, _MU, _SD, yte, T_s, thr, seed=seed)
                out["spike"][T_s] = dict(acc=s_acc, spikes=s_counts, events=s_events)
                print(f"    spiking T_s={T_s:3d}: acc {s_acc:.4f}  hidden spikes/sample {[round(c, 1) for c in s_counts]}  "
                      f"synaptic events/sample {s_events:.0f}  (rate MACs {cost['dense_macs']})", flush=True)
    print(f"  {config:10s} seed {seed}: test acc {acc:.4f}  macro-AUC {out['test_macro_auc']:.4f}  "
          f"synops {cost['synops']}  {out['fit_s'] / 60:.1f} min", flush=True)
    return out


def part_path(config, seed):
    return os.path.join(PARTS, f"{config}_s{seed}.pkl")


def summary():
    rows = []
    for f in sorted(os.listdir(PARTS)) if os.path.isdir(PARTS) else []:
        with open(os.path.join(PARTS, f), "rb") as fh:
            rows.append(pickle.load(fh))
    if not rows:
        print("no MNIST results yet")
        return
    df = pd.DataFrame(rows)
    df["synops"] = df["synops"].astype(float)
    agg = df.groupby("config").agg(seeds=("seed", "count"), acc=("test_acc", "mean"), acc_sd=("test_acc", "std"),
                                   auc=("test_macro_auc", "mean"), synops=("synops", "mean"),
                                   dense=("dense_macs", "first"), fit_min=("fit_s", lambda s: s.mean() / 60))
    agg = agg.reindex([c for c in CONFIGS if c in agg.index])
    agg["synops_frac"] = agg["synops"] / agg["dense"]
    print(agg.to_string(float_format=lambda v: f"{v:.4f}"))
    df.drop(columns="hist").to_csv(os.path.join(OUT, "runs.csv"), index=False)
    agg.to_csv(os.path.join(OUT, "summary.csv"))
    print(f"\nwrote {OUT}/summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return
    os.makedirs(PARTS, exist_ok=True)
    configs = list(CONFIGS) if args.configs == ["all"] else args.configs
    for seed in args.seeds:
        for config in configs:
            if args.resume and os.path.exists(part_path(config, seed)):
                print(f"  skip {config} seed {seed} (checkpointed)", flush=True)
                continue
            out = run(config, seed, args.epochs, args.batch)
            tmp = part_path(config, seed) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(out, f)
            os.replace(tmp, part_path(config, seed))


if __name__ == "__main__":
    main()
