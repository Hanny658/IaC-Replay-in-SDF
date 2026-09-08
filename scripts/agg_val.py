"""Aggregate run_seq checkpoints: mean +- sd of final accuracy (percent), forgetting, n seeds, and a
seed-paired margin against a reference config with a bootstrap 95% interval.

    python scripts/agg_val.py [--prefix val_] [--ref g16_sgd_refr_s10_w512] config [config ...]
    python scripts/agg_val.py --prefix val_ --grep bp_derpp      # every config matching a substring
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "results", "bio", "seq", "parts")


def load(prefix, config):
    out = {}
    for f in glob.glob(os.path.join(PARTS, f"{prefix}{config}_s[0-9]*.pkl")):
        seed = int(re.search(r"_s(\d+)\.pkl$", f).group(1))
        if seed >= 100:  # timing-only seeds are not part of any table
            continue
        with open(f, "rb") as fh:
            out[seed] = pickle.load(fh)
    return out


def fmt(runs):
    if not runs:
        return "  (no runs)"
    acc = np.array([100 * runs[s]["final_acc"] for s in sorted(runs)])
    fg = np.array([100 * runs[s]["forgetting"] for s in sorted(runs)])
    sd = acc.std(ddof=1) if len(acc) > 1 else 0.0
    return (f"{acc.mean():5.1f} +- {sd:4.1f}  (n={len(acc)}, F={fg.mean():4.1f}, "
            f"seeds {[float(round(a, 1)) for a in acc]})")


def paired(a, b, n_boot=10000, seed=0):
    common = sorted(set(a) & set(b))
    if not common:
        return ""
    d = np.array([100 * (a[s]["final_acc"] - b[s]["final_acc"]) for s in common])
    rng = np.random.RandomState(seed)
    boots = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"  paired {d.mean():+5.1f} [{lo:+.1f}, {hi:+.1f}] over {len(d)} seeds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--prefix", default="val_")
    ap.add_argument("--ref")
    ap.add_argument("--grep")
    args = ap.parse_args()
    configs = list(args.configs)
    if args.grep:
        names = sorted({re.sub(r"_s\d+\.pkl$", "", os.path.basename(f))[len(args.prefix):]
                        for f in glob.glob(os.path.join(PARTS, f"{args.prefix}*{args.grep}*_s*.pkl"))})
        configs += [n for n in names if (args.prefix or not n.startswith("val"))]
    ref = load(args.prefix, args.ref) if args.ref else None
    for c in configs:
        runs = load(args.prefix, c)
        line = f"{c:42s} {fmt(runs)}"
        if ref and c != args.ref:
            line += paired(runs, ref)
        print(line)


if __name__ == "__main__":
    main()
