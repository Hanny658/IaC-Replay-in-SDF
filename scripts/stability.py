"""Continual-evaluation metrics from the g29_stab_* runs.

    python scripts/stability.py [--prefix val_]

For every configuration it reports, over seeds:
  online      accuracy of the prediction made on each incoming batch before learning from it
  anytime     mean accuracy on the classes seen so far, over the whole retention curve
  worst       the lowest point of that curve after the first task
  gap         stability gap: the drop from the accuracy just before a task switch to the
              lowest point reached on the previously seen tasks within the following 1000
              waking batches, averaged over the four switches
  churn       fraction of the waking batch whose prediction the replay update changes
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "results", "bio", "seq", "parts")
ROWS = [
    ("isolated replay + rotation", "g29_stab_ours"),
    ("rotation, unmasked replay", "g29_stab_rot_noiso"),
    ("isolated replay, no rotation", "g29_stab_silent"),
    ("unmasked, no rotation", "g29_stab_unmasked"),
    ("offline rehearsal", "g29_stab_night"),
    ("no replay", "g29_stab_noreplay"),
    ("BP + DER++ (same schedule)", "g29_stab_derpp"),
    ("BP + ER (same schedule)", "g29_stab_er"),
]


def metrics(d, window=1000):
    tr = d.get("eval_trace")
    if not tr:
        return None
    step = np.array([r[0] for r in tr], dtype=float)
    task = np.array([r[1] for r in tr], dtype=int)
    seen = np.array([r[2] for r in tr], dtype=float) * 100
    out = {"online": 100 * d["online_acc"] if d.get("online_acc") is not None else np.nan,
           "anytime": float(seen.mean()), "worst": float(seen[task > 0].min()) if (task > 0).any() else np.nan}
    gaps = []
    for t in range(1, task.max() + 1):
        before = np.where(task == t - 1)[0]
        after = np.where((task == t) & (step <= step[task == t][0] + window))[0]
        if not len(before) or not len(after):
            continue
        # accuracy on the tasks already learnt, just before the switch and at its worst after it
        prev = np.array([np.mean(r[3:3 + t]) for r in [tr[i] for i in after]]) * 100
        gaps.append(float(np.mean([tr[before[-1]][3 + k] for k in range(t)]) * 100 - prev.min()))
    out["gap"] = float(np.mean(gaps)) if gaps else np.nan
    drift = d.get("drift")
    out["churn"] = 100 * drift[1] if drift else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="val_")
    args = ap.parse_args()
    print(f"{'configuration':32s} {'online':>14s} {'anytime':>14s} {'worst':>14s} {'gap':>14s} {'churn':>14s}  n")
    for label, cfg in ROWS:
        vals = []
        for f in sorted(glob.glob(os.path.join(PARTS, f"{args.prefix}{cfg}_s[0-9].pkl"))):
            with open(f, "rb") as fh:
                m = metrics(pickle.load(fh))
            if m:
                vals.append(m)
        if not vals:
            print(f"{label:32s} (no runs)")
            continue
        cells = []
        for k in ("online", "anytime", "worst", "gap", "churn"):
            a = np.array([v[k] for v in vals], dtype=float)
            cells.append("   ---" .rjust(14) if np.all(np.isnan(a))
                         else f"{np.nanmean(a):8.2f}+-{(np.nanstd(a, ddof=1) if len(a) > 1 else 0.0):4.2f}")
        print(f"{label:32s} " + " ".join(cells) + f"  {len(vals)}")


if __name__ == "__main__":
    main()
