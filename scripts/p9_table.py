"""Phase-9 table: accuracy, replay volume, isolated channel, code overlap and dimensionality per config."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "results", "bio", "seq", "parts")
prefixes = tuple(sys.argv[1:]) or ("g8_", "g9", "static_loc16", "static_g9", "static_ctx_none", "ctx_nrem_rand_1000", "loc16_none")
rows = []
for f in sorted(os.listdir(PARTS)):
    if not f.startswith(prefixes):
        continue
    r = pickle.load(open(os.path.join(PARTS, f), "rb"))
    ov = r.get("overlap") or [np.nan, np.nan]
    ef = r.get("eff_frac") or [np.nan, np.nan]
    dim = r.get("dim") or [np.nan, np.nan, np.nan]
    rows.append(dict(config=r["config"], seed=r["seed"], acc=r["final_acc"], replay=r.get("replay_used", np.nan),
                     ch1=ef[0], ch2=ef[1], ov_in=r.get("overlap_in", np.nan), ov1=ov[0], ov2=ov[1],
                     dim0=dim[0], dim1=dim[1], dim2=dim[2], synops=r.get("synops", np.nan), gate=r.get("gate_signal", np.nan),
                     min=r["fit_s"] / 60))
df = pd.DataFrame(rows)
agg = df.groupby("config").agg(n=("seed", "count"), acc=("acc", "mean"), sd=("acc", "std"), replay=("replay", "mean"),
                               ch1=("ch1", "mean"), ch2=("ch2", "mean"), ov_in=("ov_in", "mean"), ov1=("ov1", "mean"), ov2=("ov2", "mean"),
                               dim0=("dim0", "mean"), dim1=("dim1", "mean"), dim2=("dim2", "mean"), synops=("synops", "mean"),
                               gate=("gate", "mean"), min=("min", "mean"))
pd.set_option("display.width", 250)
print(agg.to_string(float_format=lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"))
