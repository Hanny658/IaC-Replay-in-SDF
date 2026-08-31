"""Run the cortical-constraint ablation ladder under the report's evaluation protocol.

    python src/run_bio.py --datasets nuh --variants all            # fit + checkpoint
    python src/run_bio.py --summary                                 # table against the baselines

Nothing here touches results/ or report/: every artefact goes to results/bio/.  The protocol is
evaluate() unchanged (inner OOF threshold, single test use, pooled CV: 5x5 on NUH/WDBC, 1x5 on
SUPPORT2), so the rows are directly comparable with the twelve models of the report.

Each variant is MLP-Cortex with one switch changed, so the table reads as a ladder:

    control   ReLU rates, every constraint off  (isolates the activation change from MLP-PC)
    full      every constraint on
    no_kp     full, but top-down error uses (W^{l+1})^T again   (weight transport allowed)
    no_bound  full, but the error channel is unbounded
    no_kwta   full, but no lateral inhibition (dense rates)
    no_dale   full, but neurons may have mixed-sign outputs
    no_homeo  full, but no homeostatic threshold adaptation
    no_kwta_homeo  both off: k-WTA and homeostasis are a pair (the win-rate target has no
              meaning without a competition), so this separates k-WTA's own value from its
              role of keeping homeostasis well-defined
"""
from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys
import time
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from data import LOADERS  # noqa: E402
from evaluate import evaluate  # noqa: E402
from models.cortex import CorticalPC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIO = os.path.join(ROOT, "results", "bio")
PARTS = os.path.join(BIO, "parts")
REPORT_PARTS = os.path.join(ROOT, "results", "parts")

ALL_OFF = dict(transport=True, bounded=False, kwta=False, dale=False, homeo=False)
V1 = dict(homeo="threshold")  # v1 rows keep the per-step threshold rule they were run with
VARIANTS = {
    "control": ALL_OFF,
    "full": V1,
    "no_kp": dict(V1, transport=True),
    "no_bound": dict(V1, bounded=False),
    "no_kwta": dict(V1, kwta=False),
    "no_dale": dict(V1, dale=False),
    "no_homeo": dict(homeo=False),
    # both off: tells whether k-WTA helps by itself or only by anchoring the homeostatic rule
    "no_kwta_homeo": dict(kwta=False, homeo=False),
    # ---- v2: synaptic-scaling homeostasis, sparsity sweep, Dale on the feedback synapses
    "v2_full": {},                        # scaling homeostasis, 25% active
    "v2_s10": dict(active_frac=0.10),
    "v2_s05": dict(active_frac=0.05),
    "v2_dalefb": dict(dale_fb=True),
    # ---- v3: the MNIST-proof no-transport setting (sign-concordant feedback, weak KP decay) as the
    # base; homeostasis moved offline into a sleep phase with a fixed set point
    "v3_base": dict(dale_fb=True, kp_decay=1e-3, kwta=False, homeo=False),
    "v3_kwta25": dict(dale_fb=True, kp_decay=1e-3, homeo=False),
    "v3_sleep25": dict(dale_fb=True, kp_decay=1e-3, homeo="sleep"),
    "v3_guard25": dict(dale_fb=True, kp_decay=1e-3, homeo="sleep_guard"),
    # ---- v3 base on SUPPORT2 came out 0.03 below v1's no_kwta_homeo (kp_decay 1e-2, no dale_fb);
    # these three separate the decay strength, the sign-concordant feedback, and the Adam epsilon
    "v3_base_d2": dict(dale_fb=True, kp_decay=1e-2, kwta=False, homeo=False),
    "v3_base_nofb": dict(dale_fb=False, kp_decay=1e-3, kwta=False, homeo=False),
    "v3_base_e3": dict(dale_fb=True, kp_decay=1e-3, kwta=False, homeo=False, adam_eps=1e-3),
    # ---- v6 on tabular: does weight mirroring remove the per-data-set KP decay?  On SUPPORT2 the
    # decay 1e-2 was worth 0.027 AUC over 1e-3; here the decay is 0 and either nothing or the
    # mirror phase keeps B aligned.  Same base as v3_base_d2 otherwise (kwta off, homeo off).
    "v6_nokp": dict(dale_fb=True, kp_decay=0.0, kwta=False, homeo=False, adam_eps=1e-3),
    "v6_mirror": dict(dale_fb=True, kp_decay=0.0, kwta=False, homeo=False, adam_eps=1e-3, mirror=5),
    "v6_d2_e3": dict(dale_fb=True, kp_decay=1e-2, kwta=False, homeo=False, adam_eps=1e-3),
}
REPEATS = {"nuh": 5, "wdbc": 5, "support2": 1}
BASELINES = ["LogReg", "RandomForest", "MLP-BP", "MLP-PC", "MLP-FF", "MLP-CMA-ES", "TabPFN"]
BASELINE_KEYS = {"LogReg": "logreg", "RandomForest": "rf", "MLP-BP": "mlp", "MLP-PC": "pc",
                 "MLP-FF": "ff", "MLP-CMA-ES": "cmaes", "TabPFN": "tabpfn"}


def part_path(dname: str, variant: str) -> str:
    return os.path.join(PARTS, f"{dname}_{variant}.pkl")


def run(dname: str, variant: str, seed: int) -> None:
    factory = functools.partial(CorticalPC, variant=variant, **VARIANTS[variant])
    ds = LOADERS[dname]()
    t0 = time.time()
    r = evaluate(factory, ds, n_repeats=REPEATS[dname], seed=seed)
    os.makedirs(PARTS, exist_ok=True)
    tmp = part_path(dname, variant) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"variant": variant, "result": r}, f)
    os.replace(tmp, part_path(dname, variant))
    print(f"   checkpointed {dname}/{variant} after {(time.time() - t0) / 60:.1f} min", flush=True)


def row_of(r, label: str) -> dict:
    return dict(model=label, test_auc=r.test_auc, test_eer=r.test_eer,
                oof_auc=r.train_oof_auc, oof_eer=r.train_oof_eer,
                cv_auc=r.cv_auc_mean, cv_auc_sd=r.cv_auc_std, cv_eer=r.cv_eer_mean, cv_eer_sd=r.cv_eer_std,
                fit_s=r.fit_time_s)


def summary() -> None:
    frames = []
    for dname in REPEATS:
        rows = []
        for name in BASELINES:
            p = os.path.join(REPORT_PARTS, f"{dname}_{BASELINE_KEYS[name]}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    rows.append(row_of(pickle.load(f)["result"], name))
        for variant in VARIANTS:
            p = part_path(dname, variant)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    rows.append(row_of(pickle.load(f)["result"], f"MLP-Cortex[{variant}]"))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.insert(0, "dataset", dname)
        frames.append(df)
        print(f"\n===== {dname}  (CV: {REPEATS[dname]}x5-fold pooled; selection view = OOF) =====")
        print(df.drop(columns="dataset").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if frames:
        out = pd.concat(frames)
        os.makedirs(BIO, exist_ok=True)
        out.to_csv(os.path.join(BIO, "summary.csv"), index=False)
        print(f"\nwrote {os.path.join(BIO, 'summary.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nuh"])
    ap.add_argument("--variants", nargs="+", default=["all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return
    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    for dname in args.datasets:
        for variant in variants:
            if args.resume and os.path.exists(part_path(dname, variant)):
                print(f"   skip {dname}/{variant} (checkpointed)", flush=True)
                continue
            run(dname, variant, args.seed)


if __name__ == "__main__":
    main()
