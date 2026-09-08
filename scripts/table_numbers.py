"""Print LaTeX-ready mean +- sd cells for the workshop tables from run_seq checkpoints.

    python scripts/table_numbers.py            # all rows, prefixes val_ / val4321_ / '' (development)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_val import load  # noqa: E402

import numpy as np

ROWS = [  # label, sequential config, static config, single-pass config
    ("full system", "g16_sgd_refr_s10_w512", "g16_sgd_refr_s10_w512_static", "g26_stream_refr"),
    ("- rotation (silent)", "g16_sgd_silent_s10_w512", "g16_sgd_silent_s10_w512_static", "g27_stream_silent"),
    ("- isolation (rot_noiso)", "g26_ctrl_rot_noiso", "g26_ctrl_rot_noiso_static", "g27_stream_rot_noiso"),
    ("- both (unmasked)", "g17_sgd_none_s10", "static_g17_sgd_none_s10", "g26_stream_none"),
    ("random rotation", "g26_ctrl_random_rot", "g26_ctrl_random_rot_static", "g27_stream_random_rot"),
    ("readout-only", "g26_ctrl_readout_only", "g26_ctrl_readout_only_static", "g27_stream_readout_only"),
    ("mirror", "g17_mirror_s10", "static_g17_mirror_s10", "g27_stream_mirror"),
    ("soft rotation", "g17_soft_s10", "static_g17_soft_s10", "g27_stream_soft"),
    ("no replay", "g17_ctx_none_s10", "g16_ctx_none_s10_w512_static", "g26_stream_ctx_none"),
    ("night same", "g17_night_s10", None, "g26_stream_night"),
    ("night narrow", "ctx_nrem_rand_1000", None, "g27_stream_night_narrow"),
    ("BP+ER", "bp_er_1000", "static_bp_none", "g26_stream_bp_er"),
    ("BP+DER++", "bp_derpp_a0.03_b1.0_w512_ce", None, "g27_stream_bp_derpp"),
    ("BP+ER-ACE", "bp_erace_1000", None, "g27_stream_bp_erace"),
    ("BP+A-GEM", "bp_agem_1000_w512", None, "g27_stream_bp_agem"),
    ("K200 ours / night / BP+ER / DER++", "g27_refr_K200", "ctx_nrem_rand_200", "bp_er_200"),
    ("K200 DER++", "bp_derpp_K200", None, None),
    ("K5000 ours / night / BP+ER", "g27_refr_K5000", "ctx_nrem_rand_5000", "bp_er_5000"),
    ("K5000 DER++", "bp_derpp_K5000", None, None),
    ("CIFAR ours", "cif_s10_sgd_refr", "cif_s10_sgd_refr_static", "cif_stream_sgd_refr"),
    ("CIFAR night", "cif_s10_night", None, "cif_stream_night"),
    ("CIFAR unmasked", "cif_s10_sgd_none", "cif_s10_sgd_none_static", "cif_stream_sgd_none"),
    ("CIFAR no replay", "cif_s10_ctx_none", None, "cif_stream_ctx_none"),
    ("CIFAR BP+ER", "cif_bp_er_1000", None, "cif_stream_bp_er"),
    ("CIFAR DER++", os.environ.get("CIF_DERPP", "cif_bp_derpp_a0.3_b1.0_w512_ce"), None, "cif_stream_bp_derpp"),
    ("CIFAR ER-ACE", os.environ.get("CIF_ERACE", "cif_bp_erace_1000"), None, "cif_stream_bp_erace"),
    ("CIFAR A-GEM", os.environ.get("CIF_AGEM", "cif_bp_agem_1000_w512"), None, "cif_stream_bp_agem"),
]


def cell(prefix, cfg):
    if cfg is None:
        return "---".ljust(20)
    runs = load(prefix, cfg)
    if not runs:
        return "(none)".ljust(20)
    acc = np.array([100 * runs[s]["final_acc"] for s in sorted(runs)])
    sd = acc.std(ddof=1) if len(acc) > 1 else 0.0
    return f"${acc.mean():.1f}\\pm{sd:.1f}$ n{len(acc)}".ljust(20)


prefixes = sys.argv[1:] or ["val_"]
for prefix in prefixes:
    print(f"===== prefix '{prefix}'  (seq | static | single pass)")
    for label, seq, sta, sp in ROWS:
        print(f"{label:36s} {cell(prefix, seq)} {cell(prefix, sta)} {cell(prefix, sp)}")
