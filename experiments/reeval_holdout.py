# -*- coding: utf-8 -*-
"""Rebuild summary.jsonl by re-evaluating saved best_valrmse checkpoints on the holdout split.

Use when summary.jsonl is lost but the .pth files remain. Only the val-RMSE-selected
checkpoint is evaluated (the one the paper reports). Refuses to overwrite an existing file.

Besides the all-window metrics it adds `holdout_cut_*`: one window per holdout engine,
cut at a random point as in the official test set (true RUL 0 at the last window would be
trivial). Cut points use a fixed seed, so arms with the same window size share them.
`holdout_cut_score_sum` is the NASA score summed over those engines (DAST_test.s_score).

    python experiments/reeval_holdout.py --results-dir experiments/norm_ablation_results \
        --out experiments/norm_ablation_results/summary_cut.jsonl
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DAST_Network import DAST
from experiments.train_split import evaluate, load_split, nasa_score

# the norm part is absent in clean-sweep names (those runs are oc_z)
CKPT = re.compile(r"best_valrmse_(FD\d{3})(?:_(.+?))?_(global|condfe|none|shuffle)"
                  r"_clip([\d.]+)_warmup(\d+)_seed(\d+)\.pth$")
CUT_SEED = 2026


def cut_index(E, seed=CUT_SEED):
    """One window index per engine, drawn uniformly; windows of an engine are in time order."""
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero(E == u)) for u in np.unique(E)])


def cut_metrics(model, X, Y, E, rul_max, device):
    idx = cut_index(E)
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X[idx], dtype=torch.float32, device=device)
        pred = model(xb).squeeze(-1).cpu().numpy().ravel() * rul_max
    true = Y[idx] * rul_max
    return {"holdout_cut_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
            "holdout_cut_score_sum": nasa_score(true, pred),
            "holdout_cut_n": int(len(idx))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--results-dir", default="experiments/norm_ablation_results")
    p.add_argument("--split-dir", default="split_dataset")
    p.add_argument("--out", default=None, help="Default: <results-dir>/summary.jsonl")
    args = p.parse_args()

    out = args.out or os.path.join(args.results_dir, "summary.jsonl")
    if os.path.exists(out):
        sys.exit(f"{out} already exists; pass --out to write elsewhere.")

    cfg = json.load(open(args.config, encoding="utf-8"))
    mdl = cfg["model"]
    rul_max = float(cfg["cmapss"]["rul_max"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(glob.glob(os.path.join(args.results_dir, "best_valrmse_*.pth")))
    print(f"{len(paths)} checkpoints in {args.results_dir}")
    with open(out, "w", encoding="utf-8") as f:
        for path in paths:
            m = CKPT.search(os.path.basename(path))
            if not m:
                print(f"  skip (unrecognized name): {path}")
                continue
            ds, norm, variant, clip, warmup, seed = m.groups()
            norm = norm or "oc_z"
            X, Y, E = load_split(args.split_dir, ds, norm, variant, "holdout")
            model = DAST(
                dim_val_s=mdl["dim_val"], dim_attn_s=mdl["dim_attn"],
                dim_val_t=mdl["dim_val"], dim_attn_t=mdl["dim_attn"],
                dim_val=mdl["dim_val"], dim_attn=mdl["dim_attn"],
                time_step=X.shape[1], input_size=X.shape[2],
                dec_seq_len=mdl["dec_seq_len"], out_seq_len=mdl["out_seq_len"],
                n_encoder_layers=mdl["n_encoder_layers"],
                n_decoder_layers=mdl["n_decoder_layers"],
                n_heads=mdl["n_heads"], dropout=mdl["dropout"],
            ).to(device)
            # fails loudly if the split's window length differs from the one trained on
            model.load_state_dict(torch.load(path, map_location=device))
            hm = evaluate(model, X, Y, E, rul_max, device)
            tag = os.path.basename(path)[len("best_valrmse_"):-len(".pth")]
            row = {"tag": tag, "dataset": ds, "norm": norm, "variant": variant,
                   "seed": int(seed), "clip": float(clip), "warmup": int(warmup),
                   "window": X.shape[1] - 2, "reevaluated": True}
            row.update({f"holdout_selby_valrmse_{k}": v for k, v in hm.items()})
            row.update(cut_metrics(model, X, Y, E, rul_max, device))
            f.write(json.dumps(row) + "\n")
            print(f"  {tag}: RMSE {hm['rmse']:.3f} | cut RMSE {row['holdout_cut_rmse']:.3f} "
                  f"| cut Score {row['holdout_cut_score_sum']:.1f} (n={row['holdout_cut_n']})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
