# -*- coding: utf-8 -*-
"""Post-hoc analyses of the frozen val-RMSE-selected checkpoints (no retraining).

For each best_valrmse_*.pth (clip 0 / warmup 0 arms by default) this reports:
  1. window length actually used (time steps - 2)            -> window_check
  2. eval-mode RMSE on train / val / holdout windows         -> generalization gap
  3. holdout RMSE and mean error by true-RUL range           -> rul_bins.csv
  4. holdout cut-point RMSE / NASA score over many cut seeds -> cut_sensitivity.csv
  5. learning curves from history_*.csv                      -> learning_curves.csv / .png
Holdout is only scored here; nothing is selected on it.

    python experiments/analyze_norm_ablation.py --results-dir experiments/norm_ablation_results
"""

import argparse
import collections
import csv
import glob
import json
import os
import statistics as st
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DAST_Network import DAST
from experiments.reeval_holdout import CKPT
from experiments.train_split import load_split, nasa_score

RUL_BINS = [(0, 25), (25, 50), (50, 100), (100, 125), (125, 126)]  # last bin = capped label
N_CUT_SEEDS = 50


def predict(model, X, device, batch=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.tensor(X[i:i + batch], dtype=torch.float32,
                                          device=device)).squeeze(-1).cpu().numpy().ravel())
    return np.concatenate(out)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cut_idx(E, seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(np.flatnonzero(E == u)) for u in np.unique(E)])


def mean_sd(xs):
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--results-dir", default="experiments/norm_ablation_results")
    p.add_argument("--history-dir", default=None, help="Default: --results-dir")
    p.add_argument("--split-dir", default="split_dataset")
    p.add_argument("--pattern", default="best_valrmse_*_clip0.0_warmup0_seed*.pth")
    p.add_argument("--out-dir", default=None, help="Default: <results-dir>/analysis")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    hist_dir = args.history_dir or args.results_dir
    out_dir = args.out_dir or os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    cfg = json.load(open(args.config, encoding="utf-8"))
    mdl, rul_max = cfg["model"], float(cfg["cmapss"]["rul_max"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cut_seeds = [2026] + list(range(N_CUT_SEEDS - 1))  # 2026 = the cut used in the paper

    runs, bins, cuts, curves = [], [], [], []
    splits = {}
    paths = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    print(f"{len(paths)} checkpoints")
    for path in paths:
        m = CKPT.search(os.path.basename(path))
        if not m:
            continue
        ds, norm, variant, clip, warmup, seed = m.groups()
        norm = norm or "oc_z"
        key = (ds, norm, variant)
        if key not in splits:
            splits[key] = {s: load_split(args.split_dir, ds, norm, variant, s)
                           for s in ("train", "val", "holdout")}
        S = splits[key]
        T = S["holdout"][0].shape[1]
        model = DAST(dim_val_s=mdl["dim_val"], dim_attn_s=mdl["dim_attn"],
                     dim_val_t=mdl["dim_val"], dim_attn_t=mdl["dim_attn"],
                     dim_val=mdl["dim_val"], dim_attn=mdl["dim_attn"],
                     time_step=T, input_size=S["holdout"][0].shape[2],
                     dec_seq_len=mdl["dec_seq_len"], out_seq_len=mdl["out_seq_len"],
                     n_encoder_layers=mdl["n_encoder_layers"],
                     n_decoder_layers=mdl["n_decoder_layers"],
                     n_heads=mdl["n_heads"], dropout=mdl["dropout"]).to(device)
        # fails loudly if the checkpoint was trained on a different window length
        model.load_state_dict(torch.load(path, map_location=device))

        res = {}
        for s in ("train", "val", "holdout"):
            X, Y, E = S[s]
            pred, true = predict(model, X, device) * rul_max, Y * rul_max
            res[s] = rmse(pred, true)
            if s == "holdout":
                ho_pred, ho_true, ho_E = pred, true, E
        tag = os.path.basename(path)[len("best_valrmse_"):-len(".pth")]
        row = {"tag": tag, "dataset": ds, "norm": norm, "seed": int(seed),
               "clip": float(clip), "warmup": int(warmup), "window": T - 2,
               "train_rmse": res["train"], "val_rmse": res["val"], "holdout_rmse": res["holdout"],
               "gap_holdout_minus_train": res["holdout"] - res["train"]}

        hist = os.path.join(hist_dir, f"history_{tag}.csv")
        if os.path.exists(hist):
            h = list(csv.DictReader(open(hist, encoding="utf-8")))
            vals = [float(r["val_rmse"]) for r in h]
            row["best_epoch"] = int(np.argmin(vals)) + 1
            row["epochs_to_within_5pct"] = next(i + 1 for i, v in enumerate(vals)
                                                if v <= 1.05 * min(vals))
            for r in h:
                curves.append({"dataset": ds, "norm": norm, "seed": int(seed),
                               "epoch": int(r["epoch"]),
                               # train_loss is RMSE on RUL/rul_max in train mode (dropout on)
                               "train_rmse_trainmode": float(r["train_loss"]) * rul_max,
                               "val_rmse": float(r["val_rmse"])})
        runs.append(row)

        for lo, hi in RUL_BINS:
            msk = (ho_true >= lo) & (ho_true < hi)
            if msk.any():
                err = ho_pred[msk] - ho_true[msk]
                bins.append({"dataset": ds, "norm": norm, "seed": int(seed),
                             "rul_bin": f"{lo}-{hi}" if hi <= 125 else "125 (capped)",
                             "n": int(msk.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))),
                             "mean_error": float(err.mean())})
        for cs in cut_seeds:
            idx = cut_idx(ho_E, cs)
            cuts.append({"dataset": ds, "norm": norm, "seed": int(seed), "cut_seed": cs,
                         "cut_rmse": rmse(ho_pred[idx], ho_true[idx]),
                         "cut_score_sum": nasa_score(ho_true[idx], ho_pred[idx])})
        print(f"  {tag}: T={T} window={T - 2} | train {res['train']:.2f} val {res['val']:.2f} "
              f"holdout {res['holdout']:.2f}")

    for name, data in (("per_run", runs), ("rul_bins", bins),
                       ("cut_sensitivity", cuts), ("learning_curves", curves)):
        if data:
            with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    arms = sorted({(r["dataset"], r["norm"]) for r in runs})
    by = lambda rows, ds, nm: [r for r in rows if r["dataset"] == ds and r["norm"] == nm]

    print("\n## 1. Window check (every arm should show one value)\n")
    for a in arms:
        print(f"- {a[0]} {a[1]}: window {sorted({r['window'] for r in by(runs, *a)})}")

    print("\n## 2. Fit vs generalization (eval-mode RMSE, mean ± SD over seeds)\n")
    print("| Dataset | Norm | n | Train | Val | Holdout | Holdout − Train | best epoch | epochs to ≤1.05×best |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        g = by(runs, *a)
        cols = [mean_sd([r[k] for r in g]) for k in
                ("train_rmse", "val_rmse", "holdout_rmse", "gap_holdout_minus_train")]
        ep = [mean_sd([r[k] for r in g]) if all(k in r for r in g) else "-"
              for k in ("best_epoch", "epochs_to_within_5pct")]
        print(f"| {a[0]} | {a[1]} | {len(g)} | " + " | ".join(cols + ep) + " |")

    print("\n## 3. Holdout error by true RUL (all windows; RMSE / mean error, mean over seeds)\n")
    labels = [f"{lo}-{hi}" if hi <= 125 else "125 (capped)" for lo, hi in RUL_BINS]
    print("| Dataset | Norm | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for a in arms:
        cells = []
        for lb in labels:
            g = [r for r in by(bins, *a) if r["rul_bin"] == lb]
            cells.append(f"{st.mean(r['rmse'] for r in g):.2f} / {st.mean(r['mean_error'] for r in g):+.2f}"
                         if g else "-")
        print(f"| {a[0]} | {a[1]} | " + " | ".join(cells) + " |")
    for ds in sorted({a[0] for a in arms}):
        g = [r for r in bins if r["dataset"] == ds and r["norm"] == arms[0][1] and r["seed"] == runs[0]["seed"]]
        print(f"  {ds} windows per bin: " + ", ".join(f"{r['rul_bin']}={r['n']}" for r in g))

    print(f"\n## 4. Cut-point sensitivity over {len(cut_seeds)} cut seeds\n")
    print("Arm cut RMSE = mean over training seeds for one cut seed; table shows its spread over cut seeds.\n")
    print("| Dataset | Norm | cut RMSE: seed 2026 | mean | 5–95% over cut seeds |")
    print("|---|---|---|---|---|")
    arm_cut = collections.defaultdict(dict)
    for a in arms:
        for cs in cut_seeds:
            arm_cut[a][cs] = st.mean(r["cut_rmse"] for r in by(cuts, *a) if r["cut_seed"] == cs)
        v = np.array(list(arm_cut[a].values()))
        print(f"| {a[0]} | {a[1]} | {arm_cut[a][2026]:.2f} | {v.mean():.2f} | "
              f"{np.percentile(v, 5):.2f}–{np.percentile(v, 95):.2f} |")
    print("\nPaired difference (A − B, cut RMSE averaged over training seeds) across cut seeds:\n")
    print("| Dataset | A − B | at 2026 | mean | 5–95% | share of cut seeds with A < B |")
    print("|---|---|---|---|---|---|")
    pairs = [("oc_minmax", "global_minmax"), ("oc_z", "global_z"), ("oc_z", "global_minmax"),
             ("oc_z", "oc_minmax"), ("global_z", "global_minmax")]
    for ds in sorted({a[0] for a in arms}):
        for A, B in pairs:
            if (ds, A) in arm_cut and (ds, B) in arm_cut:
                d = np.array([arm_cut[(ds, A)][cs] - arm_cut[(ds, B)][cs] for cs in cut_seeds])
                print(f"| {ds} | {A} − {B} | {d[0]:+.2f} | {d.mean():+.2f} | "
                      f"{np.percentile(d, 5):+.2f} to {np.percentile(d, 95):+.2f} | {(d < 0).mean():.0%} |")

    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dsets = sorted({a[0] for a in arms})
        fig, axes = plt.subplots(1, len(dsets), figsize=(3.4 * len(dsets), 2.8), squeeze=False)
        for ax, ds in zip(axes[0], dsets):
            for nm in [a[1] for a in arms if a[0] == ds]:
                g = by(curves, ds, nm)
                eps = sorted({r["epoch"] for r in g})
                v = np.array([[r["val_rmse"] for r in g if r["epoch"] == e] for e in eps])
                t = np.array([[r["train_rmse_trainmode"] for r in g if r["epoch"] == e] for e in eps])
                line, = ax.plot(eps, v.mean(1), lw=1, label=f"{nm} (val)")
                ax.fill_between(eps, v.min(1), v.max(1), color=line.get_color(), alpha=0.15, lw=0)
                ax.plot(eps, t.mean(1), lw=0.8, ls="--", color=line.get_color())
            ax.set_title(f"{ds}: val RMSE (solid), train RMSE in train mode (dashed)", fontsize=6)
            ax.set_xlabel("epoch", fontsize=7)
            ax.set_ylim(0, 40)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=300)
    print(f"\n-> {out_dir}/ (per_run, rul_bins, cut_sensitivity, learning_curves .csv/.png)")


if __name__ == "__main__":
    main()
