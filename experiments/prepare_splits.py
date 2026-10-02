# -*- coding: utf-8 -*-
"""Engine-disjoint preprocessing for the Idea-1 pilot.

Two things the original `data_process.py` does not do:

1. Engine-disjoint train/val/holdout split. The original selects checkpoints on the
   official test set, so every reported number is test-selected. Here the official
   test set is never touched; validation engines come out of the training file.
2. Condition-aware slope features. `_fea_extract1` fits one global slope per window.
   Under multi-condition operation the condition offsets ride on that slope. The
   `condfe` variant removes them with condition fixed effects.

Scalers and condition clusters are fit on TRAIN-split engines only.

`--norm` selects the sensor normalization (all fit on train engines only):
  oc_z           per-condition z-score (default; the original behaviour)
  global_z       one z-score per sensor, ignoring conditions
  global_minmax  one min-max per sensor, ignoring conditions (DAST paper)
Non-default norms write files as `{dataset}_{norm}_{variant}_*.mat`.
"""

import argparse
import json
import os

import numpy as np
import scipy.io as sio
from scipy.cluster.vq import kmeans as scipy_kmeans, vq as scipy_vq
from sklearn import preprocessing

SPLIT_SEED = 2026
COLS_TO_REMOVE = [2, 3, 4, 5, 9, 10, 14, 20, 22, 23]
SENSOR_COLS = slice(5, 26)
VARIANTS = ("global", "condfe", "none", "shuffle")
NORMS = ("oc_z", "oc_minmax", "global_z", "global_minmax")


def parse_args():
    p = argparse.ArgumentParser(description="Build engine-disjoint DAST datasets.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--dataset", default=None, help="Override cmapss.dataset.")
    p.add_argument("--out-dir", default="split_dataset")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--norm", default="oc_z", choices=NORMS)
    return p.parse_args()


def split_stem(dataset, norm, variant):
    """File stem shared with train_split.py; oc_z keeps the original names."""
    return f"{dataset}_{variant}" if norm == "oc_z" else f"{dataset}_{norm}_{variant}"


def scale_sensors(raw, train_mask, labels, n_centers, norm):
    """Normalize sensor columns of `raw`; statistics come from rows where train_mask is True."""
    scaled = raw.copy()
    if norm in ("oc_z", "oc_minmax"):
        groups = [labels == c for c in range(n_centers)]
    else:
        groups = [np.ones(len(raw), dtype=bool)]
    for g in groups:
        ref = raw[g & train_mask, SENSOR_COLS]
        if len(ref) == 0:
            continue
        if norm in ("global_minmax", "oc_minmax"):
            center = ref.min(axis=0)
            denom = ref.max(axis=0) - center
        else:
            center = ref.mean(axis=0)
            denom = ref.std(axis=0)
        denom[denom == 0] = 1.0
        scaled[g, SENSOR_COLS] = (raw[g, SENSOR_COLS] - center) / denom
    return scaled


def split_engines(engine_ids, seed=SPLIT_SEED):
    """70/15/15 engine-disjoint split. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(engine_ids, dtype=int)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.15 * n))
    return (
        np.sort(shuffled[:n_tr]),
        np.sort(shuffled[n_tr:n_tr + n_va]),
        np.sort(shuffled[n_tr + n_va:]),
    )


def fit_condition_clusters(rows, n_clusters):
    op = rows[:, 2:5].astype(float)
    scale = op.std(axis=0)
    scale[scale == 0] = 1.0
    centers, _ = scipy_kmeans(op / scale, n_clusters, iter=50, seed=42)
    return centers[np.argsort(centers[:, 0])], scale


def assign_clusters(rows, centers, scale):
    labels, _ = scipy_vq(rows[:, 2:5].astype(float) / scale, centers)
    return labels


def slope_global(window):
    """Original `_fea_extract1`: one OLS slope per sensor over the whole window."""
    t = np.arange(window.shape[0], dtype=np.float64)
    t = t - t.mean()
    denom = np.sum(t ** 2)
    if denom == 0:
        return np.zeros(window.shape[1])
    return (t @ window) / denom


def slope_condition_fe(window, labels):
    """Fixed-effects slope: center t and x within each condition present in the window.

    beta_s = sum_t (t - tbar_c) (x_ts - xbar_cs) / sum_t (t - tbar_c)^2

    Residualizing BOTH t and x is what makes this differ from merely subtracting
    condition means from the sensors.
    """
    t = np.arange(window.shape[0], dtype=np.float64)
    t_res = np.empty_like(t)
    x_res = np.empty_like(window, dtype=np.float64)
    for c in np.unique(labels):
        m = labels == c
        t_res[m] = t[m] - t[m].mean()
        x_res[m] = window[m] - window[m].mean(axis=0)
    denom = np.sum(t_res ** 2)
    if denom == 0:
        return np.zeros(window.shape[1])
    return (t_res @ x_res) / denom


def build_windows(rows, labels, window_size, rul_max):
    """All sliding windows for the given rows, with per-row condition labels kept."""
    X, Y, lab, eng = [], [], [], []
    for unit in np.unique(rows[:, 0]).astype(int):
        m = rows[:, 0] == unit
        data = rows[m]
        lb = labels[m]
        for j in range(len(data) - window_size + 1):
            X.append(data[j:j + window_size, 2:])
            lab.append(lb[j:j + window_size])
            Y.append(min(len(data) - window_size - j, rul_max))
            eng.append(unit)
    return (np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64),
            np.array(lab, dtype=int), np.array(eng, dtype=int))


def slope_matrix(X, labels, variant, rng):
    if variant == "global":
        return np.array([slope_global(w) for w in X])
    if variant == "condfe":
        return np.array([slope_condition_fe(w, l) for w, l in zip(X, labels)])
    if variant == "shuffle":
        # occupancy-preserving control: same label multiset, permuted in time
        return np.array([slope_condition_fe(w, rng.permutation(l)) for w, l in zip(X, labels)])
    raise ValueError(variant)


def main():
    args = parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))["cmapss"]
    dataset = args.dataset or cfg["dataset"]
    window_size = cfg["window_size"]
    rul_max = float(cfg["rul_max"])
    data_path = cfg["data_path"]
    n_clusters = int(cfg.get("condition_cluster_count", 6))
    if dataset in ("FD001", "FD003"):
        n_clusters = 1

    os.makedirs(args.out_dir, exist_ok=True)
    raw = np.loadtxt(f"{data_path}/train_{dataset}.txt")

    engines = np.unique(raw[:, 0]).astype(int)
    tr_e, va_e, ho_e = split_engines(engines)
    print(f"{dataset}: {len(engines)} engines -> train {len(tr_e)} / val {len(va_e)} / holdout {len(ho_e)}")
    print(f"  split seed {SPLIT_SEED}")

    tr_mask = np.isin(raw[:, 0], tr_e)
    tr_rows = raw[tr_mask]

    # clusters + scaler fit on TRAIN engines only
    centers, op_scale = fit_condition_clusters(tr_rows, n_clusters)
    all_labels = assign_clusters(raw, centers, op_scale)

    scaled = scale_sensors(raw, tr_mask, all_labels, len(centers), args.norm)
    print(f"  {args.norm} normalization fit on train engines only ({len(centers)} clusters)")

    reduced = np.delete(scaled, COLS_TO_REMOVE, axis=1)

    splits = {}
    for name, eids in (("train", tr_e), ("val", va_e), ("holdout", ho_e)):
        m = np.isin(reduced[:, 0], eids)
        splits[name] = build_windows(reduced[m], all_labels[m], window_size, rul_max)
        print(f"  {name:8} windows: {splits[name][0].shape}")

    rng = np.random.default_rng(SPLIT_SEED)
    mean_feat = {k: np.array([w.mean(axis=0) for w in v[0]]) for k, v in splits.items()}
    sc_mean = preprocessing.MinMaxScaler().fit(mean_feat["train"])

    for variant in args.variants:
        if variant == "none":
            slopes = None
        else:
            slopes = {k: slope_matrix(v[0], v[2], variant, rng) for k, v in splits.items()}
            sc_slope = preprocessing.MinMaxScaler().fit(slopes["train"])

        for name, (X, Y, _lab, eng) in splits.items():
            parts = [X]
            if slopes is not None:
                parts.append(sc_slope.transform(slopes[name])[:, None, :])
            parts.append(sc_mean.transform(mean_feat[name])[:, None, :])
            Xn = np.concatenate(parts, axis=1).astype(np.float32)
            stem = f"{args.out_dir}/{split_stem(dataset, args.norm, variant)}_{name}"
            sio.savemat(f"{stem}X.mat", {"X": Xn})
            sio.savemat(f"{stem}Y.mat", {"Y": (Y / rul_max).astype(np.float32)})
            sio.savemat(f"{stem}E.mat", {"E": eng})
        print(f"  variant {variant:8} -> T={Xn.shape[1]}  saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
