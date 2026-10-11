# RQ1 robustness: distribution of window-level eta^2 and a train-only check.
#   all   : every engine in the official train file, OC clusters/statistics fit on all of them
#           (the setting behind Fig. 3 and rq1_oc_variance.csv)
#   train : only the 70% training engines of the engine-disjoint split (seed 2026),
#           clusters/statistics fit on those engines, as in model training
# Run from repo root: python paper/rq1_eta2_robustness.py
import csv
import os
import sys

import numpy as np
from scipy.cluster.vq import kmeans, vq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.prepare_splits import split_engines

KEEP = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
WINDOW, STRIDE = 60, 10


def eta2(x, lab):
    m = x.mean()
    tot = ((x - m) ** 2).sum()
    if tot == 0:
        return np.nan
    return sum((lab == c).sum() * (x[lab == c].mean() - m) ** 2 for c in np.unique(lab)) / tot


def per_oc(S, lab, kind):
    Z = S.copy()
    for c in np.unique(lab):
        m = lab == c
        if kind == "z":
            a, b = S[m].mean(0), S[m].std(0)
        else:
            a = S[m].min(0)
            b = S[m].max(0) - a
        b[b == 0] = 1
        Z[m] = (S[m] - a) / b
    return Z


def windows(lab, units):
    """Start indices of windows (stride 10); returns (all windows, windows with >= 2 OCs)."""
    allw, mixed = [], []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        for j in range(0, len(idx) - WINDOW + 1, STRIDE):
            w = idx[j:j + WINDOW]
            allw.append(w)
            if len(np.unique(lab[w])) >= 2:
                mixed.append(w)
    return allw, mixed


rows = []
for d in ["FD002", "FD004"]:
    tr = np.loadtxt(f"Cmapss_data/train_{d}.txt")
    tr_e, _, _ = split_engines(np.unique(tr[:, 0]).astype(int))
    for scope in ["all", "train"]:
        data = tr if scope == "all" else tr[np.isin(tr[:, 0], tr_e)]
        op = data[:, 2:5]
        sc = op.std(0)
        sc[sc == 0] = 1
        cen, _ = kmeans(op / sc, 6, iter=50, seed=42)
        lab, _ = vq(op / sc, cen)
        S = data[:, [4 + s for s in KEEP]]
        allw, mixed = windows(lab, data[:, 0])
        for norm, X in [("raw_or_global", S), ("oc_z", per_oc(S, lab, "z")),
                        ("oc_minmax", per_oc(S, lab, "minmax"))]:
            E = np.array([[eta2(X[w, k], lab[w]) for k in range(len(KEEP))] for w in mixed])
            for k, s in enumerate(KEEP):
                v = E[:, k][~np.isnan(E[:, k])]
                q = np.percentile(v, [10, 25, 50, 75, 90])
                rows.append({"dataset": d, "scope": scope, "norm": norm, "sensor": f"s{s}",
                             "n_engines": len(np.unique(data[:, 0])), "n_windows": len(allw),
                             "n_mixed_windows": len(mixed), "n_valid": len(v),
                             "mean": v.mean(), "p10": q[0], "p25": q[1], "median": q[2],
                             "p75": q[3], "p90": q[4]})

out = "paper/data/rq1_eta2_robustness.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

print("dataset scope norm | engines windows mixed | mean-of-sensor: mean median p10 p90 | "
      "sensor range of means")
for d in ["FD002", "FD004"]:
    for scope in ["all", "train"]:
        for norm in ["raw_or_global", "oc_z", "oc_minmax"]:
            g = [r for r in rows if (r["dataset"], r["scope"], r["norm"]) == (d, scope, norm)]
            m = lambda k: np.mean([r[k] for r in g])
            print(f"{d} {scope:5} {norm:13} | {g[0]['n_engines']} {g[0]['n_windows']} "
                  f"{g[0]['n_mixed_windows']} | {m('mean'):.3f} {m('median'):.3f} "
                  f"{m('p10'):.3f} {m('p90'):.3f} | "
                  f"{min(r['mean'] for r in g):.3f}-{max(r['mean'] for r in g):.3f}")
print("->", out)
