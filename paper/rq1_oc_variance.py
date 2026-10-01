# RQ1: OC-explained sensor variance and sensor-RUL correlation, raw vs OC-aware z-score.
# Uses official train files only. Global min-max / global z-score are per-sensor affine maps,
# so their eta^2 and |rho| are identical to the raw values.
# Run from repo root: python paper/rq1_oc_variance.py
import csv

import matplotlib
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEEP = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # 14 sensors used by DAST
WINDOW, STRIDE, RUL_MAX = 40, 10, 125


def eta2(x, lab):
    m = x.mean()
    tot = ((x - m) ** 2).sum()
    btw = sum((lab == c).sum() * (x[lab == c].mean() - m) ** 2 for c in np.unique(lab))
    return btw / tot


def window_eta2(X, lab, units):
    vals = []
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        for j in range(0, len(idx) - WINDOW + 1, STRIDE):
            w = idx[j:j + WINDOW]
            if len(np.unique(lab[w])) < 2:
                continue
            vals.append([eta2(X[w, k], lab[w]) if X[w, k].var() > 0 else np.nan
                         for k in range(X.shape[1])])
    return np.nanmean(np.array(vals), axis=0)


results = {}
for d in ["FD002", "FD004"]:
    tr = np.loadtxt(f"Cmapss_data/train_{d}.txt")
    op = tr[:, 2:5]
    sc = op.std(0)
    sc[sc == 0] = 1
    cen, _ = kmeans(op / sc, 6, iter=50, seed=42)
    lab, _ = vq(op / sc, cen)
    life = np.array([tr[tr[:, 0] == u, 1].max() for u in tr[:, 0]])
    rul = np.minimum(life - tr[:, 1], RUL_MAX)

    S = tr[:, [4 + s for s in KEEP]]  # sensor s# sits in column 4+s
    Z = S.copy()
    for c in range(6):
        m = lab == c
        sd = S[m].std(0)
        sd[sd == 0] = 1
        Z[m] = (S[m] - S[m].mean(0)) / sd

    we_raw = window_eta2(S, lab, tr[:, 0])
    we_oc = window_eta2(Z, lab, tr[:, 0])
    R = np.array([[eta2(S[:, k], lab), eta2(Z[:, k], lab), we_raw[k], we_oc[k],
                   abs(spearmanr(S[:, k], rul)[0]), abs(spearmanr(Z[:, k], rul)[0])]
                  for k in range(len(KEEP))])
    results[d] = R
    print(f"{d}: OC counts {np.bincount(lab)}")
    print("  mean over sensors: pooled eta2 raw/OC-z, window eta2 raw/OC-z, |rho| raw/OC-z")
    print("  " + " | ".join(f"{v:.3f}" for v in R.mean(0)))

with open("paper/data/rq1_oc_variance.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "sensor", "eta2_pooled_raw_or_global", "eta2_pooled_ocz",
                "eta2_window_raw_or_global", "eta2_window_ocz",
                "abs_rho_rul_raw_or_global", "abs_rho_rul_ocz"])
    for d, R in results.items():
        for s, r in zip(KEEP, R):
            w.writerow([d, f"s{s}"] + [f"{v:.4f}" for v in r])

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
x = np.arange(len(KEEP))
for i, d in enumerate(["FD002", "FD004"]):
    R = results[d]
    ax = axes[0, i]
    ax.bar(x - 0.2, R[:, 2], 0.4, label="Raw / global normalization", color="#9aa5b1")
    ax.bar(x + 0.2, R[:, 3], 0.4, label="OC-aware z-score", color="#2f6db3")
    ax.set_title(f"{d}: within-window variance explained by OC (eta^2)", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax = axes[1, i]
    ax.bar(x - 0.2, R[:, 4], 0.4, color="#9aa5b1")
    ax.bar(x + 0.2, R[:, 5], 0.4, color="#2f6db3")
    ax.set_title(f"{d}: |Spearman rho| between sensor and RUL", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in KEEP], fontsize=8)
axes[0, 0].legend(fontsize=8, loc="center right", framealpha=0.9)
fig.tight_layout()
fig.savefig("paper/figures/fig3_rq1_oc_variance.png", dpi=300)
print("saved paper/figures/fig3_rq1_oc_variance.png and paper/data/rq1_oc_variance.csv")
