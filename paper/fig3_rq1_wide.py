# Full-width version of Fig. 3: the column figure's four panels split into two halves side by side,
# (a)(b) eta^2_w on the left and (c)(d) |rho| on the right. Same data and styling as make_paper_assets.py.
# Run from repo root: python paper/fig3_rq1_wide.py
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7})

R = {}
for r in csv.DictReader(open("paper/data/rq1_oc_variance.csv")):
    R.setdefault(r["dataset"], []).append(r)
sensors = [r["sensor"] for r in R["FD002"]]
x = np.arange(len(sensors))

fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.2), sharex=True)
panels = [(axes[0, 0], "FD002", "eta2_window", "(a) FD002: window-level OC-explained variance"),
          (axes[1, 0], "FD004", "eta2_window", "(b) FD004: window-level OC-explained variance"),
          (axes[0, 1], "FD002", "abs_rho_rul", "(c) FD002: |Spearman rho| with RUL"),
          (axes[1, 1], "FD004", "abs_rho_rul", "(d) FD004: |Spearman rho| with RUL")]
for ax, d, m, title in panels:
    raw = [float(r[f"{m}_raw_or_global"]) for r in R[d]]
    oc = [float(r[f"{m}_ocz"]) for r in R[d]]
    ax.bar(x - 0.2, raw, 0.4, color="#9aa5b1", label="Raw / global normalization")
    ax.bar(x + 0.2, oc, 0.4, color="#2f6db3", label="OC-aware z-score")
    ax.set_title(title, fontsize=7, loc="left")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=6)
for ax in axes[1]:
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, fontsize=6)
axes[1, 1].legend(fontsize=6, loc="upper right", framealpha=0.9)
fig.tight_layout(h_pad=0.6, w_pad=1.2)
fig.savefig("paper/figures/fig3_rq1_wide.png", dpi=300)
print("saved paper/figures/fig3_rq1_wide.png")
