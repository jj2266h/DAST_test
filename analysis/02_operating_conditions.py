"""
Phase 2 — Operating Condition Analysis
Identifies 6 OC clusters via K-Means on (op1, op2, op3).
Outputs:
  figures/02_oc_scatter_2d.png
  figures/02_oc_3d_scatter.png
  figures/02_oc_frequency.png
  figures/02_oc_time_series.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

from utils import (load_data, add_rul, fit_oc_kmeans, assign_oc,
                   FIG_DIR, save_stats, N_CONDITIONS, OP_COLS)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'figure.dpi': 120})

COLORS = cm.tab10(np.linspace(0, 0.6, N_CONDITIONS))

# ── Load & cluster ─────────────────────────────────────────────────────────────
train, test, _ = load_data()
train = add_rul(train)

km    = fit_oc_kmeans(train)
train = assign_oc(train, km)
test  = assign_oc(test, km)

# ── OC centers & stats ────────────────────────────────────────────────────────
import pandas as pd
centers = pd.DataFrame(km.cluster_centers_, columns=OP_COLS)
centers.index.name = 'OC'
print("Operating Condition Cluster Centers:")
print(centers.round(4).to_string())
print()

freq_train = train.groupby('OC')['unit'].count()
pct_train  = (freq_train / len(train) * 100).round(1)
print("OC frequency in training set:")
for oc in sorted(train['OC'].unique()):
    print(f"  OC{oc}: {freq_train[oc]:6,} rows  ({pct_train[oc]:.1f}%)")
print()

oc_stats = {}
for oc in sorted(train['OC'].unique()):
    oc_stats[f"OC{oc}_center"] = {
        "op1": round(centers.loc[oc, 'op1'], 4),
        "op2": round(centers.loc[oc, 'op2'], 4),
        "op3": round(centers.loc[oc, 'op3'], 4),
        "pct_train": float(pct_train[oc])
    }
save_stats({"oc_centers": oc_stats})

# ── Figure 1 : 2D Scatter (op1 vs op2, op1 vs op3) ────────────────────────────
sample = train.sample(min(5000, len(train)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pairs = [('op1', 'op2'), ('op1', 'op3')]
for ax, (x, y) in zip(axes, pairs):
    for oc in sorted(sample['OC'].unique()):
        sub = sample[sample['OC'] == oc]
        ax.scatter(sub[x], sub[y], s=8, alpha=0.5, color=COLORS[oc],
                   label=f'OC{oc}')
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f'{x} vs {y}')
    ax.legend(markerscale=2, fontsize=9, loc='best')

plt.suptitle("FD004 — Operating Conditions (K-Means, k=6)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_oc_scatter_2d.png"))
plt.close()
print("Saved: 02_oc_scatter_2d.png")

# ── Figure 2 : 3D Scatter ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')
for oc in sorted(sample['OC'].unique()):
    sub = sample[sample['OC'] == oc]
    ax.scatter(sub['op1'], sub['op2'], sub['op3'],
               s=6, alpha=0.5, color=COLORS[oc], label=f'OC{oc}')
# Plot cluster centers
for oc in range(N_CONDITIONS):
    c = km.cluster_centers_[oc]
    ax.scatter(*c, s=150, marker='*', color=COLORS[oc],
               edgecolors='k', linewidths=0.5, zorder=5)
ax.set_xlabel('op1'); ax.set_ylabel('op2'); ax.set_zlabel('op3')
ax.set_title("3D Operating Conditions\n(★ = cluster center)")
ax.legend(markerscale=2, fontsize=9, loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_oc_3d_scatter.png"))
plt.close()
print("Saved: 02_oc_3d_scatter.png")

# ── Figure 3 : OC Frequency Bar Chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ocs  = sorted(train['OC'].unique())
vals = [freq_train[oc] for oc in ocs]
bars = ax.bar([f'OC{o}' for o in ocs], vals,
              color=[COLORS[o] for o in ocs], edgecolor='white', linewidth=0.6)
for bar, pct in zip(bars, [pct_train[o] for o in ocs]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 300, f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=10)
ax.set_ylabel("Number of Rows")
ax.set_title("FD004 Training Set — OC Frequency")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_oc_frequency.png"))
plt.close()
print("Saved: 02_oc_frequency.png")

# ── Figure 4 : OC Label Over Time (3 sample engines) ──────────────────────────
sample_units = [1, 50, 150]
fig, axes = plt.subplots(len(sample_units), 1, figsize=(12, 7), sharex=False)
for ax, uid in zip(axes, sample_units):
    eng = train[train['unit'] == uid].sort_values('cycle')
    ax.scatter(eng['cycle'], eng['OC'], s=8,
               c=[COLORS[oc] for oc in eng['OC']], zorder=3)
    ax.set_ylabel("OC label")
    ax.set_ylim(-0.5, N_CONDITIONS - 0.5)
    ax.set_yticks(range(N_CONDITIONS))
    ax.set_yticklabels([f'OC{o}' for o in range(N_CONDITIONS)], fontsize=8)
    ax.set_title(f"Engine #{uid}  (life = {eng['cycle'].max()} cycles)")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
axes[-1].set_xlabel("Cycle")
plt.suptitle("OC Label Sequence for Sample Engines", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_oc_time_series.png"))
plt.close()
print("Saved: 02_oc_time_series.png")

print("\nPhase 2 complete.")
