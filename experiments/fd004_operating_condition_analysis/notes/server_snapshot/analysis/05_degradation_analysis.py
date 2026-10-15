"""
Phase 5 — Degradation Trend & RUL Correlation Analysis
Outputs:
  figures/05_degradation_trajectories.png
  figures/05_spearman_correlation.png
  figures/05_health_indicator.png
  figures/05_rul_aligned_trajectories.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import (load_data, add_rul, fit_oc_kmeans, assign_oc,
                   get_valid_sensors, compute_oc_stats, normalize_by_oc,
                   FIG_DIR, save_stats, N_CONDITIONS)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12,
                     'axes.labelsize': 10, 'figure.dpi': 120})

# ── Load & preprocess ──────────────────────────────────────────────────────────
train, _, _ = load_data()
train = add_rul(train)
km    = fit_oc_kmeans(train)
train = assign_oc(train, km)

valid_sensors, _ = get_valid_sensors(train)
oc_stats = compute_oc_stats(train, valid_sensors)
train_norm = normalize_by_oc(train, oc_stats, valid_sensors)
norm_cols = [f'{s}_norm' for s in valid_sensors]

# ── Spearman Correlation (normalised sensors vs RUL) ──────────────────────────
print("=" * 55)
print("  Spearman Correlation: Normalised Sensors vs RUL")
print("=" * 55)

spearman_results = {}
for col in norm_cols:
    rho, pval = sp_stats.spearmanr(train_norm[col], train_norm['RUL'])
    sensor = col.replace('_norm', '')
    spearman_results[sensor] = {'rho': round(rho, 4), 'pval': round(pval, 6)}
    print(f"  {sensor:4s}  rho = {rho:+.4f}  p = {pval:.2e}")

# Sort by |rho|
sorted_rho = sorted(spearman_results.items(),
                     key=lambda x: abs(x[1]['rho']), reverse=True)
top_sensors = [s for s, _ in sorted_rho[:6]]

print(f"\n  Top sensors by |Spearman rho|: {top_sensors}")
save_stats({
    "spearman_top6": top_sensors,
    "spearman_rho": {s: v['rho'] for s, v in spearman_results.items()},
})

# ── Figure 1 : Spearman Correlation Bar Chart ─────────────────────────────────
sensors_sorted = [s for s, _ in sorted_rho]
rho_vals       = [v['rho'] for _, v in sorted_rho]
colors = ['tomato' if r < 0 else 'steelblue' for r in rho_vals]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(sensors_sorted[::-1], rho_vals[::-1],
               color=colors[::-1], edgecolor='white', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Spearman ρ with RUL")
ax.set_title("Spearman Correlation: OC-Normalised Sensors vs RUL\n"
             "(blue = positive trend, red = negative trend with RUL)")
ax.grid(axis='x', linestyle=':', alpha=0.5)
for bar, val in zip(bars[::-1], rho_vals[::-1]):
    ax.text(val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right',
            fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_spearman_correlation.png"))
plt.close()
print("\nSaved: 05_spearman_correlation.png")

# ── Figure 2 : Degradation Trajectories (top-4 normalised sensors) ────────────
# Pick 5 random engines to overlay
np.random.seed(42)
sample_units = np.random.choice(train['unit'].unique(), size=5, replace=False)

top4 = top_sensors[:4]
CMAP = cm.plasma(np.linspace(0.1, 0.9, len(sample_units)))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, sensor in zip(axes.flat, top4):
    norm_col = f'{sensor}_norm'
    for uid, col in zip(sample_units, CMAP):
        eng = train_norm[train_norm['unit'] == uid].sort_values('cycle')
        ax.plot(eng['cycle'], eng[norm_col], lw=0.8, alpha=0.8,
                color=col, label=f'E#{uid}')
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"{sensor} (normalised)")
    ax.set_title(sensor)
    ax.legend(fontsize=7, loc='best')
    ax.grid(linestyle=':', alpha=0.4)

plt.suptitle("Degradation Trajectories — OC-Normalised Sensors\n(5 sample engines)",
             fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_degradation_trajectories.png"))
plt.close()
print("Saved: 05_degradation_trajectories.png")

# ── Figure 3 : RUL-aligned Trajectories (RUL on x-axis) ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, sensor in zip(axes.flat, top4):
    norm_col = f'{sensor}_norm'
    for uid, col in zip(sample_units, CMAP):
        eng = train_norm[train_norm['unit'] == uid].sort_values('RUL', ascending=False)
        ax.plot(eng['RUL'], eng[norm_col], lw=0.8, alpha=0.8,
                color=col, label=f'E#{uid}')
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel(f"{sensor} (normalised)")
    ax.set_title(sensor)
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc='best')
    ax.grid(linestyle=':', alpha=0.4)

plt.suptitle("RUL-Aligned Degradation Trajectories\n"
             "(x-axis = remaining life, decreasing →)",
             fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_rul_aligned_trajectories.png"))
plt.close()
print("Saved: 05_rul_aligned_trajectories.png")

# ── Health Indicator via PCA (first principal component) ──────────────────────
X_norm = train_norm[norm_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

pca = PCA(n_components=3, random_state=42)
pca.fit(X_scaled)
pc_var = pca.explained_variance_ratio_

print(f"\n  PCA explained variance: "
      f"PC1={pc_var[0]*100:.1f}%  PC2={pc_var[1]*100:.1f}%  PC3={pc_var[2]*100:.1f}%")
save_stats({
    "pca_pc1_var": round(float(pc_var[0]) * 100, 1),
    "pca_pc2_var": round(float(pc_var[1]) * 100, 1),
})

train_norm = train_norm.copy()
train_norm['HI'] = pca.transform(X_scaled)[:, 0]

# ── Figure 4 : Health Indicator (PC1) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: HI over cycle
for uid, col in zip(sample_units, CMAP):
    eng = train_norm[train_norm['unit'] == uid].sort_values('cycle')
    axes[0].plot(eng['cycle'], eng['HI'], lw=0.9, alpha=0.85,
                 color=col, label=f'E#{uid}')
axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Health Indicator (PC1)")
axes[0].set_title(f"Health Indicator vs Cycle\n(PC1, {pc_var[0]*100:.1f}% variance)")
axes[0].legend(fontsize=8)
axes[0].grid(linestyle=':', alpha=0.4)

# Right: HI vs RUL
sample2 = train_norm.sample(min(5000, len(train_norm)), random_state=2)
axes[1].scatter(sample2['HI'], sample2['RUL'], s=2, alpha=0.3, color='steelblue')
rho_hi, _ = sp_stats.spearmanr(sample2['HI'], sample2['RUL'])
axes[1].set_xlabel("Health Indicator (PC1)")
axes[1].set_ylabel("RUL (cycles)")
axes[1].set_title(f"Health Indicator vs RUL\n(Spearman ρ = {rho_hi:.3f})")
axes[1].grid(linestyle=':', alpha=0.4)

plt.suptitle("PCA-Based Health Indicator", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_health_indicator.png"))
plt.close()
print("Saved: 05_health_indicator.png")

save_stats({"hi_spearman_rho": round(float(rho_hi), 4)})
print("\nPhase 5 complete.")
