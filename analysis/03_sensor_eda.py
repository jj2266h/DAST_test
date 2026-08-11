"""
Phase 3 — Sensor EDA
Outputs:
  figures/03_sensor_std.png
  figures/03_sensor_time_series.png
  figures/03_sensor_correlation.png
  figures/03_sensor_vs_rul.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import (load_data, add_rul, fit_oc_kmeans, assign_oc,
                   get_valid_sensors, SENSOR_COLS, FIG_DIR, save_stats)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12,
                     'axes.labelsize': 10, 'figure.dpi': 120})

# ── Load ───────────────────────────────────────────────────────────────────────
train, test, _ = load_data()
train = add_rul(train)
km    = fit_oc_kmeans(train)
train = assign_oc(train, km)

# ── Sensor Validity ────────────────────────────────────────────────────────────
valid_sensors, invalid_sensors = get_valid_sensors(train, std_threshold=0.01)
stds = train[SENSOR_COLS].std().sort_values()

print("=" * 50)
print("  Sensor Standard Deviation Analysis")
print("=" * 50)
print(f"  Near-constant (excluded): {invalid_sensors}")
print(f"  Valid sensors ({len(valid_sensors)}): {valid_sensors}")
print()
for s in SENSOR_COLS:
    tag = "  [EXCLUDED]" if s in invalid_sensors else ""
    print(f"  {s:4s}  std = {train[s].std():.4f}{tag}")
print("=" * 50)

save_stats({
    "valid_sensors":   valid_sensors,
    "invalid_sensors": invalid_sensors,
    "n_valid_sensors": len(valid_sensors),
})

# ── Figure 1 : Sensor Std Bar Chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
colors = ['tomato' if s in invalid_sensors else 'steelblue' for s in stds.index]
ax.bar(stds.index, stds.values, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(0.01, color='k', linestyle='--', linewidth=1, label='Threshold (0.01)')
ax.set_xlabel("Sensor")
ax.set_ylabel("Standard Deviation")
ax.set_title("Sensor Standard Deviation — FD004 Training Set\n(red = near-constant, excluded)")
ax.legend(fontsize=9)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_sensor_std.png"))
plt.close()
print("Saved: 03_sensor_std.png")

# ── Figure 2 : Time Series (3 engines × top 6 valid sensors) ──────────────────
# Pick 3 representative engines: short / medium / long life
life = train.groupby('unit')['cycle'].max().sort_values()
short_eng  = life.index[0]
medium_eng = life.index[len(life) // 2]
long_eng   = life.index[-1]
sample_engines = [short_eng, medium_eng, long_eng]
labels = ['Short life', 'Medium life', 'Long life']

# Pick 6 sensors with highest std for display
top6 = train[valid_sensors].std().nlargest(6).index.tolist()

fig, axes = plt.subplots(6, 3, figsize=(15, 16), sharey='row')
for row, sensor in enumerate(top6):
    for col, (uid, label) in enumerate(zip(sample_engines, labels)):
        ax = axes[row, col]
        eng = train[train['unit'] == uid].sort_values('cycle')
        ax.plot(eng['cycle'], eng[sensor], lw=0.8, color='steelblue')
        if row == 0:
            ax.set_title(f"Engine #{uid}\n({label}, {len(eng)} cycles)",
                         fontsize=9)
        if col == 0:
            ax.set_ylabel(sensor, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(linestyle=':', alpha=0.4)
axes[-1, 1].set_xlabel("Cycle")
plt.suptitle("Sensor Time Series — 3 Representative Engines (top-6 by std)",
             fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_sensor_time_series.png"))
plt.close()
print("Saved: 03_sensor_time_series.png")

# ── Figure 3 : Correlation Heatmap (valid sensors) ────────────────────────────
import matplotlib.colors as mcolors
corr = train[valid_sensors].corr()
fig, ax = plt.subplots(figsize=(11, 9))
cmap = plt.get_cmap('coolwarm')
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(valid_sensors)))
ax.set_yticks(range(len(valid_sensors)))
ax.set_xticklabels(valid_sensors, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(valid_sensors, fontsize=8)
# Annotate cells
for i in range(len(valid_sensors)):
    for j in range(len(valid_sensors)):
        val = corr.values[i, j]
        color = 'white' if abs(val) > 0.7 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=6.5, color=color)
ax.set_title("Sensor Pearson Correlation Matrix (valid sensors only)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_sensor_correlation.png"))
plt.close()
print("Saved: 03_sensor_correlation.png")

# ── Figure 4 : Sensor vs RUL Scatter (top 6 valid sensors) ────────────────────
sample = train.sample(min(6000, len(train)), random_state=42)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, sensor in zip(axes.flat, top6):
    ax.scatter(sample[sensor], sample['RUL'], s=2, alpha=0.3, color='steelblue')
    # Trend line
    z = np.polyfit(sample[sensor].values, sample['RUL'].values, 1)
    p = np.poly1d(z)
    xs = np.linspace(sample[sensor].min(), sample[sensor].max(), 100)
    ax.plot(xs, p(xs), color='tomato', linewidth=1.5)
    corr_val = sample[[sensor, 'RUL']].corr().iloc[0, 1]
    ax.set_xlabel(sensor)
    ax.set_ylabel("RUL")
    ax.set_title(f"{sensor}  (r = {corr_val:.3f})")
    ax.grid(linestyle=':', alpha=0.4)
plt.suptitle("Sensor vs RUL — Scatter Plots (top-6 by std)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_sensor_vs_rul.png"))
plt.close()
print("Saved: 03_sensor_vs_rul.png")

print("\nPhase 3 complete.")
