"""
Phase 4 — Operating Condition × Sensor Interaction
Shows how OC masks sensor readings and why OC normalisation is needed.
Outputs:
  figures/04_sensor_boxplot_by_oc.png
  figures/04_oc_offset_heatmap.png
  figures/04_before_after_norm.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import (load_data, add_rul, fit_oc_kmeans, assign_oc,
                   get_valid_sensors, compute_oc_stats, normalize_by_oc,
                   FIG_DIR, save_stats, N_CONDITIONS)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10,
                     'axes.labelsize': 9, 'figure.dpi': 120})

COLORS = cm.tab10(np.linspace(0, 0.6, N_CONDITIONS))

# ── Load ───────────────────────────────────────────────────────────────────────
train, test, _ = load_data()
train = add_rul(train)
km    = fit_oc_kmeans(train)
train = assign_oc(train, km)

valid_sensors, _ = get_valid_sensors(train)
oc_stats = compute_oc_stats(train, valid_sensors)

# ── OC mean shift table ────────────────────────────────────────────────────────
means_by_oc = pd.DataFrame(
    {oc: [oc_stats[oc]['mean'][s] for s in valid_sensors]
     for oc in sorted(oc_stats.keys())},
    index=valid_sensors
)
print("Sensor mean by Operating Condition:")
print(means_by_oc.round(3).to_string())
print()

# Coefficient of variation of means across OCs (how much OC shifts the sensor)
oc_shift = means_by_oc.std(axis=1) / means_by_oc.mean(axis=1).abs().replace(0, 1)
oc_shift = oc_shift.sort_values(ascending=False)
print("Sensors most affected by OC (CoV of OC means):")
print(oc_shift.round(4).to_string())
print()

save_stats({"oc_shift_top3": oc_shift.head(3).index.tolist()})

# ── Figure 1 : Boxplot of each valid sensor grouped by OC ─────────────────────
n_sensors = len(valid_sensors)
ncols = 4
nrows = int(np.ceil(n_sensors / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2.5))
axes = axes.flat

for ax, sensor in zip(axes, valid_sensors):
    data_by_oc = [train[train['OC'] == oc][sensor].values
                  for oc in sorted(train['OC'].unique())]
    bp = ax.boxplot(data_by_oc,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3),
                    boxprops=dict(linewidth=0.8))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(sensor, fontsize=9, pad=2)
    ax.set_xticklabels([f'OC{o}' for o in range(N_CONDITIONS)], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

# Hide unused axes
for ax in list(axes)[n_sensors:]:
    ax.set_visible(False)

plt.suptitle("Sensor Distribution by Operating Condition — FD004 Training Set",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_sensor_boxplot_by_oc.png"),
            bbox_inches='tight')
plt.close()
print("Saved: 04_sensor_boxplot_by_oc.png")

# ── Figure 2 : OC Mean Offset Heatmap ─────────────────────────────────────────
# Normalize each sensor's OC means to [0,1] to show relative shift
norm_means = (means_by_oc - means_by_oc.min(axis=1).values[:, None]) / \
             (means_by_oc.max(axis=1) - means_by_oc.min(axis=1)).values[:, None].clip(1e-9)

fig, ax = plt.subplots(figsize=(9, max(4, n_sensors * 0.45)))
cmap = plt.get_cmap('YlOrRd')
im = ax.imshow(norm_means.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Normalised OC mean (0=min, 1=max)')
ax.set_xticks(range(N_CONDITIONS))
ax.set_xticklabels([f'OC{o}' for o in sorted(oc_stats.keys())])
ax.set_yticks(range(n_sensors))
ax.set_yticklabels(valid_sensors, fontsize=8)
ax.set_xlabel("Operating Condition")
ax.set_ylabel("Sensor")
ax.set_title("Sensor Mean Shift Across Operating Conditions\n"
             "(brighter = higher relative mean under that OC)")
for i in range(n_sensors):
    for j in range(N_CONDITIONS):
        ax.text(j, i, f'{norm_means.values[i, j]:.2f}',
                ha='center', va='center', fontsize=6.5,
                color='black' if norm_means.values[i, j] < 0.6 else 'white')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_oc_offset_heatmap.png"))
plt.close()
print("Saved: 04_oc_offset_heatmap.png")

# ── Figure 3 : Before vs After OC Normalisation (2 sensors vs RUL) ─────────────
train_norm = normalize_by_oc(train, oc_stats, valid_sensors)

# Pick the sensor with highest OC shift and highest |Pearson corr| with RUL
top_shifted_sensor = oc_shift.index[0]
rul_corr = train[valid_sensors].corrwith(train['RUL']).abs()
top_corr_sensor   = rul_corr.idxmax()
show_sensors = list(dict.fromkeys([top_shifted_sensor, top_corr_sensor]))[:2]
if len(show_sensors) < 2:
    show_sensors = valid_sensors[:2]

sample = train_norm.sample(min(4000, len(train_norm)), random_state=1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, sensor in enumerate(show_sensors):
    norm_col = f'{sensor}_norm'
    # Before normalisation (colour by OC)
    ax = axes[row, 0]
    for oc in sorted(sample['OC'].unique()):
        sub = sample[sample['OC'] == oc]
        ax.scatter(sub['RUL'], sub[sensor], s=3, alpha=0.4,
                   color=COLORS[oc], label=f'OC{oc}')
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel(sensor)
    ax.set_title(f"{sensor} vs RUL  [Before normalisation]")
    ax.legend(markerscale=2, fontsize=7, loc='best')
    ax.grid(linestyle=':', alpha=0.4)

    # After normalisation
    ax = axes[row, 1]
    for oc in sorted(sample['OC'].unique()):
        sub = sample[sample['OC'] == oc]
        ax.scatter(sub['RUL'], sub[norm_col], s=3, alpha=0.4,
                   color=COLORS[oc], label=f'OC{oc}')
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel(f'{sensor} (normalised)')
    ax.set_title(f"{sensor} vs RUL  [After OC normalisation]")
    ax.legend(markerscale=2, fontsize=7, loc='best')
    ax.grid(linestyle=':', alpha=0.4)

plt.suptitle("Effect of OC Normalisation on Sensor–RUL Relationship",
             fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_before_after_norm.png"))
plt.close()
print("Saved: 04_before_after_norm.png")

print("\nPhase 4 complete.")
