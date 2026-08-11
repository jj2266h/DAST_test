"""
Phase 1 — Data Loading & Basic EDA
Outputs:
  figures/01_engine_life_dist.png
  figures/01_rul_distribution.png
  figures/01_test_life_dist.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import (load_data, add_rul, FIG_DIR, save_stats, RUL_CAP)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'figure.dpi': 120})

# ── Load ───────────────────────────────────────────────────────────────────────
train, test, rul_df = load_data()
train = add_rul(train)

# ── Basic Stats ────────────────────────────────────────────────────────────────
train_units  = train['unit'].nunique()
test_units   = test['unit'].nunique()
train_rows   = len(train)
test_rows    = len(test)
train_cycles = train.groupby('unit')['cycle'].max()
test_cycles  = test.groupby('unit')['cycle'].max()

print("=" * 55)
print("  CMAPSS FD004  —  Data Overview")
print("=" * 55)
print(f"  Train engines  : {train_units}")
print(f"  Train rows     : {train_rows:,}")
print(f"  Test  engines  : {test_units}")
print(f"  Test  rows     : {test_rows:,}")
print(f"  Train life  min/mean/max : "
      f"{train_cycles.min():.0f} / {train_cycles.mean():.1f} / {train_cycles.max():.0f} cycles")
print(f"  Test  life  min/mean/max : "
      f"{test_cycles.min():.0f} / {test_cycles.mean():.1f} / {test_cycles.max():.0f} cycles")
print(f"  RUL cap        : {RUL_CAP} cycles")
print(f"  RUL (test) min/mean/max  : "
      f"{rul_df['RUL_true'].min()} / {rul_df['RUL_true'].mean():.1f} / {rul_df['RUL_true'].max()}")
print(f"  Sensors        : 21  |  Operating settings : 3")
print(f"  Fault modes    : 2   |  Operating conditions: 6")
print()
print("  Sensor descriptive stats (training set):")
print(train[[f's{i}' for i in range(1, 22)]].describe().round(3).to_string())
print("=" * 55)

stats = {
    "train_units": train_units,
    "test_units":  test_units,
    "train_rows":  train_rows,
    "test_rows":   test_rows,
    "train_life_min":  int(train_cycles.min()),
    "train_life_mean": round(float(train_cycles.mean()), 1),
    "train_life_max":  int(train_cycles.max()),
    "test_life_min":   int(test_cycles.min()),
    "test_life_mean":  round(float(test_cycles.mean()), 1),
    "test_life_max":   int(test_cycles.max()),
    "rul_true_min":    int(rul_df['RUL_true'].min()),
    "rul_true_mean":   round(float(rul_df['RUL_true'].mean()), 1),
    "rul_true_max":    int(rul_df['RUL_true'].max()),
}
save_stats(stats)

# ── Figure 1 : Engine Life Distribution (Train) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_cycles.values, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
ax.set_xlabel("Engine Life (cycles)")
ax.set_ylabel("Number of Engines")
ax.set_title("FD004 Training Set — Engine Life Distribution")
ax.axvline(train_cycles.mean(), color='tomato', linestyle='--',
           linewidth=1.5, label=f"Mean = {train_cycles.mean():.1f}")
ax.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_engine_life_dist.png"))
plt.close()
print("Saved: 01_engine_life_dist.png")

# ── Figure 2 : Piecewise RUL Distribution ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: raw RUL
raw_rul = train.groupby('unit').apply(
    lambda g: (g['cycle'].max() - g['cycle'])).reset_index(drop=True)
axes[0].hist(raw_rul.values, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].axvline(RUL_CAP, color='tomato', linestyle='--', linewidth=1.5,
                label=f'Cap = {RUL_CAP}')
axes[0].set_title("Raw RUL (before cap)")
axes[0].set_xlabel("RUL (cycles)")
axes[0].set_ylabel("Count")
axes[0].legend()

# Right: piecewise RUL (capped)
axes[1].hist(train['RUL'].values, bins=40, color='seagreen', edgecolor='white', linewidth=0.5)
axes[1].set_title(f"Piecewise RUL (capped at {RUL_CAP})")
axes[1].set_xlabel("RUL (cycles)")
axes[1].set_ylabel("Count")

plt.suptitle("FD004 Training Set — RUL Label Distribution", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_rul_distribution.png"))
plt.close()
print("Saved: 01_rul_distribution.png")

# ── Figure 3 : Test Set Life Distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(test_cycles.values, bins=30, color='darkorange', edgecolor='white', linewidth=0.5)
ax.set_xlabel("Recorded Cycles (test)")
ax.set_ylabel("Number of Engines")
ax.set_title("FD004 Test Set — Recorded Life Distribution")
ax.axvline(test_cycles.mean(), color='navy', linestyle='--',
           linewidth=1.5, label=f"Mean = {test_cycles.mean():.1f}")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_test_life_dist.png"))
plt.close()
print("Saved: 01_test_life_dist.png")

print("\nPhase 1 complete.")
