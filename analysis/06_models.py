"""
Phase 6 — Random Forest RUL Prediction
Features: OC-normalised sensors + rolling mean/std (window=30) + cycle + OC label
Outputs:
  figures/06_feature_importance.png
  figures/06_pred_vs_actual.png
  figures/06_error_distribution.png
  figures/06_error_by_rul.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import (load_data, add_rul, fit_oc_kmeans, assign_oc,
                   get_valid_sensors, compute_oc_stats, normalize_by_oc,
                   add_rolling_features, FIG_DIR, save_stats, RUL_CAP, WINDOW)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12,
                     'axes.labelsize': 10, 'figure.dpi': 120})

# ── 1. Load & Preprocess ───────────────────────────────────────────────────────
print("Loading data...")
train, test, rul_df = load_data()
train = add_rul(train)

km    = fit_oc_kmeans(train)
train = assign_oc(train, km)
test  = assign_oc(test, km)

valid_sensors, _ = get_valid_sensors(train)
norm_cols = [f'{s}_norm' for s in valid_sensors]

oc_stats = compute_oc_stats(train, valid_sensors)
train = normalize_by_oc(train, oc_stats, valid_sensors)
test  = normalize_by_oc(test,  oc_stats, valid_sensors)

# ── 2. Rolling Features ────────────────────────────────────────────────────────
print(f"Adding rolling features (window={WINDOW})...")
train = add_rolling_features(train, norm_cols, window=WINDOW)
test  = add_rolling_features(test,  norm_cols, window=WINDOW)

# Feature columns
feat_cols = (norm_cols
             + [f'{c}_rmean' for c in norm_cols]
             + [f'{c}_rstd'  for c in norm_cols]
             + ['cycle', 'OC'])

# ── 3. Training Set ────────────────────────────────────────────────────────────
# Skip first (window-1) cycles per engine to let rolling stats stabilise
def filter_window_rows(df, window=WINDOW):
    keep = []
    for _, grp in df.groupby('unit', sort=False):
        grp = grp.sort_values('cycle')
        keep.append(grp.iloc[window - 1:])
    return pd.concat(keep)

train_filtered = filter_window_rows(train, WINDOW)
X_train = train_filtered[feat_cols].values
y_train = train_filtered['RUL'].values

# ── 4. Test Set (last row of each engine) ──────────────────────────────────────
test_last = (test.sort_values(['unit', 'cycle'])
                 .groupby('unit', sort=False)
                 .last()
                 .reset_index())
X_test    = test_last[feat_cols].values
y_test    = rul_df.sort_values('unit')['RUL_true'].values

# ── 5. Train RF ────────────────────────────────────────────────────────────────
print("Training Random Forest (100 trees)...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=1
)
rf.fit(X_train, y_train)
print("Done.")

# ── 6. Evaluate ────────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test).clip(0, RUL_CAP)

rmse  = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae   = float(mean_absolute_error(y_test, y_pred))
mape  = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100)
errors = y_pred - y_test

# PHM'08 asymmetric score
def phm_score(true, pred):
    e = pred - true
    s = np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)
    return float(s.sum())

score = phm_score(y_test, y_pred)

print("=" * 50)
print("  Random Forest — Test Set Performance")
print("=" * 50)
print(f"  RMSE   : {rmse:.2f} cycles")
print(f"  MAE    : {mae:.2f} cycles")
print(f"  MAPE   : {mape:.2f}%")
print(f"  PHM'08 Score : {score:.1f}")
print(f"  # Test engines : {len(y_test)}")
print("=" * 50)

save_stats({
    "rf_rmse":  round(rmse,  2),
    "rf_mae":   round(mae,   2),
    "rf_mape":  round(mape,  2),
    "rf_phm_score": round(score, 1),
    "n_features": len(feat_cols),
    "n_train_samples": len(X_train),
})

# ── 7. Feature Importance ──────────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=feat_cols)
importances = importances.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, max(5, len(feat_cols) * 0.22)))
top_n = min(30, len(feat_cols))
top_imp = importances.head(top_n)
ax.barh(top_imp.index[::-1], top_imp.values[::-1],
        color='steelblue', edgecolor='white', linewidth=0.4)
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
ax.set_title(f"Random Forest — Top {top_n} Feature Importances")
ax.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_feature_importance.png"))
plt.close()
print("Saved: 06_feature_importance.png")

# ── 8. Predicted vs Actual ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, s=15, alpha=0.6, color='steelblue', zorder=3)
lim = max(y_test.max(), y_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='Perfect prediction')
ax.fill_between([0, lim], [15, lim + 15], [-15, lim - 15],
                alpha=0.12, color='tomato', label='±15 cycles')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"Predicted vs Actual RUL\n"
             f"RMSE={rmse:.1f}  MAE={mae:.1f}  MAPE={mape:.1f}%")
ax.legend(fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_pred_vs_actual.png"))
plt.close()
print("Saved: 06_pred_vs_actual.png")

# ── 9. Error Distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(0,           color='black',  linestyle='--', linewidth=1.2)
ax.axvline(errors.mean(), color='tomato', linestyle='--', linewidth=1.2,
           label=f'Mean error = {errors.mean():.1f}')
ax.set_xlabel("Prediction Error (Predicted − Actual, cycles)")
ax.set_ylabel("Count")
ax.set_title("Random Forest — Prediction Error Distribution")
ax.legend(fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_error_distribution.png"))
plt.close()
print("Saved: 06_error_distribution.png")

# ── 10. Error by True RUL ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(y_test, errors, s=15, alpha=0.7,
                c=np.abs(errors), cmap='YlOrRd', zorder=3)
plt.colorbar(sc, ax=ax, label="|Error| (cycles)")
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axhline(15,  color='tomato', linestyle=':', linewidth=1)
ax.axhline(-15, color='tomato', linestyle=':', linewidth=1)
ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Prediction Error (cycles)")
ax.set_title("Prediction Error vs Actual RUL")
ax.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_error_by_rul.png"))
plt.close()
print("Saved: 06_error_by_rul.png")

print("\nPhase 6 complete.")
