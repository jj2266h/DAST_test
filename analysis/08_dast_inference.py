"""
Phase 8 — DAST Model Inference on FD004
Loads the pre-trained dast_FD004_best.pth and evaluates on the test set.
Generates comparison figures with Random Forest baseline.
Outputs:
  figures/08_dast_pred_vs_actual.png
  figures/08_dast_error_dist.png
  figures/08_dast_vs_rf_comparison.png
  figures/08_dast_error_by_rul.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # project root
sys.path.insert(0, os.path.dirname(__file__))                   # analysis/

import numpy as np
import scipy.io as sio
import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import FIG_DIR, save_stats, load_stats, RUL_CAP

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'figure.dpi': 120})

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = r"C:\Users\hjs92\Desktop\DAST_test"
DATA_DIR    = os.path.join(ROOT, "train_dataset")
RUL_MAX     = float(RUL_CAP)     # 125

# ── Load model architecture from project root ──────────────────────────────────
from DAST_Network import DAST

# ── Read config.json (same as DAST_test.py) ────────────────────────────────────
import json
with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as _f:
    _cfg = json.load(_f)

_ds  = _cfg["cmapss"]
_mdl = _cfg["model"]

DATASET    = _ds["dataset"]                        # e.g. "FD004"
MODEL_PATH = os.path.join(ROOT, f"dast_{DATASET}_best.pth")

def compute_rmse(pred, true, rul_max=1.0):
    return float(np.sqrt(np.mean((pred - true) ** 2)) * rul_max)

def compute_mae(pred, true, rul_max=1.0):
    return float(np.mean(np.abs(pred - true)) * rul_max)

def s_score(y_true, y_pred):
    diff = np.array(y_pred) - np.array(y_true)
    score = [(math.exp(-d / 13.0) - 1.0) if d < 0 else (math.exp(d / 10.0) - 1.0)
             for d in diff]
    return float(np.sum(score))

# ── 1. Load preprocessed data ─────────────────────────────────────────────────
print("Loading preprocessed FD004 data...")
testX  = sio.loadmat(os.path.join(DATA_DIR, f"{DATASET}_window_size_testX_new.mat"))["test1X_new"]
testY  = sio.loadmat(os.path.join(DATA_DIR, f"{DATASET}_window_size_testY.mat"))["test1Y"].flatten()

print(f"  testX  shape : {testX.shape}")   # (248, 62, 14)
print(f"  testY  shape : {testY.shape}")   # (248,)  — normalised [0,1]

time_step  = testX.shape[1]   # 62 (60 window + 2 stat features)
input_size = testX.shape[2]   # 14 sensors
print(f"  time_step={time_step}, input_size={input_size}")

# ── 2. Build model (parameters from config.json, same as DAST_test.py) ────────
device = torch.device("cpu")
model = DAST(
    dim_val_s=_mdl["dim_val"],
    dim_attn_s=_mdl["dim_attn"],
    dim_val_t=_mdl["dim_val"],
    dim_attn_t=_mdl["dim_attn"],
    dim_val=_mdl["dim_val"],
    dim_attn=_mdl["dim_attn"],
    time_step=time_step,
    input_size=input_size,
    dec_seq_len=_mdl["dec_seq_len"],
    out_seq_len=_mdl["out_seq_len"],
    n_encoder_layers=_mdl["n_encoder_layers"],
    n_decoder_layers=_mdl["n_decoder_layers"],
    n_heads=_mdl["n_heads"],
    dropout=_mdl["dropout"],
).to(device)

# ── 3. Load weights ────────────────────────────────────────────────────────────
print(f"Loading weights: {MODEL_PATH}")
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print("Weights loaded successfully.")

# ── 4. Inference ───────────────────────────────────────────────────────────────
testX_tensor = torch.tensor(testX, dtype=torch.float32).to(device)
with torch.no_grad():
    pred_norm = model(testX_tensor).squeeze().cpu().numpy()

# Denormalise (multiply by RUL_MAX)
y_pred_dast = (pred_norm * RUL_MAX).clip(0, RUL_MAX)
y_true      = (testY    * RUL_MAX)

# ── 5. Metrics ────────────────────────────────────────────────────────────────
rmse_dast  = compute_rmse(pred_norm, testY, RUL_MAX)
mae_dast   = compute_mae (pred_norm, testY, RUL_MAX)
mape_dast  = float(np.mean(np.abs((y_true - y_pred_dast) / (y_true + 1e-6))) * 100)
score_dast = s_score(y_true, y_pred_dast)
errors_dast = y_pred_dast - y_true

print("\n" + "=" * 55)
print("  DAST Model — FD004 Test Set Performance")
print("=" * 55)
print(f"  RMSE        : {rmse_dast:.2f} cycles")
print(f"  MAE         : {mae_dast:.2f} cycles")
print(f"  MAPE        : {mape_dast:.2f}%")
print(f"  PHM'08 Score: {score_dast:.1f}  (lower = better)")
print(f"  # Engines   : {len(y_true)}")
print("=" * 55)

# Load RF results for comparison
prev = load_stats()
rf_rmse  = prev.get("rf_rmse",  "N/A")
rf_mae   = prev.get("rf_mae",   "N/A")
rf_mape  = prev.get("rf_mape",  "N/A")
rf_score = prev.get("rf_phm_score", "N/A")

print("\n  Comparison with Random Forest Baseline:")
print(f"  {'Metric':<14} {'DAST':>10} {'RF':>10} {'Winner':>8}")
print(f"  {'-'*44}")
for metric, dv, rv in [
    ("RMSE",       rmse_dast,  rf_rmse),
    ("MAE",        mae_dast,   rf_mae),
    ("MAPE (%)",   mape_dast,  rf_mape),
    ("PHM Score",  score_dast, rf_score),
]:
    try:
        winner = "DAST" if float(dv) < float(rv) else "RF"
    except Exception:
        winner = "?"
    print(f"  {metric:<14} {float(dv):>10.2f} {float(rv):>10.2f} {winner:>8}")

save_stats({
    "dast_rmse":       round(float(rmse_dast),  2),
    "dast_mae":        round(float(mae_dast),   2),
    "dast_mape":       round(float(mape_dast),  2),
    "dast_phm_score":  round(float(score_dast), 1),
})

# ── 6. Figure: Predicted vs Actual ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_true, y_pred_dast, s=15, alpha=0.65, color='darkorange', zorder=3)
lim = max(y_true.max(), y_pred_dast.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, label='Perfect prediction')
ax.fill_between([0, lim], [15, lim + 15], [-15, lim - 15],
                alpha=0.12, color='steelblue', label='±15 cycles')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"DAST — Predicted vs Actual RUL\n"
             f"RMSE={rmse_dast:.1f}  MAE={mae_dast:.1f}  MAPE={mape_dast:.1f}%")
ax.legend(fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_dast_pred_vs_actual.png"))
plt.close()
print("Saved: 08_dast_pred_vs_actual.png")

# ── 7. Figure: Error Distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors_dast, bins=30, color='darkorange', edgecolor='white',
        linewidth=0.5, alpha=0.8)
ax.axvline(0, color='black', linestyle='--', linewidth=1.2)
ax.axvline(errors_dast.mean(), color='steelblue', linestyle='--', linewidth=1.5,
           label=f'Mean error = {errors_dast.mean():.1f}')
ax.set_xlabel("Prediction Error (Predicted − Actual, cycles)")
ax.set_ylabel("Count")
ax.set_title("DAST — Prediction Error Distribution")
ax.legend(fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_dast_error_dist.png"))
plt.close()
print("Saved: 08_dast_error_dist.png")

# ── 8. Figure: DAST vs RF Side-by-Side ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sort_idx = np.argsort(y_true)
x_range  = np.arange(len(y_true))

for ax, (pred, label, color) in zip(
    axes,
    [(y_pred_dast, f'DAST  (RMSE={rmse_dast:.1f})', 'darkorange'),
     (None,        f'RF    (RMSE={rf_rmse})',         'steelblue')]
):
    ax.plot(x_range, y_true[sort_idx], 'k-', lw=1.2, label='True RUL', zorder=3)
    if pred is not None:
        ax.plot(x_range, pred[sort_idx], '-', color=color, lw=1, alpha=0.8,
                label=label)
        ax.fill_between(x_range, y_true[sort_idx], pred[sort_idx],
                        alpha=0.2, color=color)
    ax.set_xlabel("Test Engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(linestyle=':', alpha=0.4)
    ax.set_ylim(0, RUL_MAX * 1.15)

plt.suptitle("FD004 Test Set: True RUL vs Model Predictions", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_dast_vs_rf_comparison.png"))
plt.close()
print("Saved: 08_dast_vs_rf_comparison.png")

# ── 9. Figure: Error vs True RUL ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(y_true, errors_dast, s=15, alpha=0.7,
                c=np.abs(errors_dast), cmap='YlOrRd', zorder=3)
plt.colorbar(sc, ax=ax, label="|Error| (cycles)")
ax.axhline(0,    color='black', linestyle='--', linewidth=1)
ax.axhline( 15,  color='tomato', linestyle=':', linewidth=1, label='±15 cycles')
ax.axhline(-15,  color='tomato', linestyle=':', linewidth=1)
ax.set_xlabel("Actual RUL (cycles)")
ax.set_ylabel("Prediction Error (cycles)")
ax.set_title("DAST — Prediction Error vs Actual RUL")
ax.legend(fontsize=9)
ax.grid(linestyle=':', alpha=0.4)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_dast_error_by_rul.png"))
plt.close()
print("Saved: 08_dast_error_by_rul.png")

print("\nPhase 8 complete — DAST inference done.")
