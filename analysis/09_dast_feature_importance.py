"""
Phase 9 — DAST Feature Importance Analysis
Two complementary methods:
  1. Permutation Importance  — shuffle each sensor across all test samples,
                               measure RMSE increase (model-agnostic, intuitive)
  2. Gradient Saliency Map   — |grad × input| averaged over all samples,
                               produces (time_step × sensor) heatmap

Outputs:
  figures/09_dast_permutation_importance.png
  figures/09_dast_gradient_saliency_heatmap.png
  figures/09_dast_sensor_time_importance.png   (time-step aggregated saliency)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import scipy.io as sio
import torch
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import FIG_DIR, RUL_CAP

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                     'axes.labelsize': 11, 'figure.dpi': 130})

# ── Paths & config ────────────────────────────────────────────────────────────
ROOT = r"C:\Users\hjs92\Desktop\DAST_test"
sys.path.insert(0, ROOT)

from DAST_Network import DAST

with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
_ds  = cfg["cmapss"]
_mdl = cfg["model"]

DATASET    = _ds["dataset"]
RUL_MAX    = float(RUL_CAP)
DATA_DIR   = os.path.join(ROOT, "train_dataset")
MODEL_PATH = os.path.join(ROOT, f"dast_{DATASET}_best.pth")

# ── Sensor labels (14 sensors remaining after preprocessing) ─────────────────
# Original 26 cols: unit, cycle, op1,op2,op3, s1-s21
# Removed indices [2,3,4,5,9,10,14,20,22,23] → op1,op2,op3,s1,s5,s6,s10,s16,s18,s19
# Then unit & cycle removed → 14 sensor features:
SENSOR_NAMES = ['s2','s3','s4','s7','s8','s9',
                's11','s12','s13','s14','s15','s17','s20','s21']

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading test data...")
testX = sio.loadmat(os.path.join(DATA_DIR, f"{DATASET}_window_size_testX_new.mat"))["test1X_new"]
testY = sio.loadmat(os.path.join(DATA_DIR, f"{DATASET}_window_size_testY.mat"))["test1Y"].flatten()
print(f"  testX shape: {testX.shape}  testY shape: {testY.shape}")

time_step  = testX.shape[1]   # 62
input_size = testX.shape[2]   # 14

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cpu")
model = DAST(
    dim_val_s=_mdl["dim_val"], dim_attn_s=_mdl["dim_attn"],
    dim_val_t=_mdl["dim_val"], dim_attn_t=_mdl["dim_attn"],
    dim_val=_mdl["dim_val"],   dim_attn=_mdl["dim_attn"],
    time_step=time_step, input_size=input_size,
    dec_seq_len=_mdl["dec_seq_len"], out_seq_len=_mdl["out_seq_len"],
    n_encoder_layers=_mdl["n_encoder_layers"],
    n_decoder_layers=_mdl["n_decoder_layers"],
    n_heads=_mdl["n_heads"], dropout=_mdl["dropout"],
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

testX_t = torch.tensor(testX, dtype=torch.float32)

# ── Baseline RMSE ─────────────────────────────────────────────────────────────
with torch.no_grad():
    baseline_pred = model(testX_t).squeeze().numpy()
baseline_rmse = float(np.sqrt(np.mean((baseline_pred - testY) ** 2)) * RUL_MAX)
print(f"Baseline RMSE: {baseline_rmse:.4f} cycles")

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Permutation Importance
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing permutation importance (14 sensors × 30 repeats)...")
N_REPEAT = 30
perm_importance = np.zeros((input_size, N_REPEAT))

rng = np.random.default_rng(42)
for s_idx in range(input_size):
    for r in range(N_REPEAT):
        X_perm = testX.copy()
        shuffle_idx = rng.permutation(len(testX))
        X_perm[:, :, s_idx] = testX[shuffle_idx, :, s_idx]
        with torch.no_grad():
            pred_p = model(torch.tensor(X_perm, dtype=torch.float32)).squeeze().numpy()
        perm_rmse = float(np.sqrt(np.mean((pred_p - testY) ** 2)) * RUL_MAX)
        perm_importance[s_idx, r] = perm_rmse - baseline_rmse

perm_mean = perm_importance.mean(axis=1)
perm_std  = perm_importance.std(axis=1)
sort_idx  = np.argsort(perm_mean)[::-1]

print(f"\n  Permutation Importance (RMSE increase, cycles):")
print(f"  {'Sensor':<8} {'Mean':>8} {'Std':>8}")
for i in sort_idx:
    print(f"  {SENSOR_NAMES[i]:<8} {perm_mean[i]:>8.3f} {perm_std[i]:>8.3f}")

# Figure 1 — Permutation Importance bar chart
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#e05c5c' if v > 0 else '#8ec6e6' for v in perm_mean[sort_idx]]
bars = ax.barh(
    [SENSOR_NAMES[i] for i in sort_idx][::-1],
    perm_mean[sort_idx][::-1],
    xerr=perm_std[sort_idx][::-1],
    color=colors[::-1], edgecolor='white', linewidth=0.5,
    capsize=3, error_kw={'elinewidth': 1, 'ecolor': 'gray'}
)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("RMSE Increase when Sensor is Permuted (cycles)\n(Larger = More Important to the Model)")
ax.set_title(f"DAST — Permutation Feature Importance\n(Baseline RMSE = {baseline_rmse:.2f} cycles, {N_REPEAT} repeats)")
ax.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09_dast_permutation_importance.png"))
plt.close()
print("\nSaved: 09_dast_permutation_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Gradient × Input Saliency Map
# ═══════════════════════════════════════════════════════════════════════════════
print("\nComputing gradient saliency map...")
model.eval()

testX_grad = torch.tensor(testX, dtype=torch.float32, requires_grad=True)
output = model(testX_grad)
# Scalar loss = mean of predictions
loss = output.mean()
loss.backward()

# saliency: |grad × input|, shape (N, time_step, input_size)
saliency = (testX_grad.grad * testX_grad).abs().detach().numpy()

# Mean over all test samples → (time_step, input_size)
saliency_mean = saliency.mean(axis=0)          # (62, 14)

# ── Figure 2 — Heatmap (sensor × time step) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(saliency_mean.T, aspect='auto', cmap='YlOrRd',
               interpolation='nearest')
plt.colorbar(im, ax=ax, label="|Gradient × Input|")
ax.set_yticks(range(input_size))
ax.set_yticklabels(SENSOR_NAMES, fontsize=9)
ax.set_xlabel("Time Step (0 = oldest, 61 = newest)")
ax.set_ylabel("Sensor")
ax.set_title("DAST — Gradient Saliency Map\n(x=Time Step, y=Sensor, Darker=Higher Influence on Prediction)")
# Mark the boundary between window data and appended stat rows
ax.axvline(59.5, color='white', linewidth=1.5, linestyle='--',
           label='60-cycle window | stat rows')
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09_dast_gradient_saliency_heatmap.png"))
plt.close()
print("Saved: 09_dast_gradient_saliency_heatmap.png")

# ── Figure 3 — Combined: sensor importance + time-step importance ─────────────
sensor_imp_grad = saliency_mean.mean(axis=0)   # avg over time → (14,)
time_imp_grad   = saliency_mean.mean(axis=1)   # avg over sensors → (62,)

# Sort sensor importance
s_sort = np.argsort(sensor_imp_grad)[::-1]

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# 2a — Sensor importance from gradient (bar)
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(
    [SENSOR_NAMES[i] for i in s_sort][::-1],
    sensor_imp_grad[s_sort][::-1],
    color='#e07b39', edgecolor='white', linewidth=0.4
)
ax1.set_xlabel("|Gradient x Input| (mean over samples)")
ax1.set_title("Gradient Saliency — Sensor Importance")
ax1.grid(axis='x', linestyle=':', alpha=0.5)

# 2b — Permutation importance (bar, sorted)
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(
    [SENSOR_NAMES[i] for i in sort_idx][::-1],
    perm_mean[sort_idx][::-1],
    color='#5b9bd5', edgecolor='white', linewidth=0.4
)
ax2.set_xlabel("RMSE Increase (cycles)")
ax2.set_title("Permutation Importance — Sensor Importance")
ax2.grid(axis='x', linestyle=':', alpha=0.5)

# 2c — Time-step importance line (60 window + 2 stat rows)
ax3 = fig.add_subplot(gs[1, :])
window_t = time_imp_grad[:60]
stat_t   = time_imp_grad[60:]
ax3.plot(range(60), window_t, color='steelblue', linewidth=1.5,
         label='Sliding Window Steps (0-59)')
ax3.scatter([60, 61], stat_t, color='tomato', s=60, zorder=5,
            label='Appended Stat Rows (60=regression coeff, 61=mean)')
ax3.axvline(59.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xlabel("Time Step Index")
ax3.set_ylabel("|Gradient x Input| (mean over sensors)")
ax3.set_title("Gradient Saliency — Time-Step Importance\n(Recent steps near failure tend to have higher importance)")
ax3.legend(fontsize=9)
ax3.grid(linestyle=':', alpha=0.4)

plt.suptitle(f"DAST Feature Importance Analysis — {DATASET}", fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09_dast_sensor_time_importance.png"))
plt.close()
print("Saved: 09_dast_sensor_time_importance.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  DAST Feature Importance Summary")
print("=" * 55)
print("\n  Top 5 Sensors (Permutation Importance):")
for rank, i in enumerate(sort_idx[:5], 1):
    print(f"  {rank}. {SENSOR_NAMES[i]:<6}  RMSE↑ = {perm_mean[i]:.3f} ± {perm_std[i]:.3f} cycles")

print("\n  Top 5 Sensors (Gradient Saliency):")
for rank, i in enumerate(s_sort[:5], 1):
    print(f"  {rank}. {SENSOR_NAMES[i]:<6}  |grad×input| = {sensor_imp_grad[i]:.5f}")

print("\n  Most Important Time Steps (top-5, gradient):")
top_t = np.argsort(time_imp_grad)[::-1][:5]
for rank, t in enumerate(top_t, 1):
    label = f"stat_row[{t-60}]" if t >= 60 else f"t={t}"
    print(f"  {rank}. {label:<12}  |grad×input| = {time_imp_grad[t]:.5f}")

print("=" * 55)
print("\nPhase 9 complete — DAST feature importance done.")
