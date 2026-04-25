# -*- coding: utf-8 -*-
"""
PHM2012 全軸承 PSP 批次偵測（RMS 平滑基線法）
- 對所有 17 個軸承偵測 PSP
- 方法：滑動窗口平滑 RMS → 超過健康基準 μ+3σ 時觸發
- 輸出：每個軸承的退化圖 + 彙總表格
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 設定 ──────────────────────────────────────────────────────
FEMTO_PATH      = "phm2012_dataset"
REFERENCE_RATIO = 0.15      # 前 15% 作健康基準
SIGMA_K         = 3.0
SMOOTH_WINDOW   = 20        # RMS 平滑窗口（檔案數）
CONSEC          = 5         # 連續超標次數
SAVE_DIR        = "plots/psp_rms"

ALL_BEARINGS = [
    ("Bearing1_1", "Learning_set/Bearing1_1",   True),
    ("Bearing1_2", "Learning_set/Bearing1_2",   True),
    ("Bearing2_1", "Learning_set/Bearing2_1",   True),
    ("Bearing2_2", "Learning_set/Bearing2_2",   True),
    ("Bearing3_1", "Learning_set/Bearing3_1",   True),
    ("Bearing3_2", "Learning_set/Bearing3_2",   True),
    ("Bearing1_3", "Full_Test_Set/Bearing1_3",  False),
    ("Bearing1_4", "Full_Test_Set/Bearing1_4",  False),
    ("Bearing1_5", "Full_Test_Set/Bearing1_5",  False),
    ("Bearing1_6", "Full_Test_Set/Bearing1_6",  False),
    ("Bearing1_7", "Full_Test_Set/Bearing1_7",  False),
    ("Bearing2_3", "Full_Test_Set/Bearing2_3",  False),
    ("Bearing2_4", "Full_Test_Set/Bearing2_4",  False),
    ("Bearing2_5", "Full_Test_Set/Bearing2_5",  False),
    ("Bearing2_6", "Full_Test_Set/Bearing2_6",  False),
    ("Bearing2_7", "Full_Test_Set/Bearing2_7",  False),
    ("Bearing3_3", "Full_Test_Set/Bearing3_3",  False),
]


# ── 工具函數 ──────────────────────────────────────────────────

def load_rms(bearing_dir: str) -> np.ndarray:
    accs = sorted(f for f in os.listdir(bearing_dir) if f.startswith('acc'))
    out = []
    for acc in accs:
        sig = pd.read_csv(
            os.path.join(bearing_dir, acc),
            header=None, sep='[,;]', usecols=[4], engine='python'
        ).values.flatten().astype(np.float64)
        out.append(float(np.sqrt(np.mean(sig ** 2))))
    return np.array(out)


def detect_rms_baseline(rms: np.ndarray,
                        ref_ratio=0.15, k=3.0,
                        smooth_w=20, consec=5):
    """
    1. 平滑 RMS（rolling mean）
    2. 前 ref_ratio 段為健康基準，計算 μ + k·σ 閾值
    3. 連續 consec 點超標 → PSP
    """
    smoothed  = pd.Series(rms).rolling(smooth_w, center=True,
                                       min_periods=1).mean().values
    n_ref     = max(10, int(len(smoothed) * ref_ratio))
    ref       = smoothed[:n_ref]
    mu, sigma = ref.mean(), ref.std()
    threshold = mu + k * sigma

    count = 0
    for i in range(n_ref, len(smoothed)):
        if smoothed[i] > threshold:
            count += 1
            if count >= consec:
                return i - consec + 1, threshold, smoothed
        else:
            count = 0
    return len(smoothed) - 1, threshold, smoothed


# ── 主流程 ────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = []

    print(f"{'Bearing':12s} {'Split':5s} {'N':>5s}  {'PSP':>5s}  {'PSP%':>6s}")
    print("-" * 45)

    all_rms, all_psp, all_thr, all_sm = {}, {}, {}, {}

    for (name, subfolder, is_train) in ALL_BEARINGS:
        bearing_dir = os.path.join(FEMTO_PATH, subfolder)
        if not os.path.isdir(bearing_dir):
            print(f"  [SKIP] {name}")
            continue

        rms = load_rms(bearing_dir)
        psp, thr, rms_sm = detect_rms_baseline(
            rms, REFERENCE_RATIO, SIGMA_K, SMOOTH_WINDOW, CONSEC
        )
        pct   = psp / len(rms) * 100
        split = "Train" if is_train else "Test"
        cond  = name[7]

        results.append({
            "Bearing": name, "Condition": cond, "Split": split,
            "N_files": len(rms), "PSP": psp, "PSP_%": round(pct, 1),
            "Threshold": round(thr, 5),
        })
        all_rms[name] = rms
        all_psp[name] = psp
        all_thr[name] = thr
        all_sm[name]  = rms_sm

        print(f"  {name:12s} {split:5s} {len(rms):5d}  {psp:5d}  {pct:5.1f}%")

    # ── 彙總表格 ──
    df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)
    csv_path = os.path.join(SAVE_DIR, "psp_rms_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # ── 視覺化：所有軸承退化圖（5 欄網格）──
    n_bear = len(results)
    ncols  = 4
    nrows  = (n_bear + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.2),
                             squeeze=False)
    fig.suptitle(
        "Bearing Degradation — RMS Baseline PSP Detection (All PHM2012)",
        fontsize=13, fontweight="bold"
    )

    for idx, row in enumerate(results):
        name  = row["Bearing"]
        r     = all_rms[name]
        sm    = all_sm[name]
        psp   = all_psp[name]
        thr   = all_thr[name]
        T     = np.arange(len(r))
        ax    = axes[idx // ncols][idx % ncols]
        tag   = "T" if row["Split"] == "Train" else "V"
        color = "steelblue" if row["Split"] == "Train" else "darkorange"

        ax.plot(T, r,  color="lightgray",  linewidth=0.6, zorder=1)
        ax.plot(T, sm, color=color,        linewidth=1.2, zorder=2,
                label="Smoothed RMS")
        ax.axhline(thr, color="tomato", linestyle="--",
                   linewidth=0.9, zorder=3)
        ax.axvline(psp, color="tomato", linewidth=1.8, zorder=4,
                   label=f"PSP={psp} ({row['PSP_%']}%)")
        ax.axvspan(0, int(len(r) * REFERENCE_RATIO),
                   alpha=0.08, color=color, zorder=0)

        ax.set_title(f"{name}  [{tag}]  N={row['N_files']}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("Time Step (file index)", fontsize=7)
        ax.set_ylabel("RMS (g)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_bear, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(SAVE_DIR, "all_bearings_rms_psp.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"圖表: {out}")


if __name__ == "__main__":
    main()
