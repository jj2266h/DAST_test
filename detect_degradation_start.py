# -*- coding: utf-8 -*-
"""
PHM2012 軸承退化起始點偵測
方法：
  1. 3σ 準則 (3-Sigma Method)
  2. RMS 趨勢觀察 (rolling baseline deviation)
  3. 隱馬可夫模型 (HMM, 2-state Gaussian)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# ── 設定 ──────────────────────────────────────────────────────
BEARING_DIR = r"phm2012_dataset/Learning_set/Bearing1_1"
BEARING_NAME = "Bearing1_1"
REFERENCE_RATIO = 0.15      # 前 15% 作為健康基準期
SMOOTH_WINDOW   = 60        # RMS 平滑窗口（單位：檔案數）
SIGMA_K         = 3         # 3σ 倍數
CONSEC_EXCEED   = 5         # 連續超標次數才確認為異常點
HMM_STATES      = 2         # HMM 狀態數（健康 / 退化）
SAVE_FIG        = True


# ── 特徵計算 ──────────────────────────────────────────────────

def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))

def compute_kurtosis(signal: np.ndarray) -> float:
    mu = signal.mean()
    std = signal.std()
    if std == 0:
        return 0.0
    return float(np.mean(((signal - mu) / std) ** 4))

def load_features(bearing_dir: str):
    accs = sorted(f for f in os.listdir(bearing_dir) if f.startswith('acc'))
    rms_list, kurt_list = [], []
    for acc in accs:
        raw = pd.read_csv(
            os.path.join(bearing_dir, acc),
            header=None, sep='[,;]', usecols=[4], engine='python'
        ).values.flatten().astype(np.float64)
        rms_list.append(compute_rms(raw))
        kurt_list.append(compute_kurtosis(raw))
    return np.array(rms_list), np.array(kurt_list)


# ── 方法 1：3σ 準則 ───────────────────────────────────────────

def detect_3sigma(rms: np.ndarray, ref_ratio=0.15, k=3, consec=5):
    """
    以前 ref_ratio 段作為健康基準，找第一個連續 consec 點超過 μ+k·σ 的位置。
    回傳：(psp_index, threshold, mu, sigma)
    """
    n_ref = max(10, int(len(rms) * ref_ratio))
    ref   = rms[:n_ref]
    mu, sigma = ref.mean(), ref.std()
    threshold = mu + k * sigma

    count = 0
    for i in range(n_ref, len(rms)):
        if rms[i] > threshold:
            count += 1
            if count >= consec:
                return i - consec + 1, threshold, mu, sigma
        else:
            count = 0
    return len(rms) - 1, threshold, mu, sigma


# ── 方法 2：RMS 滾動基線偏差 ──────────────────────────────────

def detect_rolling_baseline(rms: np.ndarray, ref_ratio=0.15, k=3,
                             smooth_w=20, consec=5):
    """
    先用平滑 RMS，再比對健康基準，找偏差超過 k·σ 的起始點。
    """
    smoothed = pd.Series(rms).rolling(smooth_w, center=True,
                                      min_periods=1).mean().values
    n_ref = max(10, int(len(smoothed) * ref_ratio))
    ref   = smoothed[:n_ref]
    mu, sigma = ref.mean(), ref.std()
    threshold = mu + k * sigma

    count = 0
    for i in range(n_ref, len(smoothed)):
        if smoothed[i] > threshold:
            count += 1
            if count >= consec:
                return i - consec + 1, threshold, mu, sigma, smoothed
        else:
            count = 0
    return len(smoothed) - 1, threshold, mu, sigma, smoothed


# ── 方法 3：隱馬可夫模型 (HMM) ────────────────────────────────

def detect_hmm(rms: np.ndarray, n_states=2):
    """
    訓練 2 狀態 Gaussian HMM，找第一個從健康態 → 退化態的轉換。
    回傳：(psp_index, state_seq, healthy_id, degraded_id)
    """
    X = rms.reshape(-1, 1)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X_sc)
    states = model.predict(X_sc)

    # 低均值 → 健康態
    state_means = [X_sc[states == s].mean() if (states == s).any() else 0
                   for s in range(n_states)]
    healthy_id  = int(np.argmin(state_means))
    degraded_id = int(np.argmax(state_means))

    psp = len(rms) - 1
    for i in range(1, len(states)):
        if states[i - 1] == healthy_id and states[i] == degraded_id:
            psp = i
            break

    return psp, states, healthy_id, degraded_id


# ── 視覺化 ────────────────────────────────────────────────────

def plot_results(rms, kurt, res_3s, res_rb, res_hmm, bearing_name):
    psp_3s,  thr_3s,  mu_3s,  sig_3s              = res_3s
    psp_rb,  thr_rb,  mu_rb,  sig_rb,  rms_smooth  = res_rb
    psp_hmm, states,  hid,    did                   = res_hmm

    T = np.arange(len(rms))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Degradation Start Detection — {bearing_name}",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    color_3s  = "steelblue"
    color_rb  = "darkorange"
    color_hmm = "mediumseagreen"
    color_raw = "lightgray"

    # ── Panel 1: 原始 RMS + 三個偵測點 ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T, rms, color=color_raw, linewidth=0.6, label="Raw RMS")
    ax1.plot(T, rms_smooth, color="dimgray", linewidth=1.2, label="Smoothed RMS")
    ax1.axvline(psp_3s,  color=color_3s,  linestyle="--", linewidth=1.5,
                label=f"3σ PSP = t{psp_3s}")
    ax1.axvline(psp_rb,  color=color_rb,  linestyle="-.", linewidth=1.5,
                label=f"RMS Baseline PSP = t{psp_rb}")
    ax1.axvline(psp_hmm, color=color_hmm, linestyle=":",  linewidth=2.0,
                label=f"HMM PSP = t{psp_hmm}")
    ax1.set_title("RMS Over Bearing Lifetime with Detected PSP")
    ax1.set_xlabel("Time Step (file index)")
    ax1.set_ylabel("RMS (g)")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: 3σ 準則細節 ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(T, rms, color=color_raw, linewidth=0.6)
    ax2.axhline(thr_3s, color=color_3s, linestyle="--",
                label=f"Threshold (μ+3σ) = {thr_3s:.4f}")
    ax2.axhline(mu_3s,  color="navy", linestyle=":", alpha=0.6,
                label=f"Healthy mean = {mu_3s:.4f}")
    ax2.axvspan(0, int(len(rms) * REFERENCE_RATIO),
                alpha=0.12, color=color_3s, label="Reference period")
    ax2.axvline(psp_3s, color=color_3s, linewidth=1.5,
                label=f"PSP = t{psp_3s}")
    ax2.set_title("Method 1: 3σ Criterion")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("RMS")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 3: RMS 滾動基線 ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(T, rms, color=color_raw, linewidth=0.6, label="Raw RMS")
    ax3.plot(T, rms_smooth, color=color_rb, linewidth=1.2, label="Smoothed RMS")
    ax3.axhline(thr_rb, color="saddlebrown", linestyle="--",
                label=f"Threshold = {thr_rb:.4f}")
    ax3.axvspan(0, int(len(rms) * REFERENCE_RATIO),
                alpha=0.12, color=color_rb, label="Reference period")
    ax3.axvline(psp_rb, color="saddlebrown", linewidth=1.5,
                label=f"PSP = t{psp_rb}")
    ax3.set_title(f"Method 2: Smoothed RMS Baseline (window={SMOOTH_WINDOW})")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("RMS")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 4: HMM 狀態序列 ──
    ax4 = fig.add_subplot(gs[2, 0])
    state_colors = np.where(states == hid, 0.3, 0.85)
    ax4.scatter(T, rms, c=state_colors, cmap="RdYlGn_r", s=4, alpha=0.7)
    ax4.axvline(psp_hmm, color=color_hmm, linewidth=1.5,
                label=f"HMM PSP = t{psp_hmm}")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax4, label="State (0=Healthy / 1=Degraded)")
    ax4.set_title("Method 3: HMM State Sequence")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("RMS")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 5: Kurtosis ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(T, kurt, color="mediumpurple", linewidth=0.8, label="Kurtosis")
    ax5.axvline(psp_3s,  color=color_3s,  linestyle="--", linewidth=1.2,
                label=f"3σ PSP = t{psp_3s}")
    ax5.axvline(psp_rb,  color=color_rb,  linestyle="-.", linewidth=1.2,
                label=f"RMS Baseline PSP = t{psp_rb}")
    ax5.axvline(psp_hmm, color=color_hmm, linestyle=":",  linewidth=1.5,
                label=f"HMM PSP = t{psp_hmm}")
    ax5.set_title("Kurtosis (reference comparison)")
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Kurtosis")
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.4)

    if SAVE_FIG:
        os.makedirs("plots", exist_ok=True)
        path = f"plots/psp_detection_{bearing_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"圖表儲存至: {path}")
    plt.show()


# ── 主程式 ────────────────────────────────────────────────────

def main():
    print(f"載入 {BEARING_NAME} 特徵...")
    rms, kurt = load_features(BEARING_DIR)
    print(f"  共 {len(rms)} 個時間步")

    print("\n[Method 1] 3σ 準則...")
    res_3s = detect_3sigma(rms, REFERENCE_RATIO, SIGMA_K, CONSEC_EXCEED)
    print(f"  健康基準: μ={res_3s[2]:.5f}, σ={res_3s[3]:.5f}")
    print(f"  閾值: {res_3s[1]:.5f}")
    print(f"  PSP = t{res_3s[0]}  ({res_3s[0]/len(rms)*100:.1f}% 壽命)")

    print("\n[Method 2] RMS 滾動基線偏差...")
    res_rb = detect_rolling_baseline(rms, REFERENCE_RATIO, SIGMA_K,
                                     SMOOTH_WINDOW, CONSEC_EXCEED)
    print(f"  平滑閾值: {res_rb[1]:.5f}")
    print(f"  PSP = t{res_rb[0]}  ({res_rb[0]/len(rms)*100:.1f}% 壽命)")

    print("\n[Method 3] HMM...")
    res_hmm = detect_hmm(rms, HMM_STATES)
    print(f"  健康態 ID={res_hmm[2]}, 退化態 ID={res_hmm[3]}")
    print(f"  PSP = t{res_hmm[0]}  ({res_hmm[0]/len(rms)*100:.1f}% 壽命)")

    print("\n比較摘要:")
    print(f"  3σ PSP       : t{res_3s[0]:4d} / {len(rms)} ({res_3s[0]/len(rms)*100:.1f}%)")
    print(f"  RMS 基線 PSP : t{res_rb[0]:4d} / {len(rms)} ({res_rb[0]/len(rms)*100:.1f}%)")
    print(f"  HMM PSP      : t{res_hmm[0]:4d} / {len(rms)} ({res_hmm[0]/len(rms)*100:.1f}%)")

    plot_results(rms, kurt, res_3s, res_rb, res_hmm, BEARING_NAME)


if __name__ == "__main__":
    main()
