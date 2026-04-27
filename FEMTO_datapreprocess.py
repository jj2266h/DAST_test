# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import fsolve
from sklearn import preprocessing


class FEMTOPreprocessor:
    """
    PHM2012 (PRONOSTIA/FEMTO) 資料前處理器

    特徵  : 15 時域 + 1 頻域 = 16 維，單通道（水平振動 col[4]）
    標籤  : 指數型 HI（Health Index），0 = 健康初始, 1 = 失效
    輸出  : X (M, W, 16)  Y (M,)  → 以 .mat 格式儲存
    """

    def __init__(
        self,
        folders: list,          # [(軸承名稱, is_train: bool, psp_override: int or None)]
        window_size: int,
        femto_path: str,
        output_path: str,
        is_save: bool = True,
    ):
        """
        Parameters
        ----------
        folders     : 軸承清單，psp_override = 手動指定 PSP 索引。
                      None 時自動以 3σ 準則從 RMS 序列偵測 PSP。
        window_size : 滑動窗口大小（單位：10 秒快照數）。
        femto_path  : Learning_set / Full_Test_Set / Test_set 上層目錄。
        output_path : .mat 輸出目錄。
        """
        self.folders     = folders
        self.window_size = window_size
        self.FEMTO       = femto_path
        self.POST_PROCESS = output_path
        self.is_save     = is_save
        self.scaler      = None   # MinMaxScaler，在 process_all() 中 fit

    # ────────────────────────────────────────────
    # HI 建構（指數型，純數學定義）
    # ────────────────────────────────────────────

    @staticmethod
    def _hi_formula(t, a, tau):
        return 1.0 + np.exp(a) - np.exp(t * tau + a)

    def _reconstruct_hi(self, n_files: int, convergence) -> np.ndarray:
        """
        使用指數模型建立 HI 序列。

        HI(t) = 1 + exp(a) - exp(t * tau + a),  a = 1（固定）
        tau 由 fsolve 求解，使 HI(convergence) = 0。
        convergence 為 None 時自動使用 n_files - 1（整段壽命）。

        正規化後輸出：HI(0) ≈ 1（健康），HI(n_files-1) ≈ 0（失效）。
        """
        if not convergence:
            convergence = n_files * 0.035

        a = 1.0
        tau = fsolve(lambda t: self._hi_formula(convergence, a, t), x0=0.0)[0]

        t_arr = np.arange(n_files, dtype=float)
        hi_raw = self._hi_formula(t_arr, a, tau)

        # min-max → [0, 1]，1=健康（初始），0=失效（末尾），與 preprocess.py 一致
        hi_min, hi_max = hi_raw.min(), hi_raw.max()
        hi_norm = (hi_raw - hi_min) / (hi_max - hi_min)
        return hi_norm.astype(np.float32)

    # ────────────────────────────────────────────
    # 特徵提取（15 TD + 1 FD = 16 維）
    # ────────────────────────────────────────────

    def _get_feature(self, acc_path: str) -> list:
        """讀取單一 acc_XXXXX.csv → 16D 特徵向量。"""
        x = pd.read_csv(acc_path, header=None, sep='[,;]',
                        usecols=[4], engine='python')
        return self._extract_feature(x)

    def _extract_feature(self, x: pd.DataFrame) -> list:
        """
        p1~p16: 15 個時域統計特徵
        p17   : 1 個頻域特徵（FFT 振幅均值）
        """
        LEN = len(x)
        x_abs         = x.abs()
        x_avg         = x.mean()
        x_sub_mean    = x.sub(x_avg, axis=1)
        mean_sq_sum   = (x_sub_mean ** 2).sum()

        p1  = x.max()                                   # max
        p2  = x.min()                                   # min
        p3  = x_abs.max()                               # peak (abs max)
        p4  = p1 - p2                                   # peak-to-peak
        p5  = x_abs.sum() / LEN                         # mean absolute
        p6  = (x_abs.sum() ** 0.5 / LEN) * 2            # square mean root ×2
        p7  = mean_sq_sum / (LEN - 1)                   # variance
        p8  = (mean_sq_sum / LEN) ** 0.5                # std dev
        p9  = ((x ** 2).sum() / LEN) ** 0.5             # RMS
        p11 = (LEN * p9) / x_abs.sum()                  # shape factor
        p12 = p9 / p5                                   # RMS / mean_abs
        p13 = p3 / p9                                   # crest factor
        p14 = p3 / p5                                   # impulse factor
        p15 = p3 / p6                                   # margin factor
        p16 = p3 / (p9 ** 2)                            # modified factor

        fft_amp = np.abs(np.fft.fft(x.to_numpy(), axis=0))
        p17 = float(np.sum(fft_amp) / len(fft_amp))    # spectral mean amplitude

        return [
            float(p1.iloc[0]),  float(p2.iloc[0]),  float(p3.iloc[0]),  float(p4.iloc[0]),
            float(p5.iloc[0]),  float(p6.iloc[0]),  float(p7.iloc[0]),  float(p8.iloc[0]),
            float(p9.iloc[0]),  float(p11.iloc[0]), float(p12.iloc[0]),
            float(p13.iloc[0]), float(p14.iloc[0]), float(p15.iloc[0]),
            float(p16.iloc[0]), p17,
        ]

    # ────────────────────────────────────────────
    # 滑動窗口 & 統計附加行
    # ────────────────────────────────────────────

    def _slide_x_window(self, features: np.ndarray) -> np.ndarray:
        """(N, 16) → (N-W, W, 16)"""
        return np.stack(
            [features[i: i + self.window_size]
             for i in range(len(features) - self.window_size)],
            axis=0,
        )

    def _slide_y_window(self, hi: np.ndarray) -> np.ndarray:
        """(N,) → (N-W,)，每個窗口取最後一步的 HI 值作為標籤。"""
        return np.array(
            [hi[i + self.window_size - 1]
             for i in range(len(hi) - self.window_size)],
            dtype=np.float32,
        )

    # ────────────────────────────────────────────
    # 主流程
    # ────────────────────────────────────────────

    def process_all(self):
        os.makedirs(self.POST_PROCESS, exist_ok=True)

        # ── Step 1：用訓練集 fit MinMaxScaler ──
        print("Step 1: Fitting scaler on training bearings...")
        train_feats_all = []
        for (folder, is_train, convergence) in self.folders:
            if not is_train:
                continue
            bearing_dir = os.path.join(self.FEMTO, folder)
            accs = sorted(f for f in os.listdir(bearing_dir) if f.startswith('acc'))
            for acc in accs:
                train_feats_all.append(self._get_feature(os.path.join(bearing_dir, acc)))
        self.scaler = preprocessing.MinMaxScaler().fit(train_feats_all)
        print(f"  Scaler fit on {len(train_feats_all)} samples from training bearings.\n")

        # ── Step 2：逐軸承處理 ──
        for (folder, is_train, convergence) in self.folders:
            bearing_dir = os.path.join(self.FEMTO, folder)
            accs = sorted(f for f in os.listdir(bearing_dir) if f.startswith('acc'))
            n_files = len(accs)
            print(f"{folder}: {n_files} files  is_train={is_train}  convergence={convergence or n_files*0.035}")

            # 特徵 → scale
            raw = np.array(
                [self._get_feature(os.path.join(bearing_dir, acc)) for acc in accs],
                dtype=np.float32,
            )
            raw = self.scaler.transform(raw)              # (N, 16)

            # HI 標籤
            hi = self._reconstruct_hi(n_files, convergence)  # (N,)

            # 滑動窗口
            X = self._slide_x_window(raw)    # (M, W, 16)
            Y = self._slide_y_window(hi)     # (M,)

            print(f"  X: {X.shape}  Y: {Y.shape}  HI∈[{Y.min():.3f}, {Y.max():.3f}]")

            if self.is_save:
                prefix = 'train' if is_train else 'test'
                name   = os.path.basename(folder)
                sio.savemat(
                    os.path.join(self.POST_PROCESS, f'{name}_X.mat'),
                    {f'PHM_{prefix}X': X},
                )
                sio.savemat(
                    os.path.join(self.POST_PROCESS, f'{name}_Y.mat'),
                    {f'PHM_{prefix}Y': Y},
                )
                print(f"  → Saved {name}_X.mat, {name}_Y.mat")
        print("\n✓ 前處理完成。")

    def plot_hi_overview(self, save_path: str = "plots/hi_overview.png"):
        """
        讀取已儲存的 Y.mat，繪製訓練集與測試集 HI 曲線（各軸承一條線）。
        X 軸：時間（秒），每個 acc 檔案 = 10 秒。
        Y 軸：Health Index（1=健康，0=失效）。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            print(f"Matplotlib 無法載入，略過 HI 概覽圖: {e}")
            return

        train_curves, test_curves = [], []
        for (folder, is_train, _) in self.folders:
            name   = os.path.basename(folder)
            prefix = "train" if is_train else "test"
            y_path = os.path.join(self.POST_PROCESS, f"{name}_Y.mat")
            if not os.path.exists(y_path):
                continue
            y = sio.loadmat(y_path)[f"PHM_{prefix}Y"].flatten()
            t = np.arange(len(y)) * 10   # 每筆 = 10 秒
            if is_train:
                train_curves.append((name, t, y))
            else:
                test_curves.append((name, t, y))

        fig, (ax_tr, ax_te) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle("PHM2012 (FEMTO) — Health Index Overview", fontsize=13, fontweight="bold")

        cmap_tr = plt.cm.tab10
        cmap_te = plt.cm.tab20

        for i, (name, t, y) in enumerate(train_curves):
            ax_tr.plot(t, y, linewidth=1.2, label=name, color=cmap_tr(i % 10))
        ax_tr.set_title("Training set")
        ax_tr.set_xlabel("Time (s)")
        ax_tr.set_ylabel("Health Index HI")
        ax_tr.set_ylim(-0.05, 1.05)
        ax_tr.legend(fontsize=8)
        ax_tr.grid(True, linestyle="--", alpha=0.4)

        for i, (name, t, y) in enumerate(test_curves):
            ax_te.plot(t, y, linewidth=1.0, label=name, color=cmap_te(i % 20))
        ax_te.set_title("Testing data")
        ax_te.set_xlabel("Time (s)")
        ax_te.set_ylim(-0.05, 1.05)
        ax_te.legend(fontsize=7)
        ax_te.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"HI 概覽圖已儲存至: {save_path}")


# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)["femto"]

    FEMTO_PATH   = cfg["femto_path"]
    OUTPUT_PATH  = cfg["output_path"]
    WINDOW_SIZE  = cfg["window_size"]

    # convergence: HI 到達失效（=0 in raw curve）的 file index。
    # None → 自動使用各軸承的總壽命（整段都算入退化區間）。
    # 也可手動填入具體數值來調整 HI 曲線形狀。
    DATASET = [
         ('Learning_set/Bearing1_1', True,  100),  # 訓練集 (Condition 1)
        ('Learning_set/Bearing1_2', True,  30),   # 訓練集 (Condition 1)
        ('Learning_set/Bearing2_1', True,  30),   # 訓練集 (Condition 2)
        ('Learning_set/Bearing2_2', True,  30),   # 訓練集 (Condition 2)
        ('Learning_set/Bearing3_1', True,  17),   # 訓練集 (Condition 3)
        ('Learning_set/Bearing3_2', True,  50),   # 訓練集 (Condition 3)
       
        ('Full_Test_Set/Bearing1_3', False, 80),   # 測試集
        ('Full_Test_Set/Bearing1_4', False, 50),   # 測試集
        ('Full_Test_Set/Bearing1_5', False, 90),   # 測試集
        ('Full_Test_Set/Bearing1_6', False, 90),   # 測試集
        ('Full_Test_Set/Bearing1_7', False, 80),   # 測試集
        ('Full_Test_Set/Bearing2_3', False, 70),   # 測試集
        ('Full_Test_Set/Bearing2_4', False, 27),   # 測試集
        ('Full_Test_Set/Bearing2_5', False, 84),   # 測試集
        ('Full_Test_Set/Bearing2_6', False, 25),   # 測試集
        ('Full_Test_Set/Bearing2_7', False, 8),    # 測試集
        ('Full_Test_Set/Bearing3_3', False, 15),   # 測試集
    ]

    preprocessor = FEMTOPreprocessor(
        folders=DATASET,
        window_size=WINDOW_SIZE,
        femto_path=FEMTO_PATH,
        output_path=OUTPUT_PATH,
        is_save=True,
    )
    preprocessor.process_all()
    preprocessor.plot_hi_overview(save_path="plots/hi_overview.png")
