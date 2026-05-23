# DAST — Dual-Aspect Self-Attention Transformer for RUL / HI Prediction

用 PyTorch 實作的 DAST 模型，支援兩個資料集：

- **CMAPSS**（NASA 渦扇引擎，FD001–FD004）— 剩餘使用壽命（RUL）預測
- **PHM2012 / FEMTO PRONOSTIA**（滾動軸承）— 健康指標（HI）重建與 RUL 預測

模型同時對「感測器維度」與「時間步維度」做 self-attention，再經 decoder 輸出。所有超參數集中於 `config.json`，方便切換資料集與工況。

---

## 快速上手

### 1. 安裝環境

```bash
pip install -r requirements.txt
```

主要相依：`torch`、`numpy`、`pandas`、`scipy`、`scikit-learn`、`matplotlib`。
建議使用 GPU；程式會自動偵測 `torch.cuda.is_available()`。

### 2. 設定 `config.json`

切換資料集只需改一個欄位：

```json
{
  "dataset_type": "cmapss"
}
```

可選 `"cmapss"` 或 `"femto"`。兩個資料集的細節（路徑、`window_size`、`feature_len`、CMAPSS 的 `rul_max`、FEMTO 的 `condition`）都在各自區段。訓練、模型架構超參數則放在 `training` 與 `model` 區段。

### 3. 跑 CMAPSS 流程

```bash
# (1) 資料前處理：滑動視窗切片 + min-max 正規化
python data_process.py

# (2) 統計特徵抽取（產生最終 *_new.mat）
python "Statistical features process .py"

# (3) 訓練 + 測試
python DAST_test.py
```

產出：

- `train_dataset/F00X_<window_size>_{train,test}{X,Y}_new.mat` — 預處理後資料
- `model/dast_FD00X_best.pth` — 最佳權重
- `plots/` — 訓練曲線與 RUL 預測圖
- `紀錄/experiment_log.csv` — 每次實驗的超參數與指標

### 4. 跑 PHM2012 / FEMTO 流程

```bash
# (1) 從 .csv 振動訊號抽特徵、切視窗
python FEMTO_datapreprocess.py

# (2) 訓練 HI 重建模型
python DAST_FEMTO_train.py
```

工況對照（在 `DAST_FEMTO_train.py` 的 `CONDITION_MAP`）：

| Condition | 轉速     | 負載    | 訓練軸承          | 測試軸承                 |
| --------- | -------- | ------- | ----------------- | ------------------------ |
| 1         | 1800 rpm | 4000 N  | Bearing1_1, 1_2   | Bearing1_3 ~ 1_7         |
| 2         | 1650 rpm | 4200 N  | Bearing2_1, 2_2   | Bearing2_3 ~ 2_7         |
| 3         | 1500 rpm | 5000 N  | Bearing3_1, 3_2   | Bearing3_3               |

切換工況：改 `config.json` 中 `femto.condition` 為 `"1"` / `"2"` / `"3"`。

---

## 主要檔案

### 模型

| 檔案 | 用途 |
| ---- | ---- |
| `DAST_Network.py` | DAST 模型本體（Sensors encoder + Time-step encoder + Decoder） |
| `DAST_utils.py`   | Multi-head attention 模組與位置編碼等工具 |

### CMAPSS 流程

| 檔案 | 用途 |
| ---- | ---- |
| `data_process.py` | 載入原始 `.txt`、min-max 正規化、移除無用感測器、滑動視窗切片 |
| `Statistical features process .py` | 為每個視窗加入統計特徵 |
| `DAST_test.py` | 訓練 + 測試主腳本，產生實驗紀錄與繪圖 |
| `DAST_final.py` | 早期完整版本（保留參考） |

### PHM2012 / FEMTO 流程

| 檔案 | 用途 |
| ---- | ---- |
| `FEMTO_datapreprocess.py` | 振動訊號特徵抽取、降採樣、切視窗 |
| `DAST_FEMTO_train.py` | FEMTO 版訓練腳本（架構同 CMAPSS，僅資料載入與 loss 不同） |
| `detect_degradation_start.py` | 偵測軸承退化起點（Phase Start Point, PSP） |
| `detect_train_psp.py` | 用 PSP 標籤訓練輔助模型 |

### 其他

| 檔案 / 目錄 | 用途 |
| ----------- | ---- |
| `config.json` | 所有超參數與路徑集中控管 |
| `Cmapss_data/` | CMAPSS 原始資料（`train_FD00X.txt`、`test_FD00X.txt`、`RUL_FD00X.txt`） |
| `phm2012_dataset/` | PHM2012 原始振動訊號 |
| `train_dataset/`、`femto_dataset/` | 預處理後的 `.mat` 資料 |
| `model/`、`Model_trained/`、`train_model/` | 訓練好的權重檔 |
| `plots/`、`PHM-2008_prediction_result/` | 預測結果圖 |
| `紀錄/` | 實驗紀錄 CSV 與筆記 |
| `dast_FD00X_best.pth`、`dast_FEMTO_condX_best.pth` | 各資料集 / 工況的最佳權重 |
| `PHM2012_DAST_data_training_plan.md`、`PHM2012_HI_reconstruction_training_plan.md` | 訓練計劃文件 |

---

## 設定範例

### CMAPSS（FD003）

```json
"cmapss": {
  "dataset": "FD003",
  "data_path": "Cmapss_data",
  "output_path": "train_dataset",
  "window_size": 60,
  "rul_max": 125.0,
  "feature_len": 14
}
```

### FEMTO（Condition 3）

```json
"femto": {
  "femto_path": "phm2012_dataset",
  "output_path": "femto_dataset/",
  "window_size": 60,
  "feature_len": 16,
  "condition": "3"
}
```

### 訓練與模型

```json
"training": {
  "batch_size": 256,
  "epochs": 100,
  "learning_rate": 0.001,
  "model_save_path": "model",
  "seed": 42
},
"model": {
  "dec_seq_len": 10,
  "out_seq_len": 1,
  "dim_val": 64,
  "dim_attn": 64,
  "n_encoder_layers": 1,
  "n_decoder_layers": 1,
  "n_heads": 4,
  "dropout": 0.1,
  "use_full_features_for_time_encoder": false
}
```

---

## 使用預訓練權重

倉庫內已附上各資料集 / 工況的最佳權重，可直接載入做推論：

```python
import torch, json
from DAST_Network import DAST

with open("config.json") as f:
    cfg = json.load(f)
m = cfg["model"]

model = DAST(
    dim_val_s=m["dim_val"], dim_attn_s=m["dim_attn"],
    dim_val_t=m["dim_val"], dim_attn_t=m["dim_attn"],
    dim_val=m["dim_val"], dim_attn=m["dim_attn"],
    time_step=cfg["cmapss"]["window_size"],
    input_size=cfg["cmapss"]["feature_len"],
    dec_seq_len=m["dec_seq_len"], out_seq_len=m["out_seq_len"],
    n_encoder_layers=m["n_encoder_layers"],
    n_decoder_layers=m["n_decoder_layers"],
    n_heads=m["n_heads"], dropout=m["dropout"],
)
model.load_state_dict(torch.load("dast_FD003_best.pth", map_location="cpu"))
model.eval()
```

可用權重：`dast_FD001_best.pth` ~ `dast_FD004_best.pth`、`dast_FEMTO_cond1_best.pth` ~ `dast_FEMTO_cond3_best.pth`。

---

## 結果與輸出

每次訓練會產生：

- **權重**：`model/dast_<DATASET>_best.pth`（以最佳測試 RMSE / loss 為準）
- **訓練曲線圖**：`plots/<timestamp>_<DATASET>.png`（Train Loss、Test RMSE、預測 vs 真值散點圖等）
- **實驗紀錄**：`紀錄/experiment_log.csv`（每列一次實驗，包含超參數、最佳 epoch、指標）

---

## 資料集下載

- **CMAPSS**：NASA Prognostics Data Repository — Turbofan Engine Degradation
- **PHM2012 / FEMTO PRONOSTIA**：FEMTO-ST 軸承資料集（IEEE PHM 2012 Data Challenge）

下載後解壓到 `Cmapss_data/` 與 `phm2012_dataset/`。

---

## 參考

- DAST 原始論文：*Dual-Aspect Self-Attention Based on Transformer for Remaining Useful Life Prediction*（IEEE TIM, 2022）
- 訓練計劃細節見 `PHM2012_DAST_data_training_plan.md` 與 `PHM2012_HI_reconstruction_training_plan.md`
