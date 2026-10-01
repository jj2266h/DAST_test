# DAST 專案主要檔案與版本化範圍

本文件定義主要程式提交應包含的內容，以及資料集、模型與實驗產物的保存方式。

## 主要程式

| 路徑 | 用途 |
|---|---|
| `DAST_test.py` | C-MAPSS 訓練、逐 epoch 評估、checkpoint 與實驗紀錄 |
| `DAST_Network.py` | DAST 模型、encoder、decoder 與輸出層 |
| `DAST_utils.py` | Attention、multi-head attention 與 positional encoding |
| `DAST_visualization.py` | 訓練曲線、RUL 預測圖與 operating-condition 分群圖 |
| `data_process.py` | C-MAPSS 前處理、window 建立、RUL 正規化與特徵輸出 |
| `experiments/run_cmapss_sweep.py` | Gradient clipping、warmup 與 multi-seed sweep |

## 設定與依賴

| 路徑 | 用途 |
|---|---|
| `config.json` | 目前資料集、訓練與模型設定 |
| `requirements.txt` | Python 套件版本與執行依賴 |

## 資料集

| 路徑 | Git 政策 |
|---|---|
| `Cmapss_data/` | 原始 C-MAPSS 文字資料；目前由 Git 追蹤 |
| `cmapss_dataset/` | 由 `data_process.py` 產生的 `.mat`；不提交，可重建 |
| `train_dataset/` | 舊版處理資料；暫時保留，二進位更新不併入主要程式提交 |
| `phm2012_dataset/`、`femto_dataset/` | FEMTO/PHM 資料；不與 C-MAPSS 主程式提交混合 |

重新產生目前設定的 C-MAPSS 資料：

```powershell
python data_process.py --config config.json
```

## 模型

| 路徑 | Git 政策 |
|---|---|
| `model/` | `DAST_test.py` 產生的 run-specific checkpoints；不提交 |
| `Model_trained/`、`train_model/` | 舊版已追蹤模型；保留歷史，是否移除應另開提交處理 |
| `dast_*_best.pth` | 本機參考權重；由 `*.pth` 規則排除 |

需要公開的正式 checkpoint 建議使用 Git LFS 或 GitHub Release，並附上資料集、設定、seed、commit SHA 與評估協定，不與原始碼提交混合。

## 測試

| 路徑 | 覆蓋內容 |
|---|---|
| `tests/test_dast_network.py` | Decoder mask/dropout、輸出層、多 decoder layer、輸出形狀 |
| `tests/test_dast_visualization.py` | 六張目前圖表的輸出 |
| `tests/test_experiment_log.py` | CSV 欄位擴充與相容性 |
| `tests/test_run_cmapss_sweep.py` | Sweep 設定與工作產生 |

## 執行順序

```powershell
python -m pip install -r requirements.txt
python data_process.py --config config.json
python DAST_test.py --config config.json --run-label <run-label>
```

## 目前研究限制

- `condition_scaling_enabled` 僅在 FD002/FD004 啟用 operating-condition scaling；FD001/FD003 使用全域 MinMax scaling。
- `DAST_test.py` 每個 epoch 都評估 test set，並以最低 Test RMSE 選 checkpoint；相關 best 結果屬 test-selected exploratory evidence，不是獨立 final-test 結果。
- `紀錄/`、`plots/`、`experiments/results/` 與 `server_results/` 是產物或結果鏡像，不納入主要程式提交。

## 主要程式提交清單

主要提交應只包含：

- `.gitignore`
- `PROJECT_STRUCTURE.md`
- `DAST_test.py`
- `DAST_Network.py`
- `DAST_utils.py`
- `DAST_visualization.py`
- `data_process.py`
- `config.json`
- `requirements.txt`
- `experiments/run_cmapss_sweep.py`
- `tests/test_dast_network.py`
- `tests/test_dast_visualization.py`
- `tests/test_experiment_log.py`
- `tests/test_run_cmapss_sweep.py`

不要使用 `git add .`；應逐檔加入上述清單，避免把資料、模型、結果鏡像或投稿文件帶入同一提交。
