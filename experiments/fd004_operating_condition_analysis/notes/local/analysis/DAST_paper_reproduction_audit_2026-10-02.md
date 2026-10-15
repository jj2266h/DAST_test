# DAST 論文還原度稽核（2026-10-02）

## 1. 稽核範圍與判定原則

- 論文：Zhang, Song, and Li, *Dual-Aspect Self-Attention Based on Transformer for Remaining Useful Life Prediction*, IEEE TIM 71 (2022)，特別核對 Section III、Section IV-A/B、Table I、Table II。
- 目前模型與流程：`DAST_Network.py`、`DAST_utils.py`、`data_process.py`、`DAST_test.py`、`config.json`。
- 實際四資料集實驗設定：`experiments/run_once_configs/FD001.json` 至 `FD004.json`。
- 使用者提供的原始訓練檔：`C:/Users/hjs92/Downloads/DAST_main.py`。
- 論文正文是論文設定的最高證據；正文未揭露的細節標為「未揭露」，不以程式值冒充論文值。
- 「符合」表示論文明確支持；「部分符合」表示主要機制一致但精確操作未揭露或有差異；「不符合」表示有明確衝突。

## 2. 摘要結論

| 評估面向 | 權重 | 實際四資料集 run 設定得分 | 判定 |
|---|---:|---:|---|
| 模型架構與機制 | 40 | 36 | 主架構高度一致；fusion、decoder 輸入與 FFN activation 有未揭露細節 |
| 資料前處理 | 30 | 27 | 論文明示項目大致一致；短序列插值、統計特徵縮放與追加位置未揭露 |
| 訓練參數、Optimizer、Loss | 20 | 20 | 實際 run 的論文明示值全部一致 |
| 評估與重複實驗流程 | 10 | 3 | 只有單一 seed，且以 Test RMSE 選最佳 checkpoint |
| **總計** | **100** | **86** | **完整實驗流程尚未達 90%** |

核心三項（模型、前處理、訓練）為 **83/90 = 92.2%**；因此核心方法可說約 92% 還原，但完整、可公平重現論文報告的實驗流程只有約 86%。此外，現在工作樹的 `DAST_test.py` 含未解決 Git conflict marker，`py_compile` 失敗；在衝突解決前，當前 checkout 不具可執行完成態，不能正式認證「整體還原達 90%」。

## 3. 資料前處理對照

| 項目 | 論文 | 實際四資料集 run／目前程式 | 狀態 | 影響與證據 |
|---|---|---|---|---|
| C-MAPSS 子資料集 | FD001–FD004 | FD001–FD004 均已處理與訓練 | 符合 | `run_once_configs` 四份設定 |
| 感測器選擇 | 移除 s1、s5、s6、s10、s16、s18、s19，保留 14 sensors | `data_process.py:119-123` 移除對應欄位，輸入 14 維 | 符合 | 操作條件欄也移除，不餵入模型 |
| RUL 上限 | `RUL_max = 125` | `data_process.py:146-151,182-185`；設定 125 | 符合 | train/test label 均 cap 125 |
| Label 縮放 | 論文未明示 | train/test label 除以 125 | 未揭露但合理 | `data_process.py:206-209` |
| 感測器正規化 | Min-Max | 實際 run：以 train fit MinMax，再 transform test | 符合且較安全 | `data_process.py:112-115`；run configs 關閉 condition scaling |
| 主 `config.json` 的 FD002/FD004 正規化 | 論文只有 Min-Max | `condition_scaling_enabled=true`，會改用 condition-wise z-score | **不符合** | 主設定與實際 run 設定不同；若直接用主設定重做 FD002/4，已不是論文流程 |
| Sliding window | stride=1；最後一點 RUL 作為 window label | stride=1；label 為 window 最後點 RUL | 符合 | `data_process.py:139-152` |
| Window size | FD001/3=40；FD002/4=60 | 四份 run configs 分別 40/60 | 符合 | 明確對齊 Section IV-B-3 |
| Test 輸入 | 每台 test engine 預測最後時間點 | 每台 engine 取最後一個 window | 符合 | `data_process.py:155-185` |
| Test 長度不足 window | 論文未交代 | spline 插值到完整 window | 未揭露 | `data_process.py:162-174`，可能影響短軌跡 |
| 統計特徵 | 加入 mean 與 regression coefficient estimates | 每個 sensor 計算 slope 與 mean | 符合 | `data_process.py:220-249` |
| 統計特徵正規化 | 論文未交代 | slope、mean 分別以 train fit MinMax | 部分符合 | train-only 無洩漏，但無法證明與論文完全相同 |
| 統計特徵排列 | 論文稱「add to sequence」，未交代軸與順序 | 沿 time axis 追加兩列，`T -> T+2` | 部分符合 | `data_process.py:251-262`；FD001/3 為 42，FD002/4 為 62 |
| Train/validation split | 論文未報告 validation protocol | 本次無 validation | 無法確認 | 與論文還原不一定衝突，但不足以產生獨立 Test 結果 |

## 4. 模型參數與運算機制對照

| 層級／Op | 論文 | 目前實作 | 狀態 | 備註 |
|---|---|---|---|---|
| Input embedding | FC，hidden=64，Linear | sensor/time/decoder 各自 Linear 到 64 | 符合 | `DAST_Network.py:104-108` |
| Positional encoding | sine/cosine | sine/cosine | 符合 | `DAST_utils.py:166-186` |
| Sensor encoder | 沿 sensor axis 的 multihead self-attention | 先轉置成 sensor tokens，再做 self-attention | 符合 | `DAST_Network.py:119,122` |
| Time-step encoder | 沿 time axis 的 multihead self-attention | 保持 time tokens 做 self-attention | 符合 | `DAST_Network.py:123` |
| 雙編碼器關係 | 平行、避免互相干擾 | 兩分支由同一輸入平行計算 | 符合 | 無 RNN/CNN |
| Encoder blocks | sensor N=2；time N=2 | 兩者皆 2 | 符合 | config `n_encoder_layers=2` |
| Attention heads | H=4 | H=4 | 符合 | 每個 head 有獨立 Q/K/V |
| Q/K/V attention | scaled dot-product + softmax | `QK^T / sqrt(dim)` + softmax + V | 符合 | `DAST_utils.py:8-37` |
| Multihead aggregation | Concat heads 後乘 projection | stack/flatten heads 後 Linear projection | 符合 | `DAST_utils.py:78-132` |
| Encoder sublayers | attention、FFN、residual、LayerNorm | attention、兩層 FC、ELU、residual、LayerNorm | 部分符合 | 論文未明示 encoder FFN activation；目前為 ELU |
| Dropout | 每個 encoder/decoder layer，rate=0.2 | attention 與 FFN residual path 均 0.2 | 符合 | config `dropout=0.2` |
| Feature fusion | `Concat(Fs,Ft) Wf` | token concat、LayerNorm、最後維 Linear 64→64 | 部分符合 | 主要概念一致；論文的 Wf 維度敘述與精確乘法不夠清楚 |
| Decoder blocks | N=1 | N=1 | 符合 | config `n_decoder_layers=1` |
| Decoder self-attention | masked multihead self-attention | 下三角 causal mask | 符合 | `DAST_Network.py:56-65` |
| Decoder cross-attention | decoder query；fusion output 作 K/V | query 來自 decoder，K/V 來自 fused tokens | 符合 | `DAST_utils.py:47-50` |
| Decoder input length | 論文未報告 | `dec_seq_len=4` | 原始程式支持、論文未揭露 | 使用者提供的 `DAST_main.py:72` 也是 4 |
| Decoder input內容 | 論文描述先前時間點／rolling prediction | 取輸入最後 4 rows | 部分符合 | 因統計特徵追加在最後，實際為「最後 2 cycles + slope + mean」，不是純四個時間點 |
| Output head | Flatten → FC(64, ReLU) → output(1, Linear) | Flatten → FC64 → ReLU → Linear1 | 符合 | `DAST_Network.py:109-111,142` |
| Output sequence length | 1 | 1 | 符合 | 單一 RUL |
| Dmodel / Dattn | Table II 明示 hidden=64；attention 內部維度未另列 | dim_val=64，dim_attn=64 | 符合／部分未揭露 | dim_val 明確；dim_attn 的精確值正文未獨立揭露 |
| Trainable parameters | 論文未報告 | W40/T+2：461,697；W60/T+2：462,977 | 不計分 | 已實際 instantiate 與 forward 驗證 |
| Forward output shape | 單一 RUL | batch 2 時 `(2,1)` | 符合 | W40/W60 smoke forward 均成功 |

## 5. 訓練參數、Optimizer、Loss 對照

| 項目 | 論文 | 實際 run／目前設定 | 狀態 | 備註 |
|---|---:|---:|---|---|
| Optimizer（Op） | Rectified Adam (RAdam) | `torch.optim.RAdam` | 符合 | `DAST_test.py:299` |
| RAdam betas | 未揭露 | `(0.9, 0.999)`（PyTorch default） | 未揭露 | 未在建構子覆寫 |
| RAdam epsilon | 未揭露 | `1e-8`（PyTorch default） | 未揭露 | 未在建構子覆寫 |
| Weight decay | 未揭露 | `0`（PyTorch default） | 未揭露 | 無 L2/decoupled weight decay |
| RAdam 其他 flags | 未揭露 | `decoupled_weight_decay=False`、`maximize=False`、`capturable=False`、`differentiable=False`、`foreach=None` | 未揭露 | PyTorch 2.11.0+cpu defaults |
| Epochs | 100 | 100 | 符合 | 四份 run configs |
| Batch size | 256 | 256 | 符合 | 四份 run configs |
| Learning rate | 0.001 | 0.001 | 符合 | 無 scheduler 時保持固定 |
| Loss | RMSE | `sqrt(mean((yhat-y)^2))` | 符合 | `DAST_test.py:65-66` |
| Loss 單位 | 論文未交代 label 是否先除 125 | normalized RUL 空間計算 RMSE | 未完全可證 | 原始 `DAST_main.py` 先乘 125，再算 RMSE；梯度尺度不同 |
| Dropout | 0.2 | 0.2 | 符合 | encoder/decoder residual paths |
| Gradient clipping | 未提及 | 關閉 | 不衝突 | 本次正式 run 已關閉 |
| LR warmup | 未提及 | 關閉 | 不衝突 | 固定 LR 更貼近正文 |
| LR scheduler | 未提及 | 無 | 不衝突 | warmup 關閉時 `scheduler=None` |
| Shuffle | 未提及 | train loader shuffle=True | 未揭露 | 一般合理做法 |
| Seed | 未公布 | 42 | 未揭露 | 單次 deterministic run |
| Tensor dtype | 未揭露 | float32 | 未揭露 | 模型與輸入皆 float32 |
| 重複次數 | 10 次並報平均 | 1 次（seed 42） | **不符合** | 不能把單次最佳值與論文十次平均作公平等價比較 |
| 硬體 | Windows 10、64 GB、Intel 9900K CPU | conda `dast`、PyTorch CPU | 部分符合 | 都是 CPU，但硬體與軟體版本不完全相同 |

## 6. 評估與 checkpoint 流程

| 項目 | 論文 | 目前實作 | 狀態 | 影響 |
|---|---|---|---|---|
| Metrics | RMSE、NASA Score | RMSE、MAE、NASA Score | 符合 | 多報 MAE 不影響還原 |
| NASA Score | 高估使用 /10；低估使用 /13 | 公式一致 | 符合 | `DAST_test.py:74-77` |
| 負值 prediction clamp | 論文未交代 | 目前不 clamp；原始 `DAST_main.py` clamp 到 0 | 與原始程式不一致 | 可能改變 RMSE/Score |
| 每 epoch 使用 Test | 論文未清楚交代 | 每 epoch 評估官方 Test | 方法學風險 | `DAST_test.py:344-360` |
| Best checkpoint | 論文未清楚交代 | 依 Test RMSE 最低保存 | 方法學重大風險 | 產生 test-selection bias |
| 獨立 Test | 論文報告 Test 結果 | 無 validation 選模後的一次性 Test | 不足 | 目前「最佳 Test」不是嚴格獨立 Test |
| 當前可執行性 | 應可訓練 | `DAST_test.py` 有 conflict markers | **失敗** | `python -m py_compile DAST_test.py` 回傳 SyntaxError |

## 7. 論文與使用者提供 `DAST_main.py` 的矛盾

| 項目 | 論文 | `C:/Users/hjs92/Downloads/DAST_main.py` | 應採用 |
|---|---:|---:|---|
| hidden dimension | 64 | 32 | 論文 64 |
| Optimizer | RAdam | Adam | 論文 RAdam |
| Epochs | 100 | 150 | 論文 100 |
| Loss | RMSE | `sqrt(MSE)`，實質 RMSE | 兩者概念一致 |
| Batch size | 256 | 256 | 256 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Encoder/decoder/heads | 2/1/4 | 2/1/4 | 一致 |
| Decoder sequence length | 未揭露 | 4 | 只能標為官方程式細節，不能稱為論文明示值 |
| 重複次數 | 10 | `for k in range(10)` | 10 |
| 可直接執行 | 論文稱公開 PyTorch code | 路徑為 `../` placeholder，結尾 `score_list` 未定義 | 此檔只能作實作線索，不能當完整可執行基準 |

## 8. 優先修正順序

1. 先解決 `DAST_test.py` Git conflict，並以 `py_compile`、單元測試和一個 forward/train smoke test 驗證；在此之前不要宣稱目前 checkout 可重現。
2. 決定主 `config.json` 是否以「論文還原」為目標；若是，FD002/FD004 應關閉 condition-wise scaling，保留純 train-fitted Min-Max。condition-wise scaling 應放在另行標記的改良實驗。
3. 加入 engine-level validation：以 validation RMSE 選 checkpoint，載入後只測官方 Test 一次；不要用 Test RMSE 選模型。
4. 跑 10 個預先固定 seeds，報 mean ± SD；才能與論文十次平均作正式比較。
5. 明確決定是否依原始程式將負 prediction clamp 到 0，並在所有實驗固定同一規則。
6. 對 `dec_seq_len=4` 做一次結構確認：目前最後四列包含兩個統計特徵，不是四個純時間點；若維持此作法，論文撰寫時必須如實描述。

## 9. 最終判定

- **模型機制：高度還原。** 雙路平行 encoder、兩種 attention 軸、feature fusion、masked decoder、cross-attention、輸出頭均存在。
- **資料前處理：實際 run 高度還原；主 config 不完全還原。** 四資料集 run 關閉 condition scaling 時符合論文 Min-Max；主 config 開啟 condition-wise z-score 則偏離論文。
- **訓練參數：論文明示參數已對齊。** RAdam、100 epochs、batch 256、LR 0.001、RMSE、dropout 0.2 全部一致。
- **完整還原：目前不能判定達 90%。** 核心方法約 92.2%，但加入十次重複、checkpoint/Test protocol 與當前 conflict 後，完整流程約 86%，且當前 checkout 先被可執行性 gate 擋下。
