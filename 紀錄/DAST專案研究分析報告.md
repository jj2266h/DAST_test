# DAST 專案研究分析報告

> 生成日期：2026-04-14  
> 分析者：Antigravity AI Assistant  
> 專案路徑：`c:\Users\hjs92\Desktop\DAST_test`

---

## 一、專案目標與架構

### 1.1 研究目標

本專案的核心目標是開發並驗證 **DAST（Dual-Attention Spatio-Temporal Transformer）** 模型，用於**工業設備剩餘使用壽命（Remaining Useful Life, RUL）預測**。

主要研究問題：
- 如何同時捕捉航空發動機感測器資料中的**空間（感測器間）關係**與**時間（時序）關係**？
- 如何透過雙軸注意力機制提升 RUL 預測精度？
- 模型是否能在多種操作條件（FD001～FD004）下泛化？

### 1.2 資料集

| 資料集 | 描述 | 特性 |
|--------|------|------|
| **NASA C-MAPSS (FD001～FD004)** | 航空渦輪發動機模擬退化資料 | 21 個感測器 + 3 個操作變數，標準 RUL 預測基準 |

**C-MAPSS 子資料集差異：**

| 子集 | 操作條件 | 故障模式 | Window Size |
|------|----------|----------|-------------|
| FD001 | 單一 | HPC Degradation | 40 |
| FD002 | 六種 | HPC Degradation | 60 |
| FD003 | 單一 | HPC + Fan Degradation | 40 |
| FD004 | 六種 | HPC + Fan Degradation | 60 |

### 1.3 模型架構（DAST）

```
輸入: (batch_size, window_size, n_sensors)
         ↓
  ┌──────────────────────────────────┐
  │         雙路 Encoder 分支         │
  │                                  │
  │  Sensor Encoder (空間注意力)      │
  │  x.transpose(1,2)                │
  │  → Linear Proj → Multi-Head Attn │
  │  → FFN → LayerNorm               │
  │                                  │
  │  Time-Step Encoder (時序注意力)   │
  │  → Linear Proj → Positional Enc  │
  │  → Multi-Head Attn → FFN         │
  │  → LayerNorm                     │
  └──────────────────────────────────┘
         ↓ 特徵融合 (Concat)
   (batch_size, timestep+sensor, dim_val)
         ↓ LayerNorm
   Decoder (Cross-Attention)
         ↓
   Output FC → RUL 預測值 [0,1]
```

**關鍵超參數（目前預設）：**

| 超參數 | 值 |
|--------|----|
| `dim_val_s / dim_val_t / dim_val` | 64 |
| `n_encoder_layers` | 2 |
| `n_decoder_layers` | 1 |
| `n_heads` | 4 |
| `dropout` | 0.2 |
| `dec_seq_len` | 10 |
| `BATCH_SIZE` | 256 |
| `EPOCHS` | 100 |
| `LR` | 1e-3 (RAdam) |
| `RUL_max` | 125 cycles |

### 1.4 資料處理流程

```
原始 C-MAPSS .txt 資料
        ↓
① data_process.py
  - Min-Max 正規化（Col 2 以後）
  - 刪除 7 個無效感測器 [s1,s5,s6,s10,s16,s18,s19]
  - 保留 14 個有效感測器
  - 滑動時間窗口切割 → trainX, testY
  - 測試集短序列插值補齊（Spline）
  - 儲存為 .mat 格式
        ↓
② Statistical features process .py（可選）
  - 線性迴歸係數特徵（fea1）
  - 均值特徵（fea2）
  - 與原始時間窗口組合 → 增強特徵矩陣
  - 儲存為 *_new.mat
        ↓
③ DAST_test.py
  - 載入 .mat 資料 → Tensor
  - 建構 DataLoader
  - DAST 模型訓練 (100 epochs)
  - 評估指標：RMSE + Score Function
  - 儲存 best model (.pth)
```

---

## 二、研究進度

### 2.1 已完成工作

| 階段 | 狀態 | 說明 |
|------|------|------|
| 核心架構設計 | ✅ 完成 | DAST 雙編碼器 + 解碼器架構已實作 |
| 資料前處理管道 | ✅ 完成 | C-MAPSS FD001~FD004 均可處理 |
| GPU 訓練支援 | ✅ 完成 | nn.ModuleList 支援 CUDA |
| FD001 模型訓練 | ✅ 多版本 | RMSE 最佳: **11.40**（Model_trained 資料夾） |
| FD002 模型訓練 | ✅ 多版本 | RMSE 最佳: **14.75** |
| FD003 模型訓練 | ✅ 初步完成 | 已儲存一個模型 |
| FD004 模型訓練 | ✅ 正在優化 | 存有 `dast_FD004_best.pth` |


### 2.2 各資料集模型演進（FD001 為例）

| 模型版本 | Test RMSE | 備註 |
|----------|-----------|------|
| 11.61 (seed=20) | 11.61 | 固定隨機種子對照 |
| 11.62 | 11.62 | — |
| 11.59 | 11.59 | — |
| 11.46 | 11.46 | — |
| 11.42 | 11.42 | — |
| **11.40** | **11.40** | 目前最佳 |

### 2.3 FD002 模型演進

| 模型版本 | Test RMSE | 備註 |
|----------|-----------|------|
| 15.50 | 15.50 | 早期版本 |
| 15.32 | 15.32 | — |
| 15.18 | 15.18 | — |
| 15.04 | 15.04 | — |
| 15.03 (1002) | 15.03 | — |
| 14.92 | 14.92 | — |
| **14.75** | **14.75** | 目前最佳 |


---

## 三、觀察痛點

### 3.1 🔴 高重要度痛點


#### P2：Statistical features process.py 路徑為硬編碼佔位符
- **問題**：`sio.loadmat('.../train_dataset/')` 為無效路徑，腳本無法直接執行
- **影響**：統計特徵增強管道斷裂，第二步資料處理無法自動化執行
- **根本原因**：程式碼為半完成狀態，尚未整合進主流程

#### P3：data_process.py 未儲存 testX_all / testY_all
- **問題**：腳本末尾只儲存 trainX/Y 與 testX/Y，但 `testX_all` 和 `testY_all`（完整滑動序列）未儲存
- **影響**：若需對測試集做完整時序預測曲線分析，需重新執行資料前處理

### 3.2 🟡 中度重要度痛點

#### P4：Window Size 硬編碼，FD001/FD003 需手動切換
- **問題**：`data_process.py` 第 42 行 `window_Size = 60`，但 FD001/FD003 應使用 40，需手動改動
- **影響**：容易因疏忽使用錯誤 window size，影響實驗可重現性
- **建議**：改為 dict 自動對應 → `WINDOW_MAP = {'FD001': 40, 'FD002': 60, 'FD003': 40, 'FD004': 60}`

#### P5：Decoder 僅使用一層，且未設定多層迴圈
- **問題**：`DAST.forward()` 中 decoder 部分僅取 `self.decoder[0](...)`，即使 `n_decoder_layers > 1` 也只執行一層
- **影響**：設定 `n_decoder_layers=2` 時，第二層被完全忽略，資源浪費且結果不符預期

#### P6：Positional Encoding 僅套用於 Time-step Encoder，Sensor Encoder 無位置資訊
- **問題**：`self.pos_s` 定義了 sensor 端 positional encoding，但在 forward 中未使用（`self.sensor_enc_input_fc(sensor_x)` 直接進 encoder，無 pos_s 調用）
- **影響**：Sensor 的排列順序資訊未被利用

#### P7：實驗記錄未系統化（Excel 與程式碼分離）
- **問題**：`實驗記錄.xlsx` 存在，但 `DAST_test.py` 中雖有 `append_experiment_log` 函式，實際上未被 `main()` 呼叫
- **影響**：實驗結果需手動記錄，容易遺失或不一致

### 3.3 🟢 低重要度待優化

#### P8：MultiHeadAttentionBlock 拼接方式效率低
- `torch.stack(..., dim=-1).flatten(start_dim=2)` 可改為更直接的 `torch.cat(..., dim=-1)`



---

## 四、重要程度評估

| 編號 | 痛點 | 影響層面 | 改善難度 | 優先度 |
|------|------|----------|----------|--------|

| P2 | Statistical 腳本路徑失效 | 資料管道完整性 | 低（修路徑即可） | 🔴 高 |
| P5 | Decoder 多層未生效 | 模型結構正確性 | 低（加迴圈即可） | 🔴 高 |
| P4 | Window Size 硬編碼 | 實驗可重現性 | 低（加 dict 映射） | 🟡 中 |
| P6 | Sensor PE 未啟用 | 模型能力 | 低（加一行 pos_s） | 🟡 中 |
| P3 | testX_all 未儲存 | 分析完整性 | 低（加兩行 savemat） | 🟡 中 |
| P7 | 實驗記錄未自動化 | 研究管理 | 低（呼叫已有函式） | 🟡 中 |
| P8 | MHA 拼接效率 | 運算效率 | 低 | 🟢 低 |


---

## 五、改善建議

### 5.1 立即可行（程式碼小修）

```python
# ① data_process.py —— 自動對應 Window Size
WINDOW_MAP = {'FD001': 40, 'FD002': 60, 'FD003': 40, 'FD004': 60}
window_Size = WINDOW_MAP[DATASET]

# ② data_process.py —— 儲存完整測試集序列
sio.savemat(f'{dataset_path}/{DATASET}_window_size_testX_all.mat', {"testX_all": testX_all})
sio.savemat(f'{dataset_path}/{DATASET}_window_size_testY_all.mat', {"testY_all": testY_all})

# ③ DAST_Network.py —— 修復 Decoder 多層
d = self.dec_input_fc(x[:, -self.dec_seq_len:])
for dec_layer in self.decoder:
    d = dec_layer(d, p)

# ④ DAST_Network.py —— 啟用 Sensor Positional Encoding
e = self.sensor_encoder[0](self.pos_s(self.sensor_enc_input_fc(sensor_x)))

# ⑤ DAST_test.py —— 啟用實驗紀錄
append_experiment_log("experiment_log.csv", {
    "timestamp": datetime.now().isoformat(),
    "dataset": DATASET,
    "best_rmse": best_rmse,
    "epochs": EPOCHS,
    "lr": LR,
    "batch_size": BATCH_SIZE,
})
```

### 5.2 中期實驗設計

- **Ablation Study（消融實驗）**：逐一關閉 Sensor Encoder / Time-step Encoder，驗證各分支對性能的貢獻
- **FD003 / FD004 全面評估**：目前僅儲存少量模型，建議系統化記錄所有 epoch 的 RMSE/Score


### 5.3 長期改進方向

1. **損失函式非對稱化**：由於 RUL 預測對「晚預警」（over-predict）的代價遠大於「早預警」，可引入非對稱Huber Loss 或直接使用 Score Function 作為訓練目標的一部分
2. **Attention Visualization**：輸出 Sensor Encoder 的 attention weight，分析哪些感測器對特定故障模式貢獻最大
3. **統計特徵管道整合**：將 Statistical features process.py 整合進 data_process.py 形成完整管道，或使用 config 統一管理路徑


---

## 六、專案文件結構總覽

```
DAST_test/
├── Cmapss_data/              # 原始 C-MAPSS 資料 (FD001~FD004)
├── train_dataset/            # 滑動窗口後的 .mat 格式資料
├── train_model/              # 訓練過程中儲存的模型（含 best）
├── Model_trained/            # 精選最佳模型（含 FD001~FD003 多版本）

├── 紀錄/                     # 研究記錄（本文件所在）
│
├── data_process.py           # 資料前處理（滑動窗口 + 正規化）
├── Statistical features process .py  # 統計特徵提取（待修復）
├── DAST_Network.py           # DAST 模型架構定義
├── DAST_utils.py             # 注意力機制 + Positional Encoding
├── DAST_test.py              # 訓練與評估主腳本
│
├── dast_FD003_best.pth       # FD003 根目錄最佳模型
├── dast_FD004_best.pth       # FD004 根目錄最佳模型
├── 實驗記錄.xlsx              # 手動實驗記錄
└── requirements.txt          # 依賴套件
```

---

## 七、評估指標說明

### RMSE（均方根誤差）
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2} \times \text{RUL\_max}$$



---

*本報告由 AI 自動分析生成，建議與實際實驗記錄 (`實驗記錄.xlsx`) 對照使用。*
