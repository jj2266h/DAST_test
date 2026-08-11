# FD004 訓練震盪根因診斷計畫

> 目標：找出 FD004 在訓練中 **score 跨 epoch 大幅震盪** 且 **預測 RUL 大量集中於 125 (RUL_max)** 的根因，量化操作條件、感測器、故障模式對訓練的影響，最後提出改善方案（本計畫不負責實作修改、不重訓主模型）。
>
> 病徵摘要：
> - Score（CMAPSS 不對稱指數懲罰）在不同 epoch / 不同 seed 間擺幅巨大
> - 預測 RUL 直方圖在 125 附近出現高峰（mode collapse 到上限），尤其早期樣本被預測為 125 → late prediction → 巨額 score 懲罰
>
> 假設清單（會在分析中逐一驗證或反駁）：
> 1. **H1 — 操作條件混雜**：FD004 有 6 種 op condition × 2 種故障模式，全域 MinMax 正規化把退化訊號淹沒在工況差異中
> 2. **H2 — 視窗內條件切換**：window_size=60 內常含多個 op condition，使統計特徵（線性回歸斜率、均值）變雜訊
> 3. **H3 — RUL 標籤上限分布偏斜**：FD004 引擎 cycle 較長，超過 125 被截斷的樣本比例最高，模型最佳保守策略就是猜 125
> 4. **H4 — 模型對 op condition 無感**：`data_process.py` 把 cols 2,3,4 (op1/op2/op3) 整個刪掉，模型沒有 op 標籤可用
> 5. **H5 — 故障模式糾纏**：兩種 fault mode 在感測器空間軌跡不同，模型難同時擬合，loss landscape 多 local minima → epoch 間擺盪
> 6. **H6 — 統計特徵正規化跨工況失真**：linear regression coef 的尺度受 op condition 主宰，全域 MinMax 後條件差異大的視窗會被擠到極端值

---

## Phase 0：既有資源盤點（不寫 code）

| 任務 | 內容 |
|------|------|
| 0.1 | 讀 `紀錄/experiment_log.csv` 與 `實驗記錄.xlsx`，整理 FD001–FD004 的歷史 RMSE / score / best epoch |
| 0.2 | 確認既有 `dast_FD00X_best.pth` 對應的 config（避免分析用錯權重） |
| 0.3 | 確認 `Cmapss_data/` 與 `train_dataset/*.mat` 存在且未過期 |

**產出**：`analysis/00_inventory.md`（一頁式盤點，含歷史指標表）

---

## Phase 1：資料層探索（不需模型）

### 1.1 操作條件分群分析
- **腳本**：`analysis/01_op_condition_clustering.py`
- **作法**：
  - 讀 `train_FD001.txt` ~ `train_FD004.txt` 的原始 op_setting 1/2/3 欄
  - 對 FD002 / FD004 跑 KMeans(k=6)，確認確實能分出 6 群
  - 對 FD001 / FD003 確認只有 1 群（sanity check）
- **產出**：
  - `plots/fd004/op_clusters_3d.png` — 3D scatter，4 個 dataset 並排
  - `analysis/results/op_cluster_summary.csv` — 每個 cluster 的中心、樣本數、引擎數
- **可得知**：
  - 操作條件實際分布是離散還是連續（是否真的剛好 6 群）
  - 各 cluster 的樣本是否平衡（不平衡 → 罕見條件學不好）
  - FD002 vs FD004 的 op cluster 是否相同（後續可共用 scaler）

### 1.2 各操作條件下的感測器分布
- **腳本**：`analysis/02_sensor_per_condition.py`
- **作法**：
  - 用 1.1 的 cluster 標籤把 14 個保留感測器分組
  - 對每個感測器畫 6 條（每個 op condition 一條）小提琴圖
  - 計算 between-condition variance / within-condition variance（對比退化訊號的尺度）
- **產出**：
  - `plots/fd004/sensor_distribution_by_op.png` — 14×6 grid
  - `analysis/results/sensor_op_variance.csv` — 每個感測器的「條件主宰指數」(BCV/WCV)
- **可得知**：
  - 哪些感測器的讀值幾乎被 op condition 決定（高 BCV/WCV）→ 這些感測器在全域正規化下，退化訊號會被條件差異淹沒
  - 哪些感測器主要反映退化（低 BCV/WCV）→ 是真正有用的訊號
  - 列出建議「分條件正規化優先順序」清單

### 1.3 故障模式分離度
- **腳本**：`analysis/03_fault_mode_separation.py`
- **作法**：
  - FD004 原始資料無 fault mode 標籤；用引擎末段（最後 30 cycle）的多感測器特徵跑 PCA 或 GMM(k=2)
  - 把每台引擎的 PC1/PC2 軌跡疊圖
- **產出**：
  - `plots/fd004/fault_mode_pca_trajectories.png`
  - `analysis/results/inferred_fault_mode.csv` — 每台引擎的推測 fault mode
- **可得知**：
  - 兩種故障模式是否在感測器空間清楚可分
  - 若可分，後續可建議「分模式建模」；若不可分，model 在學的時候會被拉扯

### 1.4 RUL 標籤分布與 piecewise 截斷比例
- **腳本**：`analysis/04_rul_distribution.py`
- **作法**：
  - 對 FD001–FD004 各算「真實 RUL > 125 被截斷成 125」的樣本比例
  - 畫 RUL histogram（截斷前 vs 截斷後）
- **產出**：
  - `plots/cross_dataset/rul_histogram.png`
  - `analysis/results/rul_truncation_ratio.csv`
- **可得知**：
  - FD004 截斷比例若顯著高於其他資料集 → 標籤本身偏向 125 → 模型保守策略就是猜 125（H3）
  - 給出「若把 RUL_max 改成 130 / 150，截斷比例變化曲線」作為改善依據

### 1.5 視窗內操作條件切換頻率
- **腳本**：`analysis/05_window_op_switching.py`
- **作法**：
  - 對 train 與 test 切出的所有 window（size=60），數每個 window 內出現幾種 op condition
  - 算「平均每 window 切換次數」、「純條件視窗比例」
- **產出**：
  - `plots/fd004/window_op_switch_dist.png` — 4 dataset 直方圖並排
  - `analysis/results/window_purity_stats.csv`
- **可得知**：
  - FD004 視窗混雜程度的量化值（H2 直接驗證）
  - 若混雜比例極高 → 解釋為何全域統計特徵（斜率、均值）會劇烈跳

---

## Phase 2：前處理層分析（不需模型）

### 2.1 全域 vs 分條件 MinMax 比較
- **腳本**：`analysis/06_normalization_compare.py`
- **作法**：
  - 在記憶體裡對 FD004 重跑兩種 scaler：
    - (A) 現有：全域 MinMax
    - (B) 替代：用 1.1 的 cluster label 做 per-condition MinMax
  - 對同一感測器、同一引擎生命週期，把 (A)(B) 兩條曲線並排
- **產出**：
  - `plots/fd004/normalization_compare.png` — 選 3 個代表感測器，各畫 2~3 台引擎
  - `analysis/results/normalization_stats.csv` — (A)(B) 下退化單調性指標（Spearman ρ vs cycle）
- **可得知**：
  - 分條件正規化能讓多少感測器的退化單調性提升（H1 直接驗證）
  - 量化「應該分條件正規化」的證據

### 2.2 統計特徵在條件切換下的行為
- **腳本**：`analysis/07_stat_feature_jitter.py`
- **作法**：
  - 抽 100 個「同一引擎、相鄰 5 個 cycle」的視窗，分別在「視窗內條件純」與「視窗內條件混」兩組
  - 算這兩組的 14 維 regression coef + 14 維 mean 的 std
- **產出**：
  - `plots/fd004/stat_feature_jitter.png` — 純 vs 混雜兩組對比
  - `analysis/results/stat_feature_noise_floor.csv`
- **可得知**：
  - 統計特徵在條件混合視窗下的雜訊放大倍率（H6）
  - 若放大顯著 → 建議改用「per-condition 統計特徵」或拋棄 regression coef 改用更 robust 的特徵（如 Theil-Sen slope）

### 2.3 視窗大小對條件混雜的影響
- **腳本**：`analysis/08_window_size_sensitivity.py`
- **作法**：
  - 不重訓，純對 window_size ∈ {20, 30, 40, 60, 80} 算純條件視窗比例
- **產出**：
  - `plots/fd004/window_size_vs_purity.png`
- **可得知**：
  - 視窗縮小是否能換到更純的條件，但代價是看不到完整退化趨勢
  - 給出「window_size 與純度」的 trade-off 曲線

---

## Phase 3：模型層診斷（用既有 `dast_FD004_best.pth`，不重訓）

### 3.1 Per-condition 預測誤差分解
- **腳本**：`analysis/09_per_condition_error.py`
- **作法**：
  - 載入 `dast_FD004_best.pth`，對 test set 全部視窗預測
  - 用 1.1 的 cluster label 把每個視窗標上其「主導 op condition」
  - 分群算 RMSE / score
- **產出**：
  - `plots/fd004/per_condition_rmse.png`
  - `analysis/results/per_condition_score.csv`
- **可得知**：
  - 哪一個 op condition 是錯誤主要來源（若集中在某 1–2 個 cluster，建議優先處理）
  - 罕見條件 RMSE 是否顯著高（樣本不平衡的證據）

### 3.2 預測值集中於 125 的量化
- **腳本**：`analysis/10_prediction_concentration.py`
- **作法**：
  - 對 FD001–FD004 best model 都跑 test set 預測
  - 算「預測 RUL ∈ [120, 125]」的比例、預測直方圖、預測 vs 真值散點圖
  - 畫「真實 RUL 不到 50 但被預測到 ≥ 120」的 false-late 樣本比例
- **產出**：
  - `plots/cross_dataset/pred_concentration_at_125.png`
  - `analysis/results/prediction_concentration.csv`
  - `plots/fd004/false_late_samples.png`
- **可得知**：
  - 量化「mode collapse 到 125」嚴重程度，FD004 是否獨樹一幟
  - false-late（低 RUL 被預測為高 RUL）樣本特徵 → 是否集中在某 op condition 或某 fault mode
  - 直接對應 score 為何爆炸（score 對 late prediction 指數懲罰）

### 3.3 Score 震盪的時序拆解
- **腳本**：`analysis/11_score_oscillation_from_log.py`
- **作法**：
  - 讀 `紀錄/experiment_log.csv` 與 `plots/<timestamp>_FD004.png` 對應的逐 epoch 紀錄（若不夠，建議跑一次純診斷訓練：100 epoch，每 epoch 存 test predictions，**這不算改善實作，只是收集診斷數據**，需用戶確認是否同意）
  - 畫 score-vs-epoch 與 RMSE-vs-epoch
  - 對每個 epoch 算「125 集中比例」並疊圖
- **產出**：
  - `plots/fd004/score_oscillation_decomposition.png` — score、RMSE、125 集中比例三條曲線同一張
  - `analysis/results/epoch_diagnostics.csv`
- **可得知**：
  - score 震盪是否與「125 集中比例」高度相關（驗證 H3+H4 連帶 score 病因）
  - 哪一類預測（特定 op condition 或 fault mode）在 epoch 間切換 → 對應到 H5 的 loss landscape 多模態

### 3.4 Sensor / Time Attention 視覺化
- **腳本**：`analysis/12_attention_viz.py`
- **作法**：
  - 修改 `DAST_Network.py` 暫時返回中間 attention（forward hook，不改主邏輯）
  - 抽 4 種代表性視窗：純條件×早期、純條件×晚期、混條件×早期、混條件×晚期
  - 畫 sensor attention heatmap (14×14) 與 time attention heatmap (60×60)
- **產出**：
  - `plots/fd004/attention_pure_vs_mixed.png` — 4×2 grid
- **可得知**：
  - 模型在條件混合視窗 attention 是否變散亂（沒有清楚 focus）
  - 若混條件下 attention 散亂 → 解釋為何混條件視窗預測不穩定

### 3.5 梯度範數與 loss landscape 切片
- **腳本**：`analysis/13_grad_norm_landscape.py`
- **作法**：
  - 載入 best model，對 batch（純條件 vs 混條件分開組）做 forward+backward，記錄各層梯度 L2 norm
  - 對 FD001 同樣做一次當對照
  - 額外：在權重空間沿兩個隨機方向各 ±α 取 21 點，畫 1D loss section（純診斷，不更新權重）
- **產出**：
  - `plots/fd004/grad_norm_per_layer.png`
  - `plots/fd004/loss_section_1d.png`
- **可得知**：
  - 是否某些層梯度爆炸或消失，導致 epoch 間更新方向不穩
  - loss landscape 在 best 點附近是否平坦或多個局部低谷（若多模態 → 解釋震盪）

---

## Phase 4：對照組（FD001/FD002/FD003）

### 4.1 跨資料集量化對照
- **腳本**：`analysis/14_cross_dataset_summary.py`
- **作法**：
  - 把 Phase 1.1 / 1.4 / 1.5 / 3.2 的結果聚合成單一表
- **產出**：
  - `plots/cross_dataset/summary_radar.png` — 雷達圖：op cluster 數、視窗純度、RUL 截斷率、125 集中比、test RMSE、score
  - `analysis/results/cross_dataset_comparison.csv`
- **可得知**：
  - FD004 的問題哪些是「程度差異」（如截斷率比 FD002 略高）哪些是「質的差異」（如 6×2 = 12 種潛在子群）
  - 用此表向上層解釋「為何 FD004 是 hard mode」

---

## Phase 5：整合報告 + 改善方案（不實作）

### 5.1 整合分析報告
- **產出**：`analysis_report_FD004.md`
- **章節**：
  1. Executive summary（一頁，含核心結論）
  2. 各假設驗證表（H1–H6 各 ✅/❌/部分）
  3. Phase 1–4 關鍵圖表與文字解釋
  4. 改善方案（按可行性 / 預期收益排序）

### 5.2 改善方案候選（依分析結果客製，預擬清單）
| ID | 方向 | 對應驗證的假設 | 預期收益 | 實作難度 |
|----|------|----------------|----------|----------|
| F1 | 分操作條件 MinMax 正規化 | H1, H6 | 高 | 低 |
| F2 | 加入 op condition embedding（one-hot 或 learned）作為輔助輸入 | H4 | 高 | 中 |
| F3 | 分 fault mode ensemble（先聚類再分頭訓） | H5 | 中 | 高 |
| F4 | 改 piecewise RUL 切點（125 → 130/150 或學出來） | H3 | 中 | 低 |
| F5 | Loss 改 weighted MSE 或加 KL 正則防 mode collapse | H3 | 中 | 中 |
| F6 | 縮小 window_size 或改用 multi-scale window | H2 | 中 | 中 |
| F7 | LR warmup + cosine schedule + gradient clipping | 3.5 結果 | 低中 | 低 |
| F8 | 拋棄 linear regression coef，改 robust slope（Theil-Sen / RANSAC） | H6 | 低中 | 低 |
| F9 | 對 op condition 主宰的感測器做 per-cluster z-score 而非 MinMax | H1 | 中 | 低 |

> 哪幾個會被列入最終建議，取決於 Phase 1–4 的數據結果；若某假設被反駁，對應方案就會降級或刪除。

---

## 整體檔案結構

```
DAST_test/
├── FD004_diagnosis_plan.md        ← 本文件
├── analysis/
│   ├── _common.py                 ← 共用：資料 / 模型 / 繪圖 helper
│   ├── 00_inventory.md
│   ├── 01_op_condition_clustering.py
│   ├── 02_sensor_per_condition.py
│   ├── 03_fault_mode_separation.py
│   ├── 04_rul_distribution.py
│   ├── 05_window_op_switching.py
│   ├── 06_normalization_compare.py
│   ├── 07_stat_feature_jitter.py
│   ├── 08_window_size_sensitivity.py
│   ├── 09_per_condition_error.py
│   ├── 10_prediction_concentration.py
│   ├── 11_score_oscillation_from_log.py
│   ├── 12_attention_viz.py
│   ├── 13_grad_norm_landscape.py
│   ├── 14_cross_dataset_summary.py
│   └── results/
│       ├── *.csv
│       └── *.json
├── plots/
│   ├── fd004/                     ← FD004 專屬分析圖
│   └── cross_dataset/             ← 跨資料集對照圖
└── analysis_report_FD004.md       ← Phase 5 最終報告
```

## 共用模組 `_common.py` 要提供的函式

```python
load_raw_cmapss(name)                # 讀 train/test/RUL .txt
load_processed_mat(name)             # 讀 train_dataset/*.mat
fit_op_condition_clusters(name, k=6) # 回傳 KMeans + cluster label
load_dast_model(name)                # 載入 best .pth + config 同步
predict_test_set(model, name)        # 回傳 pred / true / op_label / engine_id
cmapss_score(pred, true)             # 標準 CMAPSS asymmetric score
plot_helpers (set_style, savefig_dpi300)
```

集中放 helper 可避免 14 支腳本重複載資料、避免 scaler / cluster 不一致。

---

## 執行順序與里程碑

1. **里程碑 1（不需模型，2–3 小時）**：完成 Phase 1 + 2，產出 8 張圖 + 5 張 CSV。**到此可初步驗證 H1, H2, H3, H6**
2. **里程碑 2（需要 best 模型，2–3 小時）**：完成 Phase 3.1 / 3.2 / 3.4 / 3.5，產出 5 張圖。**驗證 H4**
3. **里程碑 3（需要逐 epoch logging，最多需要跑一次純診斷訓練）**：Phase 3.3。需要先和你確認是否能跑一次「不修模型、只多存 epoch 紀錄」的訓練
4. **里程碑 4（聚合，1 小時）**：Phase 4 + Phase 5 報告

每個里程碑結束後我會回報、附圖，你確認結論後再進下一階段。

---

## 待你確認的問題

1. **Phase 3.3 的純診斷訓練**：是否同意我跑一次 100 epoch 的 FD004 訓練，唯一目的是逐 epoch 存 test predictions（不改 model、不改 hyperparam），用來分析 score 震盪？這嚴格來說是「收集診斷數據」而非「實作改善」，但會花約 30 分鐘 GPU 時間
2. **Fault mode 推測**（Phase 1.3）：用 PCA + GMM 對引擎末段聚類來推測兩種 fault mode 即可，還是你已經有更精確的標註方式？
3. **舊紀錄可用性**：`紀錄/experiment_log.csv` 和 `實驗記錄.xlsx` 裡的歷史 FD004 紀錄欄位包含哪些？（會影響 Phase 0 與 Phase 3.3 的依賴）
4. **GPU 可用性**：分析腳本需要載入 `.pth` 跑 forward。你的環境是 GPU 還是純 CPU？這會影響執行時間預估
