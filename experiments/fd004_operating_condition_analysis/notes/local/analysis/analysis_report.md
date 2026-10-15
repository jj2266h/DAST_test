# CMAPSS FD004 資料集探索分析報告

**資料集**：NASA CMAPSS FD004（渦輪扇葉引擎模擬退化資料）
**分析日期**：2025 年
**環境**：conda activate dast
**工具**：Python 3、pandas、numpy、scipy、scikit-learn、matplotlib、seaborn

---

## 1. 資料集概述

FD004 是 CMAPSS 四個子集中複雜度最高的資料集，其特點如下：

| 項目 | 數值 |
|------|------|
| 故障模式 | 2 種（HPC Degradation + Fan Degradation 同時發生） |
| 操作條件數 | 6 種 |
| 訓練集引擎數 | 249 台 |
| 測試集引擎數 | 248 台 |
| 訓練集資料行數 | 61,249 行 |
| 測試集資料行數 | 41,214 行 |
| 感測器數量 | 21 個 |
| 操作設定欄位 | 3 個（op1, op2, op3） |
| RUL 上限（Piecewise） | 125 cycles |

### 1.1 引擎壽命分佈

訓練集每台引擎的完整退化週期：

- 最短壽命：**128 cycles**
- 平均壽命：**246.0 cycles**
- 最長壽命：**543 cycles**

測試集（截斷點，非完整壽命）：

- 最短記錄：**19 cycles**
- 平均記錄：**166.2 cycles**
- 最長記錄：**486 cycles**

測試集真實 RUL（RUL_FD004.txt）：

- 最小 RUL：**6 cycles**
- 平均 RUL：**86.6 cycles**
- 最大 RUL：**195 cycles**

**圖 1-1：訓練集引擎壽命分佈**
![engine_life_dist](figures/01_engine_life_dist.png)

**圖 1-2：RUL 標籤分佈（Piecewise，上限 125 cycles）**
![rul_dist](figures/01_rul_distribution.png)

**圖 1-3：測試集記錄週期分佈**
![test_life](figures/01_test_life_dist.png)

---

## 2. 操作條件分析

FD004 包含 6 種不同的操作條件（飛行高度、節流閥開度、飛行馬赫數的組合），這些條件會直接影響感測器的讀值，是分析時必須處理的重要干擾因素。

### 2.1 操作條件識別（K-Means 聚類）

對 op1、op2、op3 三個操作設定欄位進行 K-Means 聚類（k=6），識別出 6 種操作條件（OC0–OC5）。

各操作條件的中心點（原始值）：

| 條件 | op1 (高度) | op2 (馬赫數) | op3 (油門) | 訓練集佔比 |
|------|-----------|------------|-----------|----------|
| OC0 | 0.0015 | 0.0005 | 100.0 | 15.1% |
| OC1 | 10.003 | 0.2505 | 100.0 | 15.1% |
| OC2 | 20.003 | 0.7005 | 100.0 | 14.8% |
| OC3 | 25.0031 | 0.6205 | 60.0 | 14.9% |
| OC4 | 35.003 | 0.8405 | 100.0 | 15.0% |
| OC5 | 42.003 | 0.8405 | 100.0 | 25.1% |

**圖 2-1：操作條件 2D 分佈（op1 vs op2, op1 vs op3）**
![oc_scatter](figures/02_oc_scatter_2d.png)

**圖 2-2：操作條件 3D 分佈（★ 為聚類中心）**
![oc_3d](figures/02_oc_3d_scatter.png)

**圖 2-3：各操作條件出現頻率**
![oc_freq](figures/02_oc_frequency.png)

**圖 2-4：引擎操作條件切換時序（3 台代表引擎）**
![oc_ts](figures/02_oc_time_series.png)

### 2.2 發現

1. 六種操作條件在 3D 空間中明顯分群，聚類效果良好。
2. 各引擎在同一次退化過程中**隨機切換**操作條件，並非固定在單一條件下運行。
3. 操作條件在不同引擎間的分佈比例大致均勻，不存在明顯偏差。

---

## 3. 感測器 EDA

### 3.1 感測器有效性篩選

依據訓練集標準差篩選有效感測器（閾值：std > 0.01）：

- **被排除的近常數感測器（7 個）**：s1, s5, s6, s10, s16, s18, s19
- **有效感測器（14 個）**：s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21

這些被排除的感測器在整個退化過程中幾乎沒有變化，對於退化分析無貢獻。

**圖 3-1：各感測器標準差（紅色 = 被排除）**
![sensor_std](figures/03_sensor_std.png)

### 3.2 感測器時序趨勢

選取壽命最短、中等、最長的三台引擎，觀察 6 個最高變異量感測器的時序趨勢：

**圖 3-2：感測器時序（3 台代表引擎，top-6 高變異感測器）**
![sensor_ts](figures/03_sensor_time_series.png)

### 3.3 感測器間相關性

**圖 3-3：有效感測器 Pearson 相關矩陣**
![corr](figures/03_sensor_correlation.png)

### 3.4 感測器與 RUL 的關係

**圖 3-4：感測器 vs RUL 散點圖（top-6 高變異感測器）**
![svr](figures/03_sensor_vs_rul.png)

### 3.5 發現

1. 多個感測器之間存在高度相關性，反映出渦輪引擎內部各部件的物理耦合關係。
2. 感測器時序圖在原始數據中呈現大量跳動，這是**操作條件切換**造成的干擾，而非退化訊號。
3. 直接觀察原始感測器與 RUL 的關係時，因操作條件遮蔽，趨勢並不清晰（詳見 Phase 4）。

---

## 4. 操作條件 × 感測器交互分析

### 4.1 操作條件對感測器的影響

**圖 4-1：各感測器在不同操作條件下的分佈（Boxplot）**
![boxplot](figures/04_sensor_boxplot_by_oc.png)

**圖 4-2：操作條件對感測器均值偏移熱圖（相對偏移大小）**
![offset](figures/04_oc_offset_heatmap.png)

### 4.2 OC 標準化的必要性

受操作條件影響最大的感測器：**s12, s7, s21**

這些感測器在不同操作條件下的均值差異極大（相對偏移 CoV 最高），直接使用原始值進行退化分析會產生嚴重誤導。

**圖 4-3：OC 標準化前後的感測器–RUL 關係對比**
![norm](figures/04_before_after_norm.png)

### 4.3 發現

1. 幾乎所有有效感測器都受操作條件顯著影響，不同條件下的均值分佈完全不重疊。
2. OC 標準化後，來自不同操作條件的數據點在感測器–RUL 空間中趨於重合，退化趨勢更加清晰可見。
3. **結論：對 FD004 進行任何退化分析前，必須先按操作條件進行標準化，否則操作條件變換會完全淹沒退化訊號。**

---

## 5. 降解趨勢分析

### 5.1 感測器與 RUL 的 Spearman 相關

基於 OC 標準化後的感測器值，與 RUL 進行 Spearman 相關分析（適用於非線性單調關係）：

與 RUL 相關性最高的感測器（top-6）：**s11, s4, s17, s3, s2, s13**

**圖 5-1：Spearman 相關係數（所有有效感測器，OC 標準化後）**
![spearman](figures/05_spearman_correlation.png)

### 5.2 降解軌跡

**圖 5-2：降解時序軌跡（cycle vs 標準化感測器，5 台引擎）**
![traj](figures/05_degradation_trajectories.png)

**圖 5-3：RUL 對齊降解軌跡（以剩餘壽命為 x 軸）**
![rul_traj](figures/05_rul_aligned_trajectories.png)

### 5.3 PCA 健康指標（Health Indicator）

將所有有效的 OC 標準化感測器進行 PCA，取第一主成分作為健康指標（HI）：

- PC1 解釋變異量：**47.3%**
- PC2 解釋變異量：**26.8%**
- HI 與 RUL 的 Spearman 相關：**ρ = -0.6764**

**圖 5-4：PCA 健康指標（cycle 趨勢 + vs RUL）**
![hi](figures/05_health_indicator.png)

### 5.4 發現

1. OC 標準化後，多個感測器展現出清晰的單調退化趨勢（隨 RUL 遞減而上升或下降）。
2. RUL 對齊後的軌跡顯示各引擎在接近失效時退化加速，呈現四階段降解模式：穩定期 → 膝點 → 加速退化 → 失效。
3. PCA 健康指標整合多個感測器訊號，與 RUL 的相關性顯著優於單一感測器。

---

## 6. RUL 預測模型：Random Forest

### 6.1 特徵工程

| 特徵類型 | 說明 |
|---------|------|
| 標準化感測器值 | 14 個 OC 標準化感測器 |
| 滑動視窗均值 | 前 30 cycles 的滑動平均（每個感測器） |
| 滑動視窗標準差 | 前 30 cycles 的滑動標準差（每個感測器） |
| 週期數（cycle） | 當前週期 |
| 操作條件（OC） | 當前操作條件類別（0–5） |
| **總特徵數** | **44 個** |

訓練樣本數：**54,028 筆**

### 6.2 模型設定

- 演算法：Random Forest Regressor
- 樹的數量：100
- 最大深度：15
- 最小葉節點樣本數：5
- 評估集：測試集最後一個時間點（248 台引擎）

### 6.3 測試集結果

| 評估指標 | 數值 |
|---------|------|
| RMSE | **28.45 cycles** |
| MAE | **20.85 cycles** |
| MAPE | **30.56%** |
| PHM'08 Score | **6065.2**（越低越好） |

**圖 6-1：Random Forest 特徵重要性（top-30）**
![feat_imp](figures/06_feature_importance.png)

**圖 6-2：預測值 vs 真實 RUL**
![pred_actual](figures/06_pred_vs_actual.png)

**圖 6-3：預測誤差分佈**
![err_dist](figures/06_error_distribution.png)

**圖 6-4：預測誤差 vs 真實 RUL**
![err_rul](figures/06_error_by_rul.png)

### 6.4 發現

1. 滑動視窗均值（rolling mean）特徵的重要性通常高於瞬時值，反映退化是一個累積過程。
2. MAPE 在 RUL 較短的引擎上通常偏高，因為誤差相對於小 RUL 值的比例更大。
3. 模型在高 RUL（引擎仍健康）的引擎上預測偏差較大，這是 Piecewise RUL 上限設計的預期行為。

---

## 7. RUL 預測模型：DAST（雙注意力感測器–時間 Transformer）

### 7.1 模型架構

DAST（Dual-Attention Sensor-Time Transformer）是一個基於 Transformer 的序列到點預測模型，專為多感測器時序退化預測設計：

| 超參數 | 數值 |
|--------|------|
| 時間步長（time_step） | 62（60 cycles 視窗 + 2 統計行） |
| 輸入維度（input_size） | 14 個感測器 |
| 感測器/時間注意力維度 | 64 |
| Encoder 層數 | 2 |
| Decoder 序列長度 | 10 |
| 注意力頭數 | 4 |
| Dropout | 0.1 |

資料預處理流程：MinMax 標準化 → 移除 op1/op2/op3 及 7 個無資訊感測器 → 60-cycle 滑動視窗 → 附加回歸係數與均值統計行 → 形狀 (N, 62, 14)

### 7.2 測試集結果

| 評估指標 | 數值 |
|---------|------|
| RMSE | **27.68 cycles** |
| MAE | **21.96 cycles** |
| MAPE | **42.57%** |
| PHM'08 Score | **4055.9**（越低越好） |

**圖 7-1：DAST 預測值 vs 真實 RUL**
![dast_pred_actual](figures/08_dast_pred_vs_actual.png)

**圖 7-2：DAST 預測誤差分佈**
![dast_err_dist](figures/08_dast_error_dist.png)

**圖 7-3：DAST 預測誤差 vs 真實 RUL（熱圖顯示誤差大小）**
![dast_err_rul](figures/08_dast_error_by_rul.png)

---

## 8. DAST 特徵重要性分析

DAST 為深度學習模型，無法直接輸出 Random Forest 式的特徵重要性。本節採用兩種互補方法進行分析：

### 8.1 排列重要性（Permutation Importance）

**方法**：對每個感測器的數值進行隨機排列（打亂樣本間的對應關係），量測 RMSE 的增加量。RMSE 增幅越大，代表該感測器被破壞後模型表現越差，即該感測器對預測越重要。共重複 30 次取平均值與標準差。

**圖 8-1：DAST 感測器排列重要性（±標準差）**
![perm_imp](figures/09_dast_permutation_importance.png)

前 5 名感測器（排列重要性）：

| 排名 | 感測器 | RMSE 增加量 (cycles) |
|------|--------|---------------------|
| 1 | s8（HPC Outlet 溫度） | ~39.2 |
| 2 | s11（HPC 壓力比） | ~33.8 |
| 3 | s4（LPC 出口溫度） | ~32.4 |
| 4 | s7（LPC 出口壓力） | ~31.8 |
| 5 | s3（總溫入口） | ~29.5 |

### 8.2 梯度顯著性圖（Gradient Saliency）

**方法**：計算輸出 RUL 預測值對輸入的梯度，取 `|梯度 × 輸入|` 作為各時間步、各感測器對預測的貢獻量，對所有 248 個測試樣本取平均。

**圖 8-2：DAST 梯度顯著性熱圖（感測器 × 時間步）**
![saliency](figures/09_dast_gradient_saliency_heatmap.png)

**圖 8-3：感測器重要性 + 時間步重要性（梯度法 vs 排列法）**
![combined](figures/09_dast_sensor_time_importance.png)

### 8.3 發現

1. **兩種方法高度一致**：排列重要性與梯度顯著性均將 s8、s11、s4 列為前三名重要感測器，驗證了結果的可靠性。

2. **s8 最重要**：s8（HPC Outlet Total Temperature）是最關鍵的退化指標，打亂後 RMSE 增加約 39.2 cycles（+142%），符合渦輪引擎退化機理—— HPC 高壓壓縮機的溫度上升是核心故障特徵。

3. **s9 貢獻最低**：s9 的排列重要性僅 0.12 cycles，梯度顯著性也極低，代表 DAST 幾乎不依賴此感測器做預測。

4. **統計附加行最重要（時間步維度）**：梯度分析顯示，data_process.py 附加的 2 個統計行（回歸係數、均值）的顯著性遠超過所有 60 個視窗時間步，說明這兩行的長期趨勢資訊對模型預測至關重要。

5. **近期時間步比遠期重要**：在 60 個滑動視窗步中，接近當前時刻（t=58, 57, 54）的重要性高於早期時間步，符合「越接近當前狀態越能反映退化程度」的物理直覺。

---

## 9. DAST vs Random Forest 模型比較

**圖 8-1：DAST 與 Random Forest 真實 RUL vs 預測值對比**
![dast_vs_rf](figures/08_dast_vs_rf_comparison.png)

### 8.1 指標對比表

| 評估指標 | DAST | Random Forest | 優勝 |
|---------|------|---------------|------|
| RMSE（cycles） | **27.68** | 28.45 | **DAST** |
| MAE（cycles） | 21.96 | **20.85** | **RF** |
| MAPE（%） | 42.57 | **30.56** | **RF** |
| PHM'08 Score | **4055.9** | 6065.2 | **DAST** |

### 8.2 分析

1. **RMSE（均方根誤差）**：DAST 的 RMSE（27.68 cycles）低於 Random Forest（28.45 cycles），代表在平均預測精度上 DAST 略勝一籌。

2. **PHM'08 非對稱評分**：DAST 的 PHM Score（4055.9）遠低於 RF（6065.2），說明 DAST 更少出現「過晚預測失效」（低估 RUL）的危險錯誤。PHM Score 對於早期預警更具實用意義。

3. **MAE 與 MAPE**：RF 的中位數誤差（MAE）和比例誤差（MAPE）較低，顯示 RF 在多數引擎上的估計較為穩定，而 DAST 可能在少數引擎上產生較大的絕對誤差。

4. **架構優勢**：DAST 透過雙路徑注意力機制同時建模感測器間互動與時序動態，更適合捕捉 FD004 的複雜退化模式（2 種同時發生的故障模式）。

---

## 10. 結論與總結

### 10.1 主要發現

1. **操作條件是最大干擾源**：FD004 的 6 種操作條件對所有感測器讀值造成顯著偏移，必須在分析前進行 OC 標準化。

2. **有效感測器**：21 個感測器中有 7 個為近常數（無資訊量），應予排除；其餘 14 個感測器在 OC 標準化後均展現退化趨勢。

3. **退化模式**：FD004 的 2 種故障模式（HPC + Fan Degradation）造成各引擎退化軌跡具有較大個體差異，比 FD001/FD003（單一故障模式）更難預測。

4. **PCA 健康指標**：第一主成分整合多感測器訊號，與 RUL 的 Spearman 相關 ρ = -0.6764，是一個有效的整合健康狀態指標。

5. **DAST 模型表現最佳**：在 RMSE（27.68 cycles）和 PHM'08 Score（4055.9）兩項關鍵指標上，預訓練 DAST 均優於 Random Forest 基線，特別是在安全關鍵的非對稱評分上領先顯著。

### 10.2 後續改進方向

- 加入更長滑動視窗（如 50 cycles）或多尺度視窗特徵以強化 RF 基線
- 針對 DAST 進行 FD004 微調訓練以進一步提升 MAE 表現
- 嘗試集成方法（Ensemble DAST + RF）以兼顧兩者優點
- 對兩種故障模式分別建模，針對各自退化特性進行優化

---

*本報告由自動化分析腳本產出，所有圖表儲存於 `analysis/figures/` 目錄。*
