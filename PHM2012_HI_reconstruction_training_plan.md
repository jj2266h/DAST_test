# PHM2012 以 15 個時域特徵、頻域前處理與 HI Index Reconstruction 進行壽命建模的計畫

## 1. 重新定位

這份計畫改以 `phm2012_dataset/Paper_Huang_YaunDi.pdf` 對 IEEE PHM 2012 bearing problem 的分析為核心。該文指出 PHM2012 bearing RUL 的主要困難不是模型不夠複雜，而是：

- end-of-life degradation signature 不清楚。
- 只有 6 顆完整 run-to-failure learning bearings，資料量少。
- 3 種 operating regimes，且各 bearing 壽命差異大。
- 直接從原始 vibration features 對 RUL regression 容易不穩。
- 對這類 unclear trend 問題，先建立 degradation indicator / health index，再用趨勢到 failure threshold 估 RUL，是重要策略。

因此，本計畫不再單純把 PHM2012 轉成 DAST 的 supervised RUL input，而是先建立可解釋且可平滑的 `HI(t)`，再用 HI reconstruction 建立壽命模型。DAST 可以保留為第二階段的序列模型，但第一階段應先完成穩定的 HI。

## 2. 方法可行性評估

### 2.1 可行，但不應期待一次到位

此方法可行，原因如下：

- PHM2012 每 10 秒一個 `acc_*.csv`，每筆都有 2560 點水平與垂直 vibration，足夠萃取時域與頻域統計特徵。
- 15 個時域特徵可將高頻 waveform 壓縮成低維 degradation descriptors。
- 頻域前處理能突出 bearing impulse / resonance band，降低原始訊號的混雜噪聲。
- HI reconstruction 可以把多特徵融合為單一健康指標，比直接 RUL regression 更容易檢查單調性與退化趨勢。
- PHM2012 的官方 failure criterion 是 acceleration exceeding 20g，適合把 HI 對齊到 failure threshold。

主要風險：

- 只有 6 顆 learning bearings，HI reconstruction 很容易對特定 bearing 過擬合。
- 不同工況與同工況內的退化模式差異都很大。
- 如果頻域前處理 band 選錯，HI 可能只反映噪聲或個別 bearing 的局部現象。
- Test set 是截斷資料，不能用 `Full_Test_Set` 做訓練或特徵選擇，否則會資料洩漏。

結論：值得實行，但要分成 baseline、檢查、再建模三步，不要一開始直接套深度模型。

## 3. 整體流程

建議流程如下：

```text
raw acc_*.csv
  -> frequency preprocessing
  -> extract 15 time-domain features
  -> normalize features
  -> feature quality evaluation
  -> HI reconstruction
  -> HI smoothing / monotonic correction
  -> degradation stage detection
  -> RUL estimation
  -> optional: feed HI/features into DAST
```

## 4. 資料使用原則

### 4.1 Learning set

用於：

- fit feature scaler。
- 選擇 frequency preprocessing 參數。
- 建立 HI reconstruction 模型。
- 建立 failure threshold。
- 建立 HI-to-RUL 或 HI trend extrapolation 模型。

Learning bearings：

```text
Bearing1_1, Bearing1_2
Bearing2_1, Bearing2_2
Bearing3_1, Bearing3_2
```

### 4.2 Test set

用於：

- online HI reconstruction。
- 根據截斷點預測 RUL。
- 與 PDF 官方 RUL 比較。

官方 RUL：

```text
Bearing1_3 5730 s
Bearing1_4 339 s
Bearing1_5 1610 s
Bearing1_6 1460 s
Bearing1_7 7570 s
Bearing2_3 7530 s
Bearing2_4 1390 s
Bearing2_5 3090 s
Bearing2_6 1290 s
Bearing2_7 580 s
Bearing3_3 820 s
```

### 4.3 Full_Test_Set

只建議用於最後 sanity check：

- 驗證 `Full_Test_Set acc_count - Test_set acc_count` 是否約等於 official RUL / 10。
- 不用於 fit scaler、選頻帶、建 HI、調參。

## 5. 頻域前處理

### 5.1 優先方案：MAS Kurtosis band-pass preprocessing

Paper 對 PHM2012 的回顧提到 champion solution 使用 moving-averaged spectral kurtosis，簡稱 MAS kurtosis，並以逐漸單調的 degradation trend 作為 indicator。

建議第一版做法：

1. 對每個 `acc_*.csv` 的水平 vibration `acc_h` 做 band-pass filtering。
2. 頻帶先使用文獻常見且與 PHM2012 solution 對應的 `5.5 kHz - 6.0 kHz`。
3. 對 filtered signal 萃取 15 個時域特徵。
4. 每個特徵序列再做 moving average，例如 window = 100 個 samples，也就是約 1000 秒。

為什麼先用水平訊號：

- 多篇 PHM2012 bearing 研究常使用 horizontal vibration，通常退化資訊較明顯。
- 減少第一版維度與雜訊。
- 若 baseline 成立，再加入 vertical channel。

### 5.2 替代方案：多頻帶候選搜尋

若固定 `5.5-6.0 kHz` 對本地資料效果不好，可建立候選頻帶：

```text
0.5-1.0 kHz
1.0-2.0 kHz
2.0-3.0 kHz
3.0-4.0 kHz
4.0-5.0 kHz
5.0-6.0 kHz
6.0-7.0 kHz
7.0-8.0 kHz
8.0-10.0 kHz
```

每個頻帶都計算 15 個特徵，然後用以下指標選 band：

- monotonicity：HI 或候選特徵是否隨壽命推進單調變化。
- trendability：不同 bearing 的趨勢是否一致。
- prognosability：failure 附近的特徵值是否集中。
- robustness：短期波動是否小。

第一版可以先固定 band，第二版再做搜尋。這樣比較不會一開始變成龐大的調參工程。

## 6. 15 個時域特徵

每個 `acc_*.csv` 經頻域前處理後，對 filtered signal `x = [x1, x2, ..., xN]` 萃取 15 個特徵。

建議使用以下組合：

| 編號 | 特徵 | 說明 |
|---:|---|---|
| 1 | mean | 平均值 |
| 2 | absolute mean | 平均絕對值 |
| 3 | standard deviation | 標準差 |
| 4 | variance | 變異數 |
| 5 | rms | 均方根 |
| 6 | square root amplitude | 方根幅值 |
| 7 | max | 最大值 |
| 8 | min | 最小值 |
| 9 | peak-to-peak | 最大最小差 |
| 10 | skewness | 偏態 |
| 11 | kurtosis | 峰態 |
| 12 | crest factor | peak / rms |
| 13 | shape factor | rms / absolute mean |
| 14 | impulse factor | peak / absolute mean |
| 15 | clearance factor | peak / square root amplitude |

其中：

```text
peak = max(abs(x))
rms = sqrt(mean(x^2))
absolute_mean = mean(abs(x))
square_root_amplitude = mean(sqrt(abs(x)))^2
```

若要同時使用 horizontal 與 vertical，則有兩種策略：

- 方案 A：只用 horizontal，feature_len = 15。
- 方案 B：horizontal + vertical 各 15 個，feature_len = 30。

建議第一版採方案 A，因為 PHM2012 訓練 bearing 太少，feature_len = 30 會增加 HI fusion 的不穩定性。

## 7. 特徵正規化

只用 Learning set fit scaler。

建議兩種都保留比較：

### 7.1 Z-score

適合 PCA / T2 / HI reconstruction：

```text
z = (x - mean_train) / std_train
```

### 7.2 MinMax

適合輸入 DAST 或 neural network：

```text
x_scaled = (x - min_train) / (max_train - min_train)
```

第一版 HI reconstruction 建議用 Z-score。

## 8. HI Index Reconstruction

### 8.1 目標

將 15 維特徵序列：

```text
F(t) = [f1(t), f2(t), ..., f15(t)]
```

融合為單一健康指標：

```text
HI(t)
```

理想 HI 應滿足：

- 初期穩定。
- 退化開始後逐漸上升或下降。
- 接近 failure 時達到 threshold。
- 不同 bearing 的 failure threshold 相對一致。
- 對短期噪聲不過度敏感。

### 8.2 Baseline 方法：PCA + T2 statistic

Paper 提到 Wang 的方法使用 PCA 與 T2 statistic 建立 health index。這很適合作為第一個可解釋 baseline。

步驟：

1. 將所有 learning bearings 的 15 維特徵合併。
2. 使用 early-life healthy samples fit PCA，例如每顆 bearing 前 10% 或前 20%。
3. 將每個時間點特徵投影到 PCA 空間。
4. 使用 Hotelling T2 作為異常程度：

```text
T2(t) = score(t)^T * Lambda^-1 * score(t)
```

5. 將 T2 正規化為 HI：

```text
HI(t) = (T2(t) - min_healthy_T2) / (failure_threshold_T2 - min_healthy_T2)
```

當 `HI = 1` 可視為 failure threshold。

優點：

- 可解釋。
- 對多特徵融合簡單。
- 不需要 RUL label 直接參與 HI 建立。

缺點：

- 若 early healthy window 選錯，PCA baseline 會偏。
- T2 對 outlier 敏感，需要 smoothing。

### 8.3 第二方法：KPCA / Autoencoder Reconstruction Error

若 PCA/T2 不夠抓非線性退化，可改用 reconstruction-based HI：

```text
train healthy feature distribution
input current feature vector
reconstruct feature vector
HI(t) = reconstruction error
```

可選方法：

- KPCA reconstruction error。
- Autoencoder reconstruction error。

建議先不要直接使用 autoencoder，因為 Learning set 太少。第二版用 KPCA 即可。

### 8.4 第三方法：Feature weighted HI

若 PCA/T2 不穩，可用人工評分挑選特徵：

```text
score(feature) = 0.4 * monotonicity + 0.3 * trendability + 0.2 * prognosability + 0.1 * robustness
```

取前 3-5 個特徵加權平均：

```text
HI(t) = sum(w_i * normalized_feature_i(t))
```

這比 PCA 更容易控制，也更適合資料少的情況。

## 9. HI 平滑與單調修正

PHM2012 的 vibration feature 短期波動很大，因此 HI 必須平滑。

建議比較：

### 9.1 Moving Average

```text
HI_smooth(t) = mean(HI[t - k + 1 : t])
```

建議 `k = 50` 或 `100`。

### 9.2 EWMA

```text
HI_smooth(t) = alpha * HI(t) + (1 - alpha) * HI_smooth(t-1)
```

建議：

```text
alpha = 0.02 到 0.05
```

### 9.3 Monotonic envelope

如果 HI 定義為越大越差，可使用：

```text
HI_mono(t) = max(HI_smooth(1), ..., HI_smooth(t))
```

這很符合「退化不可逆」的工程假設，但要小心過早 outlier 會把 HI 拉高。因此建議先 smoothing，再 monotonic envelope。

## 10. 壽命建模方式

### 10.1 Threshold crossing

最直接方法：

1. 用 Learning set 建立 failure threshold。
2. 對 test bearing 的已觀測 HI 建立趨勢模型。
3. 外推到 threshold crossing time。
4. 預測 RUL：

```text
RUL_pred = t_failure_pred - t_current
```

### 10.2 趨勢模型

建議比較三個模型：

```text
linear:      HI(t) = a * t + b
exponential: HI(t) = a * exp(b * t) + c
quadratic exponential: HI(t) = a * exp(b * t^2) + c
```

Paper 對 Sutrisno solution 的整理提到 exponential degradation model 對 MAS kurtosis 有較好 fit，因此 exponential 應列為主模型。

### 10.3 Degradation start point 偵測

不要用整段 test sequence fit 趨勢，因為健康初期 HI 幾乎水平，會拉低 slope。

建議先偵測 degradation onset：

```text
healthy_mean = mean(HI 前 10%)
healthy_std = std(HI 前 10%)
onset = first t where HI(t) > healthy_mean + 3 * healthy_std for m consecutive points
```

只用 `onset` 後的 HI fit trend。

### 10.4 Multi-stage degradation

Paper 對 PHM2012 的結論是 multi-stage degradation identification 很重要。建議第二版加入：

- Stage 1：healthy / stable
- Stage 2：incipient degradation
- Stage 3：rapid degradation

可以用 HI slope 或 change point detection 實作：

```text
slope(t) = HI_smooth(t) - HI_smooth(t-k)
if slope < threshold_1 -> Stage 1
if threshold_1 <= slope < threshold_2 -> Stage 2
if slope >= threshold_2 -> Stage 3
```

RUL 建模時：

- Stage 1：給出高不確定性的長 RUL。
- Stage 2：用 exponential fit。
- Stage 3：用 short-horizon linear/exponential fit，讓模型更快貼近 failure。

## 11. 與 DAST 模型的整合方式

### 11.1 不建議第一版直接用 DAST

DAST 的優勢是 sequence modeling，但 PHM2012 的挑戰首先是 HI 是否可信。若 HI 都不穩，DAST 只會學到噪聲。

第一版應先完成：

```text
features -> HI -> RUL
```

### 11.2 第二版：DAST 使用 feature window 預測 HI

將 DAST 目標從 RUL 改成 HI：

```text
X = 40 個連續時間點的 15 維特徵
y = 下一個時間點或 window 結尾的 HI
```

用途：

- 讓 DAST 學習 feature sequence 到 HI 的平滑映射。
- 避免直接學跨 bearing 的 RUL 差異。

### 11.3 第三版：DAST 使用 HI sequence 預測 RUL

輸入：

```text
X = 40 個連續 HI 或 [HI, slope, stage]
y = normalized RUL
```

這比直接輸入 15 features 更穩，因為 HI 已經把 bearing degradation 壓縮成單調趨勢。

## 12. 實作計畫

### 12.1 新增 preprocessing script

建議建立：

```text
FEMTO_HI_preprocess.py
```

功能：

- 掃描 `Learning_set`、`Test_set`。
- 讀取每個 bearing 的 `acc_*.csv`。
- 對 horizontal signal 做 band-pass filtering。
- 萃取 15 個時域特徵。
- fit scaler。
- 儲存 feature matrix。

輸出：

```text
train_dataset/femto_hi/
  learning_features.npz
  test_features.npz
  scaler.pkl
  metadata.csv
```

### 12.2 新增 HI reconstruction script

建議建立：

```text
FEMTO_HI_reconstruct.py
```

功能：

- 載入 feature matrix。
- PCA/T2 建 HI。
- smoothing。
- monotonic envelope。
- 計算 feature/HI quality metrics。
- 輸出 HI curve plot。

輸出：

```text
train_dataset/femto_hi/
  hi_learning.csv
  hi_test.csv
  pca_model.pkl
  hi_quality_report.csv
plots/femto_hi/
  Bearing1_1_HI.png
  ...
```

### 12.3 新增 RUL estimation script

建議建立：

```text
FEMTO_HI_rul_estimate.py
```

功能：

- 對每顆 test bearing 讀取已觀測 HI。
- 偵測 degradation onset。
- fit linear / exponential trend。
- 外推到 threshold。
- 輸出 11 顆 bearing 的 RUL prediction。

輸出：

```text
PHM2012_HI_RUL_results.csv
plots/femto_hi_rul/
  Bearing1_3_RUL_fit.png
  ...
```

### 12.4 可選 DAST dataset export

若 HI baseline 成立，再建立：

```text
FEMTO_HI_to_DAST_dataset.py
```

輸出：

```text
FEMTO_HI_window_size_trainX_new.mat
FEMTO_HI_window_size_trainY.mat
FEMTO_HI_window_size_testX_new.mat
FEMTO_HI_window_size_testY.mat
```

其中 feature 可選：

- `15 features`
- `15 features + HI`
- `HI + HI slope + stage`

## 13. 評估指標

### 13.1 HI 品質

每顆 bearing 計算：

- monotonicity
- trendability
- prognosability
- robustness
- failure threshold consistency

最重要的是畫圖檢查：

```text
raw feature curves
PCA/T2 HI
smoothed HI
monotonic HI
failure threshold
```

### 13.2 RUL 評估

使用：

- absolute error seconds
- percent error
- RMSE seconds
- MAE seconds
- PHM2012 official score

PHM2012 scoring：

```text
Er = 100 * (true_RUL - pred_RUL) / true_RUL
if Er <= 0:
    A = exp(-ln(0.5) * Er / 5)
else:
    A = exp( ln(0.5) * Er / 20)
Score = mean(A)
```

官方 score 越接近 1 越好。

## 14. 實驗順序

### Experiment 1：固定頻帶 + PCA/T2

- horizontal only
- band-pass 5.5-6.0 kHz
- 15 time-domain features
- Z-score
- PCA fit early-life healthy data
- T2 as HI
- EWMA + monotonic envelope
- exponential threshold crossing

這是第一個 baseline。

### Experiment 2：固定頻帶 + weighted feature HI

- 同 Experiment 1
- 不用 PCA
- 用 monotonicity / trendability 評分挑前 3-5 features
- weighted sum 建 HI

若 PCA/T2 過度受 outlier 影響，此法可能更穩。

### Experiment 3：多頻帶搜尋

- 搜尋不同 band-pass frequency。
- 每個 band 重跑 HI quality。
- 選出最穩定的 band。

### Experiment 4：horizontal + vertical

- 若 horizontal baseline 成立，加入 vertical。
- feature_len = 30。
- 比較 HI quality 與 RUL error 是否改善。

### Experiment 5：DAST with HI

- 使用 `15 features + HI` 或 `HI + slope + stage`。
- 預測 normalized RUL 或 future HI。
- 與 threshold-crossing baseline 比較。

## 15. 預期結果與判斷標準

方法值得繼續的條件：

- 至少 4/6 顆 learning bearings 的 HI 呈現明顯退化趨勢。
- failure 附近 HI threshold 大致一致。
- 11 顆 test bearings 的 RUL RMSE 明顯優於 naive baseline。
- 對短 RUL bearing，例如 `Bearing1_4`、`Bearing2_7`、`Bearing3_3`，能在截斷點附近反映已退化狀態。

應調整方法的情況：

- HI 在 healthy phase 就大量上升：smoothing 或 band 有問題。
- 同工況 bearing 的 HI 方向不一致：feature sign / normalization / PCA fit 需調整。
- long RUL bearing 被預測太短：onset detection 太敏感。
- short RUL bearing 被預測太長：HI 對 rapid degradation 不敏感，需要改 band 或加入 vertical。

## 16. 建議的最小可行版本

第一階段只做以下內容：

1. 使用 horizontal vibration。
2. band-pass `5.5-6.0 kHz`。
3. 萃取 15 個時域特徵。
4. Z-score 正規化。
5. PCA/T2 建 HI。
6. EWMA smoothing。
7. monotonic envelope。
8. exponential trend extrapolation 到 failure threshold。
9. 輸出 11 顆 test bearing 的 RUL 與 PHM2012 score。

這個版本能快速回答核心問題：

```text
15 time-domain features + frequency preprocessing 是否能重建出足夠穩定的 HI？
HI-based RUL 是否比直接 feature-to-RUL regression 更適合 PHM2012？
```

若答案是肯定，再把 HI sequence 接到 DAST。若答案是否定，應先修正 feature/band/HI construction，而不是增加模型複雜度。
