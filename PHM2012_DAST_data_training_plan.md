# PHM2012 / FEMTO 軸承資料轉換與 DAST 訓練計畫

## 1. 目標

本計畫的目標，是把 `phm2012_dataset/` 內的 IEEE PHM 2012 Data Challenge 軸承資料，轉成目前專案中的 DAST 模型可以直接訓練的格式。

目前 DAST 流程主要沿用 C-MAPSS：

- `data_process.py` 產生 `.mat` 檔。
- `DAST_test.py` 載入 `*_window_size_trainX_new.mat`、`*_window_size_trainY.mat`、`*_window_size_testX_new.mat`、`*_window_size_testY.mat`。
- 模型輸入形狀為 `(samples, time_step, feature_len)`。
- `trainX_new` 會在原本 sliding window 後追加兩個統計 time step，因此實際 `time_step = window_size + 2`。

PHM2012 原始資料與 C-MAPSS 不同：它不是每個 cycle 一列多感測器，而是每 10 秒記錄一段 0.1 秒的高頻振動訊號。因此關鍵工作是先把每個 `acc_*.csv` 壓縮成一筆「時間步特徵」，再用 bearing 生命序列做 sliding window。

## 2. 資料理解

### 2.1 官方資料規格

根據 `phm2012_dataset/IEEEPHM2012-Challenge-Details.pdf`：

- 平台：PRONOSTIA 軸承加速壽命測試平台。
- 任務：預測 rolling bearing 的 Remaining Useful Life，簡稱 RUL。
- RUL 定義：到加速度超過 20g 為止的剩餘時間。
- 工況共有三組：
  - Condition 1：1800 rpm，4000 N
  - Condition 2：1650 rpm，4200 N
  - Condition 3：1500 rpm，5000 N
- 振動訊號：
  - 水平與垂直兩軸。
  - 取樣頻率 25.6 kHz。
  - 每 10 秒記錄一次，每次 2560 samples，也就是 0.1 秒。
  - `acc_*.csv` 欄位為 `hour, minute, second, micro-second, horizontal_acc, vertical_acc`。
- 溫度訊號：
  - 取樣頻率 10 Hz。
  - 每分鐘記錄 600 samples。
  - `temp_*.csv` 欄位為 `hour, minute, second, 0.x_second, temperature`。

### 2.2 資料集切分

Learning set 是完整 run-to-failure，可用來訓練：

| Bearing | 工況 | acc 檔數 | temp 檔數 | 說明 |
|---|---:|---:|---:|---|
| Bearing1_1 | 1 | 2803 | 466 | vibration + temperature |
| Bearing1_2 | 1 | 871 | 144 | vibration + temperature |
| Bearing2_1 | 2 | 911 | 151 | vibration + temperature |
| Bearing2_2 | 2 | 797 | 0 | vibration only |
| Bearing3_1 | 3 | 515 | 89 | vibration + temperature |
| Bearing3_2 | 3 | 1637 | 0 | vibration only |

Test set 是截斷序列，官方 PDF 給出真實 RUL，可用來驗證：

| Bearing | 工況 | Test acc 檔數 | Full acc 檔數 | 官方 RUL |
|---|---:|---:|---:|---:|
| Bearing1_3 | 1 | 1802 | 2375 | 5730 s |
| Bearing1_4 | 1 | 1139 | 1428 | 339 s |
| Bearing1_5 | 1 | 2302 | 2463 | 1610 s |
| Bearing1_6 | 1 | 2302 | 2448 | 1460 s |
| Bearing1_7 | 1 | 1502 | 2259 | 7570 s |
| Bearing2_3 | 2 | 1202 | 1955 | 7530 s |
| Bearing2_4 | 2 | 612 | 751 | 1390 s |
| Bearing2_5 | 2 | 2002 | 2311 | 3090 s |
| Bearing2_6 | 2 | 572 | 701 | 1290 s |
| Bearing2_7 | 2 | 172 | 230 | 580 s |
| Bearing3_3 | 3 | 352 | 434 | 820 s |

注意：本資料夾內的 `Test_set` 檔案數與 PDF 有些 bearing 不完全一致，但 `Full_Test_Set` 與官方 RUL 大致可互相驗證，因為每個 `acc_*.csv` 約代表 10 秒，`Full_Test_Set acc 檔數 - Test_set acc 檔數` 約等於官方 RUL 秒數除以 10。

## 3. 建議轉換策略

### 3.1 不直接餵入原始 2560 點訊號

原始 `acc_*.csv` 每筆有 2560 個時間點、2 軸訊號。如果直接把 40 個檔案串成模型輸入，單一 window 會變成 `40 * 2560 * 2` 的高維資料，與目前 DAST 架構不合，而且資料量與記憶體壓力會很高。

建議先把每個 `acc_*.csv` 轉成一列特徵，例如：

```text
acc_00001.csv -> feature vector
acc_00002.csv -> feature vector
...
```

接著每 40 個連續檔案形成一個 DAST training sample：

```text
sample X = [feature_t, feature_t+1, ..., feature_t+39]
sample y = normalized RUL at t+39
```

### 3.2 每個 acc 檔案的特徵設計

第一版建議使用穩定、低風險的 time-domain + frequency-domain 統計特徵。每個檔案有水平與垂直兩軸，可對兩軸各自計算：

Time-domain：

- mean
- std
- rms
- max
- min
- peak-to-peak
- skewness
- kurtosis
- crest factor
- absolute mean
- impulse factor
- shape factor

Frequency-domain：

- spectral centroid
- spectral rms
- dominant frequency
- band energy ratio，例如 0-1kHz、1-3kHz、3-6kHz、6-10kHz

若每軸取 8 到 12 個特徵，雙軸約 16 到 24 維。`config.json` 目前 `femto.feature_len` 是 16，可以先從 16 維版本開始，讓 DAST 模型維度簡單可控。

建議第一版 16 維如下：

| 特徵 | horizontal | vertical |
|---|---:|---:|
| mean | 1 | 1 |
| std | 1 | 1 |
| rms | 1 | 1 |
| max_abs | 1 | 1 |
| peak_to_peak | 1 | 1 |
| skewness | 1 | 1 |
| kurtosis | 1 | 1 |
| crest_factor | 1 | 1 |

合計 16 維。

第二版再加入 FFT 頻域特徵，將 `feature_len` 擴到 24 或 32。先不要一開始就做太多，因為資料只有 6 顆完整訓練軸承，過高維度容易過擬合。

### 3.3 溫度資料處理

第一版建議不要把 temperature 納入主模型，原因：

- Learning set 與 Test set 都有部分 bearing 沒有 temperature。
- vibration 是所有 bearing 都有的共同訊號。
- DAST 第一版需要先建立穩定 baseline。

第二版可將溫度作為 optional feature：

- 依時間對齊到每個 acc index。
- 沒有 temperature 的 bearing 用 NaN mask、前向填補或 training-set median 填補。
- 新增 `use_temperature` config 開關。

## 4. Label 設計

### 4.1 Learning set label

Learning set 是完整壽命資料，所以第 `i` 個 acc 檔案的 RUL 可定義為：

```text
rul_seconds_i = (num_acc_files - 1 - i) * 10
```

若 window size 為 40，sample 使用第 `start` 到 `start + 39` 個檔案，label 應對應 window 結尾：

```text
end_idx = start + window_size - 1
rul_seconds = (num_acc_files - 1 - end_idx) * 10
```

### 4.2 Test set label

若只評估每顆 Test bearing 的最後一個 window，label 直接使用官方 PDF RUL：

```text
Bearing1_3 -> 5730 s
Bearing1_4 -> 339 s
...
```

若要做完整序列評估，也可根據官方 RUL 回推每個 sliding window 的 label：

```text
rul_seconds_at_end_idx = official_rul_seconds + (num_acc_files - 1 - end_idx) * 10
```

這樣可以得到 `testX_all` / `testY_all`，類似目前 C-MAPSS 的 `testX_all` 概念。

### 4.3 RUL 上限與正規化

C-MAPSS 目前使用 `rul_max = 125` 並做 piecewise clipping。PHM2012 單位是秒，壽命範圍約數百秒到數萬秒；若直接回歸秒數，數值尺度太大。

建議：

- `rul_unit = "seconds"`，但訓練前正規化。
- 第一版 `rul_max = 20000.0` 秒，約 5.56 小時。
- label：

```text
y = min(rul_seconds, rul_max) / rul_max
```

這與現有 DAST 的輸出 `[0, 1]` 回歸形式一致。

也可以嘗試用「10 秒為一 cycle」：

```text
rul_cycles = rul_seconds / 10
rul_max = 2000 cycles
```

兩者等價。為了和官方 RUL 對照清楚，建議內部保留秒數 metadata，訓練 label 則 normalized。

## 5. Sliding Window 產出格式

建議輸出與現有 C-MAPSS 管線一致的 `.mat`：

```text
train_dataset/
  FEMTO_window_size_trainX.mat       key: train1X
  FEMTO_window_size_trainY.mat       key: train1Y
  FEMTO_window_size_testX.mat        key: test1X
  FEMTO_window_size_testY.mat        key: test1Y
  FEMTO_window_size_trainX_new.mat   key: train1X_new
  FEMTO_window_size_testX_new.mat    key: test1X_new
```

形狀：

```text
train1X      = (N_train_windows, 40, 16)
train1Y      = (N_train_windows,)
test1X       = (11, 40, 16)
test1Y       = (11,)
train1X_new  = (N_train_windows, 42, 16)
test1X_new   = (11, 42, 16)
```

其中 `+2` 的兩個 time step 延續目前程式邏輯：

- 第 41 個 time step：每個 feature 在該 window 內的 linear trend slope。
- 第 42 個 time step：每個 feature 在該 window 內的 mean。

## 6. 前處理流程設計

建議建立或完成 `FEMTO_datapreprocess.py`，流程如下。

### 6.1 掃描資料

輸入：

```text
phm2012_dataset/
  Learning_set/
  Test_set/
  Full_Test_Set/
```

建立 metadata：

```text
bearing_id, split, condition, acc_count, temp_count, official_rul_seconds
```

工況可由 bearing 名稱判斷：

```text
Bearing1_* -> condition 1
Bearing2_* -> condition 2
Bearing3_* -> condition 3
```

### 6.2 讀取 acc 檔案

每個 `acc_*.csv` 無 header，讀取欄位：

```text
hour, minute, second, microsecond, acc_h, acc_v
```

只取 `acc_h`、`acc_v`。

應加入基本防呆：

- 檢查每個 acc 檔案是否約 2560 rows。
- 若 rows 不等於 2560，仍可計算統計特徵，但記錄 warning。
- 檢查 NaN / inf，必要時以 0 或該檔案均值替代。

### 6.3 單檔特徵萃取

對每個 `acc_*.csv` 得到一列 feature：

```text
bearing_feature_matrix.shape = (num_acc_files, feature_len)
```

第一版 feature_len = 16。

### 6.4 正規化

正規化必須只 fit Learning set，避免資料洩漏：

```text
scaler.fit(all_learning_features)
learning_features = scaler.transform(...)
test_features = scaler.transform(...)
```

建議使用 `MinMaxScaler`，與現有 C-MAPSS 程式一致。若 outlier 影響嚴重，第二版可改 `StandardScaler` 或 `RobustScaler`。

### 6.5 產生 train windows

對 6 顆 Learning bearings 分別做 sliding window：

```text
for each bearing:
    for start in range(0, num_acc_files - window_size + 1):
        end = start + window_size
        X.append(features[start:end])
        y.append(min(rul_seconds_at_end, rul_max) / rul_max)
```

不要讓 window 跨 bearing，因為每顆 bearing 是不同壽命序列。

### 6.6 產生 test windows

第一版與目前 C-MAPSS testX 類似：每顆 Test bearing 只取最後 `window_size` 筆。

若 Test bearing 長度小於 `window_size`，使用 interpolation 或前端 padding。PHM2012 的 test set 最短 `Bearing2_7` 有 172 筆 acc，若 `window_size = 40`，不會遇到不足問題。

```text
testX.append(features[-window_size:])
testY.append(official_rul_seconds / rul_max after clipping)
```

## 7. 模型與 config 調整

### 7.1 config 建議

`config.json` 已有 `femto` 區塊，可調整為：

```json
{
  "dataset_type": "femto",
  "femto": {
    "femto_path": "phm2012_dataset",
    "output_path": "train_dataset",
    "dataset": "FEMTO",
    "window_size": 40,
    "rul_max": 20000.0,
    "feature_len": 16,
    "use_temperature": false
  }
}
```

### 7.2 訓練程式調整

`DAST_test.py` 目前固定讀 `config["cmapss"]`。建議改成根據 `dataset_type` 選擇：

```python
dataset_type = config["dataset_type"]
_ds = config[dataset_type]
DATASET = _ds["dataset"]
RUL_max = _ds["rul_max"]
traindata_dir = _ds["output_path"]
```

這樣 C-MAPSS 與 FEMTO 可以共用同一套訓練入口。

### 7.3 評估指標

保留目前：

- RMSE
- score
- prediction vs true plot

但 PHM2012 官方 score 與目前 NASA scoring 不同，建議新增 `calculate_phm2012_score(pred_seconds, true_seconds)`：

```text
Er = 100 * (true - pred) / true
if Er <= 0:
    A = exp(-ln(0.5) * (Er / 5))
else:
    A = exp(+ln(0.5) * (Er / 20))
Score = mean(A)
```

注意官方 score 越接近 1 越好；目前 NASA score 是越低越好，兩者不要混用。

## 8. 實驗設計

### 8.1 Baseline 1：全部工況混合訓練

使用 6 顆 Learning bearings 全部訓練，11 顆 Test bearings 驗證。

優點：

- 資料量最大。
- 可快速確認整條管線能跑。

風險：

- 三種工況分布不同，模型可能學到混雜關係。

### 8.2 Baseline 2：加入 condition feature

在每個 time step 的 feature 後追加 one-hot condition：

```text
condition 1 -> [1, 0, 0]
condition 2 -> [0, 1, 0]
condition 3 -> [0, 0, 1]
```

若原 feature_len = 16，加入後變成 19。

### 8.3 Baseline 3：分工況訓練

分別訓練：

- Condition 1：Bearing1_1, Bearing1_2 -> test Bearing1_3 到 Bearing1_7
- Condition 2：Bearing2_1, Bearing2_2 -> test Bearing2_3 到 Bearing2_7
- Condition 3：Bearing3_1, Bearing3_2 -> test Bearing3_3

優點是同工況資料較一致；缺點是每個模型訓練資料更少。

### 8.4 建議優先順序

1. Baseline 1：vibration 16 features，全部工況混合。
2. Baseline 2：vibration 16 features + condition one-hot。
3. Baseline 3：分工況模型。
4. 加入 FFT features。
5. 加入 temperature optional features。

## 9. 驗證檢查清單

前處理完成後，必須檢查：

- `train1X.shape[1] == window_size`
- `train1X_new.shape[1] == window_size + 2`
- `train1X.shape[2] == feature_len`
- `train1Y.min() >= 0`
- `train1Y.max() <= 1`
- `test1X.shape[0] == 11`
- `test1Y.shape[0] == 11`
- 任一 window 不可跨 bearing。
- scaler 只能 fit Learning set。
- Test label 使用 PDF 的官方 RUL，不使用 `Full_Test_Set` 直接偷看 label。

訓練完成後，建議輸出：

- 11 顆 test bearing 的 true RUL 秒數。
- 11 顆 test bearing 的 predicted RUL 秒數。
- RMSE seconds。
- PHM2012 official score。
- 每顆 bearing 的 percent error。

## 10. 預期實作檔案

建議修改或新增：

```text
FEMTO_datapreprocess.py
  - 讀取 PHM2012 原始資料
  - 萃取 acc features
  - 產生 train/test .mat

DAST_test.py
  - 支援 dataset_type = "cmapss" 或 "femto"
  - 新增 PHM2012 score

config.json
  - 修正 femto_path
  - 加入 dataset、rul_max、use_temperature
```

可選新增：

```text
DAST_femto_train.py
  - 若不想動現有 C-MAPSS 訓練程式，可獨立建立 FEMTO 訓練入口

plots/femto/
  - 存放 FEMTO 訓練結果圖
```

## 11. 最小可行版本

最小可行版本應完成：

1. 只用 `acc_*.csv`。
2. 每個 acc 檔案萃取 16 維振動統計特徵。
3. window size = 40。
4. RUL label 單位使用秒，`rul_max = 20000.0`，訓練前 normalize 到 0 到 1。
5. 產出與現有 DAST 相同命名邏輯的 `.mat`。
6. 讓 `DAST_test.py` 可以直接訓練 FEMTO。
7. 評估 11 顆 Test bearings 的 RMSE 與 PHM2012 score。

這個版本可以先回答最重要的問題：目前 DAST 架構是否能從 PHM2012 的 vibration degradation pattern 中學到可用的 RUL 表徵。等 baseline 穩定後，再逐步加入 condition、FFT、temperature 與分工況模型。
