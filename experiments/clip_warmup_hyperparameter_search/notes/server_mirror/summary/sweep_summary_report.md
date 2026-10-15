# DAST C-MAPSS Sweep 實驗結果整理

## 資料來源與範圍

- 來源檔案：`server_results/紀錄/experiment_log.csv`
- 分析筆數：正式 sweep 共 144 筆
- 排除資料：舊紀錄與 1 epoch smoke test 共 51 筆
- 實驗範圍：
  - Dataset：FD001、FD002、FD003、FD004
  - Gradient clipping threshold：0.5、1.0、2.0、5.0
  - Warmup epochs：5、10、15
  - Seeds：20、42、100

本報告以四種核心角度整理結果：

1. 各 dataset 最佳設定表
2. Clip threshold 主效果分析
3. Warmup epoch 主效果分析
4. Seed mean +/- std 穩定性分析

綜合排名方法：先在每個 dataset 內分別依 `mean RMSE` 與 `mean Score` 排名，再取兩者平均；最後跨 dataset 取平均。這樣可以避免 FD004 的 Score 尺度較大而支配整體結論。

---

## 1. 綜合平均最佳設定

跨 FD001-FD004 綜合比較後，最佳設定為：

```text
clip = 5.0
warmup_epochs = 15
```

| Rank | Clip | Warmup | Avg Combined Rank | Avg RMSE Rank | Avg Score Rank | Mean RMSE | Mean Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 | 15 | 3.250 | 2.250 | 4.250 | 12.503 | 532.520 |
| 2 | 1.0 | 10 | 3.625 | 4.000 | 3.250 | 12.559 | 523.511 |
| 3 | 2.0 | 10 | 4.125 | 4.250 | 4.000 | 12.562 | 532.556 |
| 4 | 1.0 | 15 | 5.375 | 6.250 | 4.500 | 12.658 | 534.782 |
| 5 | 2.0 | 15 | 6.250 | 5.500 | 7.000 | 12.543 | 570.140 |
| 6 | 5.0 | 10 | 6.375 | 5.250 | 7.500 | 12.581 | 565.280 |
| 7 | 0.5 | 10 | 7.250 | 8.000 | 6.500 | 12.705 | 563.412 |
| 8 | 0.5 | 5 | 7.625 | 9.000 | 6.250 | 12.772 | 530.424 |
| 9 | 5.0 | 5 | 7.750 | 7.000 | 8.500 | 12.637 | 557.549 |
| 10 | 2.0 | 5 | 8.625 | 8.750 | 8.500 | 12.774 | 590.867 |
| 11 | 0.5 | 15 | 8.875 | 8.750 | 9.000 | 12.736 | 577.800 |
| 12 | 1.0 | 5 | 8.875 | 9.000 | 8.750 | 12.833 | 599.881 |

解讀：

- `clip=5.0, warmup=15` 的綜合平均排名最好，代表較寬鬆的 gradient clipping 搭配較長 warmup 整體最穩。
- `clip=1.0, warmup=10` 的平均 Score 最低，若研究重點放在 NASA Score 或 late prediction 風險，可列為第二候選。
- `clip=0.5` 整體排名偏後，表示 clipping 太緊可能限制模型更新。

---

## 2. 各 Dataset 最佳設定

| Dataset | Best Clip | Best Warmup | Seeds | Mean RMSE | RMSE Std | Mean Score | Score Std | Mean MAE | Mean Best Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 1.0 | 5 | 3 | 12.177 | 0.268 | 229.415 | 15.083 | 8.763 | 26.000 |
| FD002 | 5.0 | 15 | 3 | 13.085 | 0.398 | 765.596 | 122.802 | 9.590 | 6.667 |
| FD003 | 2.0 | 15 | 3 | 10.984 | 0.963 | 197.553 | 52.932 | 8.020 | 73.000 |
| FD004 | 5.0 | 15 | 3 | 13.351 | 0.103 | 917.743 | 146.904 | 9.284 | 7.000 |

解讀：

- FD001 的最佳組合是 `clip=1.0, warmup=5`，不需要很寬鬆的 clipping 或很長 warmup。
- FD002 與 FD004 都偏好 `clip=5.0, warmup=15`，表示多工況資料集可能較需要寬鬆 clipping。
- FD003 最佳為 `clip=2.0, warmup=15`，與 FD004 不完全相同，顯示穩定化技巧有 dataset-dependent 特性。

---

## 3. Clip Threshold 主效果

| Clip | Avg Combined Rank | Mean RMSE | Mean Score |
|---:|---:|---:|---:|
| 5.0 | 1.875 | 12.574 | 551.783 |
| 2.0 | 2.250 | 12.627 | 564.521 |
| 1.0 | 2.750 | 12.683 | 552.725 |
| 0.5 | 3.125 | 12.738 | 557.212 |

解讀：

- `clip=5.0` 是整體主效果最佳的 threshold。
- `clip=2.0` 排名第二，仍屬於可保留的候選。
- `clip=1.0` 的 Score 不差，但 RMSE 排名較弱。
- `clip=0.5` 排名最後，推測太強的 clipping 會讓模型更新幅度不足。

研究結論可以寫成：

```text
Gradient clipping 並非越強越好；在本次 sweep 中，較寬鬆的 max_norm=5.0 取得最佳整體表現，顯示先前固定 max_norm=1.0 可能偏緊。
```

---

## 4. Warmup Epoch 主效果

| Warmup Epochs | Avg Combined Rank | Mean RMSE | Mean Score |
|---:|---:|---:|---:|
| 10 | 1.375 | 12.602 | 546.190 |
| 15 | 1.875 | 12.610 | 553.811 |
| 5 | 2.750 | 12.754 | 569.680 |

解讀：

- 單看 warmup 主效果，`warmup=10` 最佳。
- `warmup=15` 與 `warmup=10` 的 Mean RMSE 接近，但 Mean Score 較高。
- `warmup=5` 整體最弱，表示 warmup 太短時訓練穩定化不足。

需要注意的是，主效果不等於最佳組合。雖然 warmup 主效果以 10 epochs 最好，但與 clip threshold 搭配後，綜合最佳組合仍是 `clip=5.0, warmup=15`。

---

## 5. Seed 穩定性分析

下表列出每個 dataset 中 RMSE 標準差較小的前三組設定。這代表同一組設定在 seed=20、42、100 下比較穩定。

| Dataset | Clip | Warmup | Seeds | RMSE Mean | RMSE Std | Score Mean | Score Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | 1.0 | 15 | 3 | 12.431 | 0.068 | 260.021 | 16.666 |
| FD001 | 1.0 | 10 | 3 | 12.306 | 0.071 | 229.410 | 13.851 |
| FD001 | 5.0 | 10 | 3 | 12.166 | 0.097 | 241.829 | 5.122 |
| FD002 | 1.0 | 15 | 3 | 13.172 | 0.130 | 747.783 | 69.784 |
| FD002 | 1.0 | 10 | 3 | 13.186 | 0.188 | 781.150 | 84.350 |
| FD002 | 2.0 | 15 | 3 | 13.136 | 0.246 | 820.623 | 90.156 |
| FD003 | 0.5 | 5 | 3 | 11.590 | 0.086 | 221.716 | 27.566 |
| FD003 | 2.0 | 5 | 3 | 11.540 | 0.086 | 207.273 | 19.061 |
| FD003 | 5.0 | 5 | 3 | 11.634 | 0.175 | 226.034 | 8.209 |
| FD004 | 1.0 | 10 | 3 | 13.421 | 0.085 | 877.428 | 104.241 |
| FD004 | 5.0 | 15 | 3 | 13.351 | 0.103 | 917.743 | 146.904 |
| FD004 | 0.5 | 15 | 3 | 13.527 | 0.104 | 998.249 | 158.743 |

解讀：

- FD004 最穩定的是 `clip=1.0, warmup=10`，但最佳綜合表現是 `clip=5.0, warmup=15`。
- 這代表 FD004 有一個取捨：`clip=5.0, warmup=15` 的平均效果較好，但 `clip=1.0, warmup=10` 的 seed 穩定性與 Score 更有吸引力。
- FD003 的穩定設定不等於最佳設定，表示低變異不一定代表最低 RMSE / Score。

---

## 6. 總結與建議

本次 144 筆正式 sweep 顯示：

1. 綜合 RMSE 與 NASA Score，最佳設定是 `clip=5.0, warmup=15`。
2. 若特別重視 NASA Score，`clip=1.0, warmup=10` 是重要候選。
3. `clip=0.5` 整體偏弱，表示 clipping 太緊可能不利於模型更新。
4. Warmup 不宜太短，`warmup=10` 或 `15` 都明顯優於 `5`。
5. FD002 與 FD004 較偏好寬鬆 clipping，符合多工況資料集需要更大更新彈性的直覺。

建議後續研究採用兩條主線：

| 目的 | 建議設定 | 理由 |
|---|---|---|
| 綜合最佳模型 | `clip=5.0, warmup=15` | 跨 dataset 綜合排名第一，FD002/FD004 表現尤其好 |
| Score / 穩定性候選 | `clip=1.0, warmup=10` | 平均 Score 最低，FD004 seed 穩定性較佳 |

可放入簡報或論文的結論文字：

```text
本研究進一步以 4 個 C-MAPSS 子資料集、4 組 gradient clipping threshold、3 組 warmup epochs 與 3 個 random seeds 進行 144 筆 sweep。結果顯示，gradient clipping 並非越嚴格越好；較寬鬆的 max_norm=5.0 搭配 15 epochs warmup 在 RMSE 與 NASA Score 的綜合排名中最佳。若單獨考量 NASA Score 與 seed 穩定性，max_norm=1.0 搭配 10 epochs warmup 亦是值得保留的候選設定。整體而言，warmup 與 clipping 的效果具有 dataset-dependent 特性，尤其在 FD002 與 FD004 等多工況資料集上，較寬鬆的 clipping 更有利於模型取得穩定表現。
```
