# condition_slope_feature_compare

工況斜率特徵比較：global／condfe／none／shuffle。

本次歸類 183 個歷史檔案（含外部 logs／分析程式關聯）；不等於独立 run 數。

| 作用 | 檔案數 |
|---|---:|
| checkpoints | 120 |
| epoch_metrics | 60 |
| notes/source_snapshot | 1 |
| results | 2 |

## 查找與回溯

- [file_mapping.csv](../../docs/organization_2026-10-07/file_mapping.csv)：以 experiment=`condition_slope_feature_compare` 篩選。
- [run_catalog.csv](../../docs/organization_2026-10-07/run_catalog.csv)：以 experiment=`condition_slope_feature_compare` 篩選。
- [unknown_and_review.csv](../../docs/organization_2026-10-07/unknown_and_review.csv)：以 experiment=`condition_slope_feature_compare` 篩選。
- [歷史 log／epoch 指標](../../logs/condition_slope_feature_compare)，依日期及來源分類；undated 表示無可靠日期。
- [checkpoints/](checkpoints/)
- [notes/](notes/)
- [results/](results/)

`local`、`server_mirror`、`server_experiments`、`server_snapshot` 是來源識別。歷史設定原文不改；`configs/relocated/` 是另存的路徑更新版。

新 C-MAPSS 執行由根目錄 `DAST_test.py`／`run_cmapss_sweep.py` 啟動；`notes/source_snapshot/` 只保存歷史來源，不是可直接啟動的入口。歷史缺少 commit、資料生成版本或 split 的欄位仍待補。
