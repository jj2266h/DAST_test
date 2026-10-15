# cmapss_no_clip_no_warmup

關閉 clipping 與 warmup 的 C-MAPSS 執行。

本次歸類 45 個歷史檔案（含外部 logs／分析程式關聯）；不等於独立 run 數。

| 作用 | 檔案數 |
|---|---:|
| checkpoints | 9 |
| configs | 4 |
| epoch_metrics | 4 |
| figures | 24 |
| log | 4 |

## 查找與回溯

- [file_mapping.csv](../../docs/organization_2026-10-07/file_mapping.csv)：以 experiment=`cmapss_no_clip_no_warmup` 篩選。
- [run_catalog.csv](../../docs/organization_2026-10-07/run_catalog.csv)：以 experiment=`cmapss_no_clip_no_warmup` 篩選。
- [unknown_and_review.csv](../../docs/organization_2026-10-07/unknown_and_review.csv)：以 experiment=`cmapss_no_clip_no_warmup` 篩選。
- [歷史 log／epoch 指標](../../logs/cmapss_no_clip_no_warmup)，依日期及來源分類；undated 表示無可靠日期。
- [checkpoints/](checkpoints/)
- [configs/](configs/)
- [figures/](figures/)

`local`、`server_mirror`、`server_experiments`、`server_snapshot` 是來源識別。歷史設定原文不改；`configs/relocated/` 是另存的路徑更新版。

新 C-MAPSS 執行由根目錄 `DAST_test.py`／`run_cmapss_sweep.py` 啟動；`notes/source_snapshot/` 只保存歷史來源，不是可直接啟動的入口。歷史缺少 commit、資料生成版本或 split 的欄位仍待補。
