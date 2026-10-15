# fd004_operating_condition_analysis

FD004 工況、感測器與退化分析。

本次歸類 90 個歷史檔案（含外部 logs／分析程式關聯）；不等於独立 run 數。

| 作用 | 檔案數 |
|---|---:|
| analysis_code | 11 |
| figures | 59 |
| notes | 18 |
| results | 2 |

## 查找與回溯

- [file_mapping.csv](../../docs/organization_2026-10-07/file_mapping.csv)：以 experiment=`fd004_operating_condition_analysis` 篩選。
- [run_catalog.csv](../../docs/organization_2026-10-07/run_catalog.csv)：以 experiment=`fd004_operating_condition_analysis` 篩選。
- [unknown_and_review.csv](../../docs/organization_2026-10-07/unknown_and_review.csv)：以 experiment=`fd004_operating_condition_analysis` 篩選。
- [figures/](figures/)
- [notes/](notes/)
- [results/](results/)
- [報告閱讀副本：analysis_report_relocated.md](notes/local/analysis/analysis_report_relocated.md)
- [報告閱讀副本：analysis_report_v1_relocated.md](notes/local/analysis/analysis_report_v1_relocated.md)
- [報告閱讀副本：analysis_report_relocated.md](notes/server_snapshot/analysis/analysis_report_relocated.md)
- [報告閱讀副本：analysis_report_v1_relocated.md](notes/server_snapshot/analysis/analysis_report_v1_relocated.md)

`local`、`server_mirror`、`server_experiments`、`server_snapshot` 是來源識別。歷史設定原文不改；`configs/relocated/` 是另存的路徑更新版。

新 C-MAPSS 執行由根目錄 `DAST_test.py`／`run_cmapss_sweep.py` 啟動；`notes/source_snapshot/` 只保存歷史來源，不是可直接啟動的入口。歷史缺少 commit、資料生成版本或 split 的欄位仍待補。
