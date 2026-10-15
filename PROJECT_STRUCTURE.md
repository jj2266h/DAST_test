# DAST 專案結構與回溯入口

2026-10-07 依用途整理；核心 architecture、train、test、preprocessing 保留根目錄。未更動模型架構、loss、資料切分或選模規則。

| 位置 | 用途 |
|---|---|
| `DAST_Network.py`、`DAST_utils.py` | 共用模型與 attention |
| `DAST_test.py`、`test.py` | C-MAPSS 訓練／評估 |
| `data_process.py` | C-MAPSS 前處理，含統計特徵，不需再執行舊統計特徵腳本 |
| `run_cmapss_sweep.py` | sweep 入口，已由 experiments/ 提升根目錄 |
| `DAST_FEMTO_train.py`、`FEMTO_datapreprocess.py` | FEMTO 流程 |
| `DAST_final.py`、`DAST_main.py`、`Statistical features process .py` | 保留的舊流程；仍有原有輸入佔位與配對限制 |
| `analysis/` | FD004 專用分析程式；新輸出進對應 experiments |
| `data/raw/cmapss/local/` | 本機原始 C-MAPSS |
| `data/processed/` | cmapss_current、cmapss_legacy、femto 分開 |
| `phm2012_dataset/` | 原位保留的 PHM2012 資料儲存庫 |
| [experiments/](experiments/README.md) | 每個用途下 configs／checkpoints／results／figures／notes |
| [logs/](logs/README.md) | 用途 → 日期 → run 或歷史來源 |
| `docs/experiment_registry/` | 跨實驗總帳與工作簿原文 |
| `docs/research_audit/`、`docs/manuscripts/`、`docs/presentations/` | 證據審查、投稿、簡報 |
| `literature_review/`、`docs/literature/` | 文獻工作流與 PDF／擷取資料 |
| `archive/` | 舊來源、文件製作檔、快取、終端輸出、可還原備份 |

## 回溯一筆歷史實驗

1. 在 [run_catalog.csv](docs/organization_2026-10-07/run_catalog.csv) 找 `source + row`；來源副本未去重，列數不是獨立實驗數。
2. 在 [file_mapping.csv](docs/organization_2026-10-07/file_mapping.csv) 用 experiment／run_key／原始檔名找 proposed（現在的位置）。時間相近不等於同一 run。
3. 查 [actual_migrations.csv](docs/organization_2026-10-07/execution/actual_migrations.csv) 的實際位置、搬移前及最終雜湊；現用路徑修改有個別原檔備份。
4. 設定、checkpoint、資料版本、split 或 commit 無證據時標待補；[待確認清單](docs/organization_2026-10-07/unknown_and_review.csv) 保存 282 個未完整對應的歷史檔案。

## 繼續實驗

從根目錄執行。`config.json` 的 cmapss_current 是新的前處理輸出位置，本次沒有訓練或重建 MAT。

```powershell
python data_process.py --config config.json
python DAST_test.py --config config.json --run-label descriptive_label
python run_cmapss_sweep.py --config config.json --datasets FD004 --epochs 1 --limit 1
```

以上是之後執行用的指令，本次未啟動。`--epochs 1` 存入 clip_warmup_smoke_test，其餘 sweep 存入 clip_warmup_hyperparameter_search。新訓練保存 config snapshot、時間命名權重、run 圖目錄、日期分類的 metrics.csv；C-MAPSS summary 保存 config／history／figure／checkpoint 路徑。終端 stdout 若另存為 train.log，也應放相同用途與日期的 logs 目錄。

歷史 `configs/local/` 保持原文；五份 `configs/relocated/` 只更新路徑，仍讀歷史 MAT。若要重跑前處理，先為 output_path 指定新的實驗資料目錄，避免覆寫 legacy MAT。不從現用 config 推定舊權重的架構。

## 保留的限制

- `DAST_test.py` 在 Git index 仍為 UU，整理未選擇合併版本；執行的是原工作檔加必要路徑修改。
- 19 個模型原本已在工作區缺失；未恢復或刪除 Git 紀錄。
- 歷史 test-selected 與 validation／holdout 結果分開保存；資料夾整理不改變證據限制。
- 原始報告保持內容；部分 analysis 報告另有 `_relocated.md` 閱讀副本修正圖連結。
- 舊入口 `DAST_main.py`／`Statistical features process .py` 的佔位輸入尚未修復；這需要獨立確認資料對應，不能藉整理猜測。
- `tmp/`、`_tmp_pptx_build/` 如仍有項目，是原有外部依賴連結；不跟隨或移除。
- `server_results/DAST_test/` 只剩受保護資料，完整舊 checkout 已備份；不要在殘留目錄啟動訓練。

## 版本控制

本次未 stage／commit／解決衝突。只明確加入需要的程式、設定與索引，禁止 `git add .`。權重、處理資料、結果、圖、logs、快照與備份由新 `.gitignore` 規則排除。歷史設定與 research 文件應另定提交範圍。

[搬移前 A–G 方案](docs/organization_2026-10-07/整理方案.md) · [執行報告](docs/organization_2026-10-07/執行報告.md)
