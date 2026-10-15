# DAST — Dual-Aspect Self-Attention Transformer

C-MAPSS RUL 與 PHM2012／FEMTO HI 研究專案。模型、訓練、測試與前處理程式保留根目錄，實驗產物按用途分類。

**先看 [專案結構與回溯方法](PROJECT_STRUCTURE.md) 或 [實驗索引](experiments/README.md)。**

## C-MAPSS

從專案根目錄執行：

```powershell
python -m pip install -r requirements.txt
python data_process.py --config config.json
python DAST_test.py --config config.json --run-label descriptive_label
```

`data_process.py` 已包含統計特徵輸出，不需另跑舊的 `Statistical features process .py`。
`config.json` 保留原本的超參數與選模行為，只更新資料／輸出路徑及實驗名稱。

- 原始資料：`data/raw/cmapss/local/`。
- 新前處理資料：`data/processed/cmapss_current/local/`；整理沒有重建資料，訓練前先做前處理。
- 新結果：`experiments/<training.experiment_name>/` 下的 configs、checkpoints、results、figures。
- epoch metrics：`logs/<實驗名稱>/<日期>/<run_suffix>/metrics.csv`。

評估已確認與資料／架構配對的 checkpoint：

```powershell
python test.py --config <matching_config.json> --checkpoint <checkpoint.pth>
```

歷史 `dast_FD001_best.pth` 等已移至 `experiments/cmapss_legacy_runs/checkpoints/local/`；不可只靠 FD 名称假設它與現用 config 相容。

## Sweep

```powershell
python run_cmapss_sweep.py --config config.json --datasets FD004 --epochs 1 --limit 1
```

這是啟動新實驗的指令，本次整理未執行。1 epoch 歸 clip_warmup_smoke_test，其餘歸 clip_warmup_hyperparameter_search。舊的 `python experiments/run_cmapss_sweep.py` 入口已提升至根目錄。

## PHM2012／FEMTO

```powershell
python FEMTO_datapreprocess.py
python DAST_FEMTO_train.py
```

原始 `phm2012_dataset/` 連同 Git 原位保留；處理資料在 `data/processed/femto/local/`。工況由 `config.json` 的 `femto.condition` 選擇。新 FEMTO 權重與圖在 `experiments/femto_condition_training/`，逐 epoch 指標在日期分類的 logs，best 權重不再覆寫根目錄檔案。

## 分析與研究資料

- `analysis/`：原有 FD004 分析程式，新輸出在 `experiments/fd004_operating_condition_analysis/` 的 current 子目錄。
- `docs/experiment_registry/`：跨實驗原始總帳與實驗工作簿。
- `docs/research_audit/`、`docs/manuscripts/ciie2026/`、`docs/presentations/`：研究審查、投稿與簡報。
- `literature_review/`、`docs/literature/`：文獻工作流與資料。
- [搬移、備份與驗證紀錄](docs/organization_2026-10-07/執行報告.md)。

## 驗證與限制

```powershell
python -B -m unittest discover -s tests -v
```

目前仍有既有 Git 衝突 `DAST_test.py`（UU），整理未替你選擇版本。歷史 test-selected 搜尋、validation sweep 與工況特徵比較分開保存；本次沒有變更 architecture、loss、資料切分、選模協定或重跑研究實驗。部分 legacy 入口仍有原本的輸入佔位，請使用上方現用流程。
