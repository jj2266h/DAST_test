"""Generate current navigation and derived report links without rewriting evidence."""
from pathlib import Path
from collections import Counter,defaultdict
from urllib.parse import quote,unquote
import csv,hashlib,json,os,re

ROOT=Path(__file__).resolve().parents[2]
AUDIT=Path(__file__).resolve().parent
OUT=AUDIT/'execution'
with (AUDIT/'file_mapping.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
mapping={r['original']:r['proposed'] for r in rows}
created=[]
def new(rel,text):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),rel
    p.write_text(text,encoding='utf-8');created.append(rel)
def edit(rel,text):
    p=ROOT/rel;before=p.read_bytes();backup=OUT/'before_edits'/rel
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_bytes(before)
    p.write_text(text,encoding='utf-8')
    edits=json.loads((OUT/'edited_files.json').read_text(encoding='utf-8'))
    edits.append({'path':rel,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'backup':backup.relative_to(ROOT).as_posix()})
    (OUT/'edited_files.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2),encoding='utf-8')
def link(base,target):
    return quote(os.path.relpath(ROOT/target,(ROOT/base).parent).replace('\\','/'),safe='/._-')

# Reading copies fix only resolvable Markdown links. Original reports stay byte-exact.
reading=[]
for r in rows:
    if r['original'] not in {'analysis/analysis_report.md','analysis/analysis_report_v1.md','server_results/DAST_test/analysis/analysis_report.md','server_results/DAST_test/analysis/analysis_report_v1.md'}:continue
    original=Path(r['original']);dest=Path(r['proposed']);derived=dest.with_name(dest.stem+'_relocated.md').as_posix()
    content=(ROOT/r['proposed']).read_text(encoding='utf-8-sig')
    def replace_link(m):
        target=unquote(m.group(2)).strip('<>')
        if re.match(r'[a-zA-Z]+:|#',target):return m.group(0)
        resolved=os.path.normpath(str(original.parent/target)).replace('\\','/')
        actual=mapping.get(resolved)
        if actual:return m.group(1)+'('+link(derived,actual)+')'
        return m.group(0)
    content=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',replace_link,content)
    header=f'> 2026-10-07 路徑更新閱讀副本；只修正可解析的 Markdown 連結。歷史原文：[來源]({link(derived,r["proposed"])})。研究內容與結論未重新驗證。\n\n'
    new(derived,header+content);reading.append({'source':r['proposed'],'derived':derived})

families=defaultdict(list)
for r in rows:
    if r['experiment'] and r['experiment']!='document_build':families[r['experiment']].append(r)
titles={
 'clip_warmup_hyperparameter_search':'歷史 clipping × warmup 搜尋；test-selected',
 'clip_warmup_validation_search':'validation 選模的 clipping × warmup 搜尋',
 'clip_warmup_smoke_test':'短程流程測試；不可當完整實驗',
 'condition_slope_feature_compare':'工況斜率特徵比較：global／condfe／none／shuffle',
 'cmapss_no_clip_no_warmup':'關閉 clipping 與 warmup 的 C-MAPSS 執行',
 'cmapss_run_once_trial':'單次試跑紀錄',
 'cmapss_legacy_runs':'歷史 C-MAPSS；具體 run／設定關聯待確認',
 'femto_condition_training':'FEMTO 各工況訓練',
 'fd004_operating_condition_analysis':'FD004 工況、感測器與退化分析',
 'phm2012_health_indicator':'PHM2012 HI 計畫與重建',
 'phm2008_prediction':'PHM2008 預測；模型來源待確認',
 'condition_scaling_verification':'condition scaling 前處理驗證副本',
 'operating_condition_normalization':'工況正規化歷史筆記',
 'learning_rate_warmup':'warmup 歷史筆記',
 'cmapss_preprocessing_exploration':'前處理／EDA notebook',
}
index=['# 實驗索引\n\n用途分類不代表公平對照或已通過結果審查。歷史副本保留來源，不自動合併。\n\n| 用途 | 盤點檔案數 | 說明 |\n|---|---:|---|']
for e,items in sorted(families.items()):
    folder=ROOT/'experiments'/e
    if not folder.exists():continue
    rel=f'experiments/{e}/README.md'
    counts=Counter(r['role'] for r in items)
    text=f'# {e}\n\n{titles.get(e,e)}。\n\n本次歸類 {len(items)} 個歷史檔案（含外部 logs／分析程式關聯）；不等於独立 run 數。\n\n'
    text+='| 作用 | 檔案數 |\n|---|---:|\n'+''.join(f'| {k} | {v} |\n' for k,v in sorted(counts.items()))
    text+='\n## 查找與回溯\n\n'
    for name in ['file_mapping.csv','run_catalog.csv','unknown_and_review.csv']:
        target='docs/organization_2026-10-07/'+name
        text+=f'- [{name}]({link(rel,target)})：以 experiment=`{e}` 篩選。\n'
    if (ROOT/'logs'/e).exists():text+=f'- [歷史 log／epoch 指標]({link(rel,"logs/"+e)})，依日期及來源分類；undated 表示無可靠日期。\n'
    for child in sorted(folder.iterdir()):
        if child.is_dir():text+=f'- [{child.name}/]({quote(child.name)}/)\n'
    for report in reading:
        if report['derived'].startswith(f'experiments/{e}/'):text+=f'- [報告閱讀副本：{Path(report["derived"]).name}]({link(rel,report["derived"])})\n'
    text+='\n`local`、`server_mirror`、`server_experiments`、`server_snapshot` 是來源識別。歷史設定原文不改；`configs/relocated/` 是另存的路徑更新版。\n\n'
    text+='新 C-MAPSS 執行由根目錄 `DAST_test.py`／`run_cmapss_sweep.py` 啟動；`notes/source_snapshot/` 只保存歷史來源，不是可直接啟動的入口。歷史缺少 commit、資料生成版本或 split 的欄位仍待補。\n'
    new(rel,text)
    index.append(f'| [{e}]({quote(e)}/README.md) | {len(items)} | {titles.get(e,e)} |')
new('experiments/README.md','\n'.join(index)+'\n')
new('logs/README.md','# 執行紀錄\n\n歷史 log 與逐 epoch metrics 按 `<實驗用途>/<YYYY-MM-DD 或 undated>/<來源>/` 保存。原始檔名保留；mtime 不視為執行日期。\n\n新訓練的 metrics 使用 `<實驗用途>/<日期>/<run_suffix>/metrics.csv`。跨 run summary 放在對應 `experiments/<用途>/results/`。\n\n[全專案索引](../PROJECT_STRUCTURE.md)\n')
new('data/README.md','# 資料用途\n\n- `raw/cmapss/local/`：原始 C-MAPSS；server_snapshot 副本獨立保留。\n- `processed/cmapss_legacy/local/`：既有歷史 MAT，生成設定與版本未完全確立，請勿用新前處理覆寫。\n- `processed/cmapss_current/local/`：現用 config 的新前處理輸出；本次未產生資料。\n- `processed/femto/local/`：搬移前的 FEMTO 處理資料。\n- `legacy_packages/cmapss_window40/`：舊 ZIP，未解壓合併。\n- 根目錄 `phm2012_dataset/`：獨立 Git 資料集儲存庫，原位保留。\n\n模型權重、特徵維度與 MAT 的配對不可只靠資料集名稱判定。\n')
(ROOT/'data/processed/cmapss_current/local').mkdir(parents=True,exist_ok=True)
new('server_results/ARCHIVED_README.md','# 已拆分封存的伺服器快照\n\n此處留下的 `.git` 與工具資料受到保護。實驗產物已按用途搬移，這裡不再是可直接執行的完整 checkout。\n\n- [完整搬移前 ZIP（含 .git）](../archive/organization_backups/2026-10-07/server_dast_test_before.zip)\n- [逐檔對照](../docs/organization_2026-10-07/file_mapping.csv)\n- [實驗索引](../experiments/README.md)\n\n未刪除或改寫這裡的 Git index／history；請勿在殘留 checkout 執行 restore 或訓練。\n')

structure='''# DAST 專案結構與回溯入口

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
'''
edit('PROJECT_STRUCTURE.md',structure)
readme='''# DAST — Dual-Aspect Self-Attention Transformer

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
'''
edit('README.md',readme)
ignore=(ROOT/'.gitignore').read_text(encoding='utf-8')
ignore+='''
# Purpose-based experiment organization (2026-10-07)
/archive/
/logs/
/data/processed/
/data/legacy_packages/
/data/raw/cmapss/server_snapshot/
/experiments/*/checkpoints/
/experiments/*/results/
/experiments/*/figures/
/experiments/*/data/
/experiments/*/notes/source_snapshot/
/docs/organization_2026-10-07/execution/
/docs/manuscripts/
/docs/presentations/
/docs/research_audit/
/docs/experiment_registry/
/docs/literature/
# Run configs are small provenance records; include only explicitly selected ones.
!/experiments/*/configs/**/*.json
!/docs/organization_2026-10-07/inventory_summary.json
'''
edit('.gitignore',ignore)
(OUT/'reading_copies.json').write_text(json.dumps(reading,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'new_navigation_files.json').write_text(json.dumps(created,ensure_ascii=False,indent=2),encoding='utf-8')
print('INDEXES_OK',len(created),'new files')
