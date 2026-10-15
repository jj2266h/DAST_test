"""Read-only project inventory; writes reports only beside this script. No moves."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import os
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))
PROTECTED = {'.git', '.agents', '.claude', '.codex', '.aris', '.github'}
TEXT = {'.py', '.sh', '.ps1', '.mjs', '.md', '.txt', '.json', '.jsonl', '.csv', '.yml', '.yaml', '.ipynb', '.log'}
errors, entries, dirs = [], [], []

def write_csv(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def git(*args):
    p = subprocess.run(['git', '-c', 'core.quotepath=false', *args], cwd=ROOT, capture_output=True)
    return p.stdout.decode('utf-8', errors='replace'), p.stderr.decode('utf-8', errors='replace')

status, git_warning = git('status', '--porcelain=v1', '-uall')
(OUT / 'git_status_before.txt').write_text(status + '\n' + git_warning, encoding='utf-8')
tracked = set(git('ls-files', '-z')[0].split('\0'))
missing = [p for p in tracked if p and not (ROOT / p).exists()]
write_csv('tracked_missing.csv', [{'original': p, 'action': 'PRE_EXISTING_MISSING_DO_NOT_DELETE_OR_RESTORE'} for p in sorted(missing)], ['original', 'action'])

# Exact source pointers from the historical ledger are stronger than filenames.
association = {}
ledger = ROOT / 'server_results/summary/full_sweep_records.csv'
with ledger.open(encoding='utf-8-sig', newline='') as f:
    ledger_rows = list(csv.DictReader(f))
for row in ledger_rows:
    for field in ['history_csv', 'plot_path', 'config_path', 'best_checkpoint']:
        for value in row.get(field, '').split(';'):
            if value:
                association[Path(value.replace('\\', '/')).name] = (row['run_label'] + '__' + row['run_id'], field)

for base, subdirs, files in os.walk(ROOT, followlinks=False, onerror=lambda e: errors.append(str(e))):
    b = Path(base)
    if b == OUT:
        subdirs[:] = []
        continue
    dirs.append(b.relative_to(ROOT).as_posix())
    for name in subdirs[:]:
        p = b / name
        if p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            entries.append((p, 'link'))
            subdirs.remove(name)
    for name in files:
        p = b / name
        entries.append((p, 'link' if p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT else 'file'))

def source_path(rel):
    if rel.startswith('server_results/DAST_test/'):
        return 'server_snapshot', rel[len('server_results/DAST_test/'):]
    if rel.startswith('server_results/'):
        return 'server_mirror', rel[len('server_results/'):]
    return 'local', rel

def classify(rel, kind):
    parts = Path(rel).parts
    origin, q = source_path(rel)
    p = Path(q); name = p.name; low = q.lower(); suffix = p.suffix.lower()
    if parts[0] in PROTECTED or '.git' in parts or '.claude' in parts:
        return rel, 'tooling_or_git', '', '工具、版本控制或既有工作樹；保留，不合併', False
    if kind == 'link':
        return rel, 'external_link', '', '連結不跟隨、不搬動；外部目標不在盤點範圍', False
    def exp(e, role, tail=None, reason='依來源目錄與檔名辨識用途', review=False):
        return f'experiments/{e}/{role}/{origin}/{tail or name}', role, e, reason, review
    if origin == 'local' and '/' not in q and (suffix == '.py' and name not in {'read_pdf.py', 'read_pdf2.py'} or name in {'config.json','README.md','CLAUDE.md','PROJECT_STRUCTURE.md','requirements.txt','.gitignore'}):
        return rel, 'core_or_entrypoint', '', '核心模型／訓練／測試／前處理／入口留根目錄；不按舊新版猜測刪除', False
    if q == 'experiments/run_cmapss_sweep.py' and origin == 'local':
        return 'run_cmapss_sweep.py', 'entrypoint', '', '主要 sweep 執行入口升到根目錄；需更新 tests import 與指令', False
    if '__pycache__' in parts or suffix == '.pyc':
        return f'archive/cache/{origin}/{q}', 'cache', '', 'Python 快取；候選清理但本次不刪除', False
    if q.startswith('tests/') and origin == 'local':
        return rel, 'tests', '', '測試程式保留 tests/', False
    if q.startswith('analysis/') and suffix == '.py' and origin == 'local':
        return rel, 'analysis_code', 'fd004_operating_condition_analysis', '專用分析工具保留模組關係；輸出路徑另修', False
    if q.startswith('Cmapss_data/'):
        if suffix == '.pdf': return f'docs/literature/{origin}/{name}', 'literature', '', '資料目錄中混入論文 PDF', False
        return f'data/raw/cmapss/{origin}/{p.relative_to("Cmapss_data").as_posix()}', 'raw_data', '', '原始 C-MAPSS；保留來源副本，未去重', False
    if q.startswith('phm2012_dataset/'):
        return rel, 'raw_data', '', 'PHM2012 為內嵌 Git 資料集儲存庫，整體原位保留，不拆開 .git 與資料', False
    for folder, label in [('train_dataset','cmapss_legacy'),('cmapss_dataset','cmapss_current'),('femto_dataset','femto')]:
        if q.startswith(folder + '/'):
            return f'data/processed/{label}/{origin}/{p.relative_to(folder).as_posix()}', 'processed_data', '', '處理後輸入資料，不視為結果；無法從現用 config 倒推歷史生成設定', False
    if re.fullmatch(r'(train|test)1[XY]_40.zip', name):
        return f'data/legacy_packages/cmapss_window40/{origin}/{name}', 'data_package', '', '已讀 ZIP 目錄，內含同名 MAT；window40 來源包，不解壓覆蓋', False
    if q.startswith('literature_review/') and origin == 'local':
        return rel, 'literature_workspace', '', '既有文獻工作流完整保留，避免相對連結失效', False
    if q.startswith('工工協會投稿/'):
        return f'docs/manuscripts/ciie2026/{p.relative_to("工工協會投稿").as_posix()}', 'manuscript', '', '投稿模板與草稿', False
    if name.startswith(('CIIE_', 'PAPER_PLAN')) or name == 'MANIFEST.md' and '/' not in q:
        return f'docs/manuscripts/ciie2026/{origin}/{name}', 'manuscript', '', '論文計畫與產物索引；副本保留', False
    if q.startswith('refine-logs/') or name in {'CLAIMS_FROM_RESULTS.md','EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','findings.md'}:
        return f'docs/research_audit/{origin}/{q}', 'research_audit', '', '證據審查與計畫，不是 train.log；原文歷史路徑保留並用搬移索引解析', False
    if name in {'read_pdf.py','read_pdf2.py','pdf_content.txt','pdf_content2.txt'}:
        return f'docs/literature/dast_paper/{origin}/{name}', 'literature_extraction', '', 'DAST 論文分頁擷取工具與文字；工具目前用舊絕對路徑', False
    if name == '存檔1.md':
        return 'archive/non_experiment/存檔1.md', 'non_experiment', '', '內容為非研究存檔；保留，不刪除', False
    if name in {'et --soft HEAD~1','how'}:
        return f'archive/terminal_exports/{name}', 'terminal_export', '', '已讀內容為帶 ANSI 色碼的 Git 輸出，非程式；用途已辨識，仍保留', False
    if q.startswith('_tmp_pptx_build/') or q.startswith('tmp/') and not q.startswith('tmp/condition_scaling_verify/'):
        if suffix == '.log':
            return f'logs/document_build/undated/{origin}/{q}', 'log', 'document_build', '文件渲染 log 無可靠執行日期；不以 mtime 當實驗日期', False
        return f'archive/document_build/{q}', 'document_build', '', '文件／簡報產製來源及中間產物；不可整包刪除', False
    if q.startswith('tmp/condition_scaling_verify/'):
        role = 'data' if suffix == '.mat' else 'configs' if name == 'config.json' else 'notes/source_snapshot'
        return exp('condition_scaling_verification', role, str(p.relative_to('tmp/condition_scaling_verify')).replace('\\','/'), 'OC scaling 驗證副本；保留來源程式與資料')
    if 'dast_best_models_noclip' in name:
        return exp('cmapss_no_clip_no_warmup', 'checkpoints', reason='已讀 ZIP 清單：四個 best 權重、設定、test.py；整包保留')
    if name == 'Untitled-1.ipynb':
        return exp('cmapss_preprocessing_exploration', 'notes', reason='已讀 notebook cells：C-MAPSS 前處理與 EDA 混合探索')
    if q.startswith('outputs/analysis_preview/') or q.startswith('analysis/'):
        role = 'figures' if suffix in {'.png','.jpg','.svg','.pdf'} else 'results' if suffix in {'.json','.csv'} else 'notes'
        return exp('fd004_operating_condition_analysis', role, q)
    if q.startswith('PHM-2008_prediction_result/'):
        return exp('phm2008_prediction', 'results/predictions', reason='PHM-2008 預測輸出；執行日期與模型來源待補', review=True)
    if name.startswith('PHM2012_') and suffix == '.md':
        return exp('phm2012_health_indicator', 'notes', reason='PHM2012 資料／HI 重建計畫，不能當已完成實驗')
    if name == 'FD004_diagnosis_plan.md': return exp('fd004_operating_condition_analysis','notes')
    if name == '實驗記錄.xlsx':
        return f'docs/experiment_registry/{origin}/{name}', 'multi_experiment_registry', '', '跨實驗工作簿；保留原表，不假設每個 sheet 同協定', False
    if '進度' in name or '簡報' in name or suffix in {'.pptx','.ndjson'} and '紀錄/' in q:
        return f'docs/presentations/{origin}/{q}', 'presentation', '', '研究進度簡報／擷取稿', False
    if 'clean_sweep_results/' in q or name == 'run_clean_sweep.sh':
        e = 'clip_warmup_validation_search'
    elif 'idea1_results/' in q or name == 'run_idea1_pilot.sh':
        e = 'condition_slope_feature_compare'
    elif 'femto' in low or 'FEMTO' in name:
        e = 'femto_condition_training'
    elif 'run_once_noclip' in name or 'run_once_configs/' in q or 'run_once_logs/' in q:
        e = 'cmapss_no_clip_no_warmup'
    elif 'run_once_20260926' in name:
        e = 'cmapss_run_once_trial'
    elif origin == 'local' and ('2026-08-11' in name and 'clip' in name or q.startswith('experiments/generated_configs/')):
        e = 'clip_warmup_smoke_test'
    elif name in association or 'clip' in name and 'warmup' in name or q.startswith('summary/'):
        e = 'clip_warmup_hyperparameter_search'
    elif name in {'prepare_splits.py','train_split.py'}:
        return f'archive/source_snapshots/validation_protocol/{name}', 'source_snapshot', 'validation_protocol_shared', 'server 版前處理／訓練快照；不提升成現用核心、不改研究協定', False
    elif name == 'run_cmapss_sweep.py':
        return exp('clip_warmup_hyperparameter_search','notes/source_snapshot',q,'伺服器 sweep 來源快照；不與現用入口合併')
    elif name.startswith(('experiment_log','expirement')) or name == 'sweep_runs.csv':
        return f'docs/experiment_registry/{origin}/{q}', 'multi_experiment_registry', '', '含跨 run 原始總帳，保留原文及 source path', False
    elif q.startswith(('plots/','紀錄/','model/','Model_trained/','train_model/','experiments/results/')) or suffix in {'.pth','.pt','.ckpt'} or name.startswith('training_result_'):
        e = 'cmapss_legacy_runs'
    elif origin == 'server_snapshot':
        return f'archive/source_snapshots/server_dast_test/{q}', 'source_snapshot', '', '伺服器舊程式／文件快照，保留每個檔案與原路徑對照；不當現用核心', False
    else:
        return f'archive/unknown/{origin}/{q}', 'unknown', '', '沒有足夠用途證據，待使用者確認', True
    reason = '由已讀來源家族／名稱辨識；家族分類不代表完整 run 關聯'
    review = e == 'cmapss_legacy_runs'
    if name in association:
        reason = 'full_sweep_records.csv:' + association[name][1] + ' 精確檔名對照；不同來源副本分開'
    if suffix in {'.pth','.pt','.ckpt'} or 'Model_trained/' in q or 'train_model/' in q:
        return exp(e,'checkpoints',q,reason,review)
    if suffix == '.log' or name.startswith(('history_','training_history_')) and suffix == '.csv':
        m = re.search(r'20\d{2}-\d{2}-\d{2}',name)
        date = m.group(0) if m else 'undated'
        if suffix == '.log' and date == 'undated':
            with (ROOT / rel).open(encoding='utf-8',errors='replace') as f: head = f.read(1200)
            m = re.search(r'20\d{2}-\d{2}-\d{2}',head)
            if m: date = m.group(0)
        return f'logs/{e}/{date}/{origin}/{q}', 'epoch_metrics' if suffix == '.csv' else 'log', e, reason + '; undated 表示無執行日期證據', review
    if suffix == '.json' and ('config' in low): return exp(e,'configs',q,reason,review)
    if suffix in {'.png','.jpg','.svg'}: return exp(e,'figures',q,reason,review)
    if suffix in {'.py','.sh'}: return exp(e,'notes/source_snapshot',q,reason,review)
    if suffix in {'.md','.txt','.docx','.pdf'}: return exp(e,'notes',q,reason,review)
    return exp(e,'results',q,reason,review)

rows, text_evidence, hash_groups = [], [], defaultdict(list)
for i, (p, kind) in enumerate(entries):
    rel = p.relative_to(ROOT).as_posix()
    s = p.lstat()
    target, role, experiment, reason, review = classify(rel,kind)
    protected = role in {'tooling_or_git','external_link'}
    digest = ''
    if not protected and role != 'raw_data':
        try:
            h = hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
            digest = h.hexdigest();hash_groups[(s.st_size,digest)].append(rel)
        except OSError as exc: errors.append(f'{rel}: {exc}')
    origin, q = source_path(rel)
    m = re.search(r'(20\d{2}-\d{2}-\d{2})[_ ](\d{2}[-:]\d{2}[-:]\d{2})',p.name)
    run = association.get(p.name, ('',''))[0] if experiment == 'clip_warmup_hyperparameter_search' else ''
    if not run and m: run = m.group(0)
    if not run and experiment in {'clip_warmup_validation_search','condition_slope_feature_compare'}:
        tag = re.search(r'(FD00\d_(?:global|condfe|none|shuffle)_clip[^/]+)',p.stem)
        if tag: run = tag.group(1)
    row = dict(original=rel, proposed=target, action='KEEP' if rel==target else 'PROPOSE_MOVE', kind=kind, role=role, experiment=experiment, origin=origin, run_key=run, bytes=s.st_size, modified_at=datetime.fromtimestamp(s.st_mtime,TZ).isoformat(), sha256=digest, tracked=rel in tracked, needs_review=review, evidence=reason)
    rows.append(row)
    if not protected and p.suffix.lower() in TEXT and role not in {'raw_data','processed_data'} and s.st_size < 3_000_000:
        content = p.read_text(encoding='utf-8-sig',errors='replace')
        if p.suffix.lower() in {'.py','.sh','.ps1','.mjs','.md','.json','.yml','.yaml','.ipynb','.jsonl'}:
            for line_no,line in enumerate(content.splitlines(),1):
                if re.search(r'Cmapss_data|cmapss_dataset|train_dataset|phm2012_dataset|femto_dataset|server_results|experiments[/\\]|Model_trained|train_model|["\x27]model["\x27]|紀錄|plots|Desktop[/\\]+DAST_test|sys\.path|__file__|read_pdf|pdf_content',line):
                    text_evidence.append(dict(file=rel,line=line_no,kind='runtime' if p.suffix.lower() in {'.py','.sh','.ps1','.mjs','.yml','.yaml'} else 'config_or_historical_reference',text=line[:1200]))
    if i%5000==0: print(f'inventoried {i}/{len(entries)}',flush=True)

write_csv('file_mapping.csv',rows,list(rows[0]))
write_csv('unknown_and_review.csv',[r for r in rows if r['needs_review']],list(rows[0]))
write_csv('path_references.csv',text_evidence,['file','line','kind','text'])
duplicates=[]
for (size,digest),names in hash_groups.items():
    if len(names)>1:
        for name in names:duplicates.append(dict(group=digest,bytes=size,copies=len(names),original=name,action='KEEP_ALL_PENDING_PROVENANCE_REVIEW'))
write_csv('duplicates_sha256.csv',duplicates,['group','bytes','copies','original','action'])
candidates=[]
for r in rows:
    if r['role'] in {'cache','terminal_export'}:
        candidates.append(dict(original=r['original'],reason=r['evidence'],recommendation='可刪候選；本次未刪除，cache 先確認來源可重建'))
write_csv('deletion_candidates.csv',candidates,['original','reason','recommendation'])
write_csv('directories.csv',[dict(original=d,action='KEEP_PROTECTED' if any(x in PROTECTED for x in Path(d).parts) else 'REVIEW_EMPTY_AFTER_FILE_MIGRATION') for d in dirs],['original','action'])
bytarget=defaultdict(list)
for r in rows:bytarget[r['proposed'].casefold()].append(r['original'])
collisions={k:v for k,v in bytarget.items() if len(v)>1}
existing_conflicts=[r['original'] for r in rows if r['action']=='PROPOSE_MOVE' and (ROOT/r['proposed']).exists()]
checkpoints=[r for r in rows if r['role']=='checkpoints']
stats=dict(total_entries=len(rows),regular_files=sum(r['kind']=='file' for r in rows),links=sum(r['kind']=='link' for r in rows),bytes=sum(r['bytes'] for r in rows if r['kind']=='file'),hashed_files=sum(bool(r['sha256']) for r in rows),actions=dict(Counter(r['action'] for r in rows)),roles=dict(Counter(r['role'] for r in rows)),experiments=dict(Counter(r['experiment'] for r in rows if r['experiment'])),top_level=dict(Counter(r['original'].split('/')[0] if '/' in r['original'] else '[root]' for r in rows)),unknown_files=sum(r['role']=='unknown' for r in rows),review_files=sum(r['needs_review'] for r in rows),duplicate_groups=len({r['group'] for r in duplicates}),duplicate_files=len(duplicates),duplicate_extra_bytes=sum(size*(len(names)-1) for (size,_),names in hash_groups.items() if len(names)>1),tracked_missing=len(missing),case_insensitive_collisions=collisions,existing_target_conflicts=existing_conflicts,scan_errors=errors,ledger_rows=len(ledger_rows),checkpoint_files=len(checkpoints),root_checkpoints=sum('/' not in r['original'] for r in checkpoints))
(OUT/'inventory_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
assert not errors, errors
assert not collisions, collisions
assert not existing_conflicts, existing_conflicts
assert all('/checkpoints/' in r['proposed'] for r in checkpoints)
print(json.dumps({k:v for k,v in stats.items() if k not in {'roles','top_level'}},ensure_ascii=False,indent=2))
