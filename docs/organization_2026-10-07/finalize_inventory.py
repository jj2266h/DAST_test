"""Refine role assignments from ledgers and build a reviewable run index."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows,fields):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

rows=read('file_mapping.csv'); catalog=[]; bindings=defaultdict(list)
for item in rows:
    p=ROOT/item['original']
    if item['role']=='tooling_or_git':continue
    if p.name.startswith('experiment_log') and p.suffix=='.csv':
        with p.open(encoding='utf-8-sig',newline='',errors='replace') as f:
            for n,r in enumerate(csv.DictReader(f),2):
                label=r.get('run_label','') or ''; run=r.get('run_id','') or ''
                if 'noclip' in label:e='cmapss_no_clip_no_warmup'
                elif 'run_once' in label:e='cmapss_run_once_trial'
                elif r.get('epochs')=='1':e='clip_warmup_smoke_test'
                elif 'clip' in label and 'warmup' in label:e='clip_warmup_hyperparameter_search'
                elif 'FEMTO' in str(r.get('dataset','')):e='femto_condition_training'
                else:e='cmapss_legacy_runs'
                key=label+'__'+run
                catalog.append(dict(source=item['original'],row=n,experiment=e,run_key=key,dataset=r.get('dataset',''),seed=r.get('seed',''),epochs=r.get('epochs',''),time_step=r.get('time_step',''),protocol='historical_ledger_not_independent_test',notes='原總帳列；可能與其他來源重複，不當獨立實驗加總'))
                for col in ['history_csv','plot_path','best_checkpoint','model_path']:
                    for val in (r.get(col,'') or '').split(';'):
                        name=Path(val.replace('\\','/')).name
                        if name:bindings[(item['origin'],name)].append((e,key,item['original'],n,col))
for item in rows:
    src=item['original'];p=Path(src); origin=item['origin']
    if src.startswith('phm2012_dataset/'):
        item['proposed']=src;item['action']='KEEP'
        if item['role']=='raw_data':item['evidence']='PHM2012 為內嵌 Git 資料集儲存庫，整體原位保留，不拆開 .git 與資料'
    if src=='server_results/experiments/run_cmapss_sweep.py':
        item['role']='notes/source_snapshot';item['experiment']='clip_warmup_hyperparameter_search';item['needs_review']='False'
        item['proposed']='experiments/clip_warmup_hyperparameter_search/notes/source_snapshot/server_mirror/experiments/run_cmapss_sweep.py'
        item['evidence']='已讀伺服器 sweep 程式；保留來源快照，不合併現用入口'
    candidates=bindings.get((origin,p.name),[])
    family_keys={(v[0],v[1]) for v in candidates}
    if len(family_keys)==1 and item['role'] in {'figures','checkpoints','epoch_metrics'}:
        e,key,ledger,n,col=candidates[0]
        old=item['experiment'];item['experiment']=e;item['run_key']=key
        item['proposed']=item['proposed'].replace('/'+old+'/','/'+e+'/')
        item['evidence']=f'{ledger}:{n} {col} 檔名對照；限同來源命名空間'
        if e!='cmapss_legacy_runs':item['needs_review']='False'
    if '/psp_rms/' in src or '/degradation_' in src:
        item['experiment']='phm2012_degradation_detection';item['role']='figures' if p.suffix.lower()=='.png' else 'results'
        item['proposed']=f'experiments/{item["experiment"]}/{item["role"]}/{origin}/{p.name}'
        item['needs_review']='False';item['evidence']='detect_train_psp.py:SAVE_DIR 或 detect_degradation_start.py 輸出用途'
    if p.name.startswith('OC標準化'):
        item['experiment']='operating_condition_normalization';item['role']='notes'
        item['proposed']=f'experiments/operating_condition_normalization/notes/{origin}/{p.name}'
        item['needs_review']='False';item['evidence']='已讀標準化實驗 epoch 表；沒有對照證據，不命名為 ablation'
    if p.name=='LR_wramup_0525.md':
        item['experiment']='learning_rate_warmup';item['role']='notes'
        item['proposed']=f'experiments/learning_rate_warmup/notes/{origin}/{p.name}'
        item['needs_review']='False';item['evidence']='warmup 歷史筆記；年份不從短檔名臆測'
    # Keep provenance namespaces but remove obsolete folder names from destinations.
    if item['role'] in {'checkpoints','configs','figures','results','epoch_metrics','log'} and item['experiment']:
        old=item['proposed']; marker='/'+origin+'/'
        if marker in old:
            prefix,tail=old.split(marker,1)
            namespace=origin
            if origin=='server_mirror' and src.startswith('server_results/experiments/'):
                namespace='server_experiments'
            item['proposed']=prefix+'/'+namespace+'/'+p.name

# Same target name never implies same run. Preserve a path-derived suffix if needed.
groups=defaultdict(list)
for r in rows:groups[r['proposed'].casefold()].append(r)
for group in groups.values():
    if len(group)>1:
        for r in group:
            p=Path(r['proposed']);source_parent=Path(r['original']).parent.as_posix().replace('/','__')
            r['proposed']=(p.parent/source_parent/p.name).as_posix()
            r['evidence']+='；同名目標保留來源子目錄，禁止覆蓋'

for source,e in [('server_results/summary/full_sweep_records.csv','clip_warmup_hyperparameter_search'),('server_results/experiments/clean_sweep_results/summary.jsonl','clip_warmup_validation_search'),('server_results/experiments/idea1_results/summary.jsonl','condition_slope_feature_compare'),('server_results/idea1_results/summary.jsonl','condition_slope_feature_compare')]:
    p=ROOT/source
    if p.suffix=='.csv':
        with p.open(encoding='utf-8-sig',newline='') as f: records=list(csv.DictReader(f))
    else:records=[json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    for n,r in enumerate(records,2 if p.suffix=='.csv' else 1):
        key=r.get('tag') or r.get('run_label','')+'__'+r.get('run_id','')
        catalog.append(dict(source=source,row=n,experiment=e,run_key=key,dataset=r.get('dataset',''),seed=r.get('seed',''),epochs=r.get('epochs',''),time_step=r.get('time_step',''),protocol='test_selected' if p.suffix=='.csv' else 'validation_selected_holdout_fields_present',notes='summary 原始列；跨 summary 相同 tag 不等於同次執行；缺少日期不填造'))
write('run_catalog.csv',catalog,['source','row','experiment','run_key','dataset','seed','epochs','time_step','protocol','notes'])
fields=list(rows[0]);write('file_mapping.csv',rows,fields)
write('unknown_and_review.csv',[r for r in rows if r['needs_review']=='True'],fields)
s=json.loads((OUT/'inventory_summary.json').read_text(encoding='utf-8'))
s['roles']=dict(Counter(r['role'] for r in rows));s['experiments']=dict(Counter(r['experiment'] for r in rows if r['experiment']))
s['actions']=dict(Counter(r['action'] for r in rows))
s['unknown_files']=sum(r['role']=='unknown' for r in rows);s['review_files']=sum(r['needs_review']=='True' for r in rows)
s['run_catalog_rows']=len(catalog)
dest=[r['proposed'].casefold() for r in rows]
s['case_insensitive_collisions']={k:v for k,v in Counter(dest).items() if v>1}
s['existing_target_conflicts']=[r['proposed'] for r in rows if r['action']=='PROPOSE_MOVE' and (ROOT/r['proposed']).exists()]
assert not s['case_insensitive_collisions']
assert not s['existing_target_conflicts']
assert all('/checkpoints/' in r['proposed'] for r in rows if r['role']=='checkpoints')
(OUT/'inventory_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(s,ensure_ascii=False,indent=2))
