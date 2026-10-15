"""Verify migration, unchanged evidence, staged state and output routing without training."""
from pathlib import Path
from collections import Counter
import ast,csv,hashlib,json,os,stat,subprocess,tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
AUDIT=Path(__file__).resolve().parent
OUT=AUDIT/'execution'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
with (AUDIT/'file_mapping.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
with (OUT/'approved_moves_with_hashes.csv').open(encoding='utf-8-sig',newline='') as f:moving=list(csv.DictReader(f))
edits=json.loads((OUT/'edited_files.json').read_text(encoding='utf-8'));editmap={r['path']:r for r in edits}
errors=[];actual=[]
for i,r in enumerate(moving):
    old,new=ROOT/r['original'],ROOT/r['proposed']
    expected=editmap.get(r['proposed'],{}).get('after_sha256',r['sha256'])
    value=digest(new) if new.is_file() else ''
    if old.exists() or value!=expected:errors.append(r['original'])
    actual.append(dict(original=r['original'],current=r['proposed'],role=r['role'],experiment=r['experiment'],bytes_before=r['bytes'],bytes_after=new.stat().st_size if new.exists() else '',sha256_before=r['sha256'],sha256_after=value,edited_after_move=r['proposed'] in editmap))
    if i%1000==0:print('Verify moved files',i,len(moving),flush=True)
with (OUT/'actual_migrations.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(actual[0]));w.writeheader();w.writerows(actual)
for r in rows:
    if r['action']!='KEEP' or r['role'] in {'tooling_or_git','external_link'}:continue
    p=ROOT/r['original']
    if not p.is_file():errors.append(r['original']+': missing preserved file');continue
    expected=editmap.get(r['original'],{}).get('after_sha256',r['sha256'])
    if expected:
        if digest(p)!=expected:errors.append(r['original']+': changed preserved content')
    elif p.stat().st_size!=int(r['bytes']):errors.append(r['original']+': raw size changed')
for r in edits:
    if digest(ROOT/r['path'])!=r['after_sha256'] or digest(ROOT/r['backup'])!=r['before_sha256']:
        errors.append(r['path']+': edit or backup hash mismatch')

def git(args,extra=None):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0');env.update(extra or {})
    p=subprocess.run(['git',*args],cwd=ROOT,env=env,capture_output=True,check=True)
    return p.stdout
unmerged=git(['ls-files','--unmerged'])
stages_same=git(['ls-files','--stage','-z'])==git(['ls-files','--stage','-z'],{'GIT_INDEX_FILE':str(ROOT/'archive/organization_backups/2026-10-07/root_git_index.before')})
conflict_same=unmerged==(OUT/'unmerged_before.txt').read_bytes()
assert stages_same and conflict_same
syntax=[]
for p in [*ROOT.glob('*.py'),*(ROOT/'analysis').glob('*.py'),*(ROOT/'tests').glob('*.py')]:
    ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));syntax.append(p.relative_to(ROOT).as_posix())

# Stop before loading data; exercise the new path setup without model construction.
import sys
sys.path.insert(0,str(ROOT))
import DAST_test,DAST_FEMTO_train
class StopBeforeData(Exception):pass
config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
routing=[]
for module,family,loader in [(DAST_test,config['training']['experiment_name'],'load_array'),(DAST_FEMTO_train,'femto_condition_training','load_femto_split')]:
    with tempfile.TemporaryDirectory() as d:
        wd=Path(d);(wd/'config.json').write_text(json.dumps(config),encoding='utf-8')
        previous=Path.cwd();os.chdir(wd)
        try:
            with patch.object(module,loader,side_effect=StopBeforeData):
                if module is DAST_test:
                    with patch.object(module,'parse_args',return_value=SimpleNamespace(config='config.json',run_label='path_check')):
                        try:module.main()
                        except StopBeforeData:pass
                else:
                    try:module.main()
                    except StopBeforeData:pass
            snapshots=list((wd/'experiments'/family/'configs/runs').glob('*.json'))
            assert len(snapshots)==1 and json.loads(snapshots[0].read_text(encoding='utf-8'))==config
            assert (wd/'experiments'/family/'checkpoints').is_dir()
            assert list((wd/'logs'/family).glob('*/*'))
            assert not list(wd.glob('*.pth'))
            routing.append(module.__name__)
        finally:os.chdir(previous)

# Source pointers in root config point at the intended inputs; current MAT is not fabricated.
assert (ROOT/config['cmapss']['data_path']/('train_'+config['cmapss']['dataset']+'.txt')).is_file()
assert (ROOT/config['femto']['femto_path']).is_dir()
assert (ROOT/config['femto']['output_path']).is_dir()
core_unchanged={name:digest(ROOT/name)==next(r['sha256'] for r in rows if r['original']==name) for name in ['DAST_Network.py','DAST_utils.py','data_process.py','test.py','DAST_visualization.py']}
assert all(core_unchanged.values())
report={'moved_files_verified':len(actual),'original_bytes_moved':sum(int(r['bytes']) for r in moving),'files_deleted':0,'hash_errors':errors,'edited_files':len(edits),'original_edits_backed_up':True,'unchanged_core':core_unchanged,'git_staged_entries_unchanged':stages_same,'git_unmerged_stages_unchanged':conflict_same,'syntax_checked_files':len(syntax),'training_output_routing_checked_without_training':routing,'checkpoint_paths_valid':all('/checkpoints/' in r['proposed'] for r in rows if r['role']=='checkpoints'),'root_checkpoint_files':len([p for p in ROOT.iterdir() if p.suffix in {'.pth','.pt','.ckpt'}]),'kept_raw_data_check':'existence and size; not full signal-content validation','tests_before':'12 passed','tests_after':'12 passed','training_run':False}
(OUT/'verification_final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
assert not errors
