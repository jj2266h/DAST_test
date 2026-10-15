"""Apply the approved per-file map, with preflight hashes and a durable journal."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
OUT = AUDIT / 'execution'
BACKUP = ROOT / 'archive/organization_backups/2026-10-07'
PROTECTED = {'.git', '.agents', '.claude', '.aris', '.codex', '.github'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def checked(relative):
    p = ROOT / relative
    resolved = p.resolve()
    assert resolved.is_relative_to(ROOT), str(p)
    assert not any(x in PROTECTED for x in p.relative_to(ROOT).parts), str(p)
    # No source or destination traverses a junction/symlink.
    for part in [p, *p.parents]:
        if part == ROOT: break
        if part.exists():
            assert not part.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT, str(part)
    return p

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'move_journal.jsonl').exists(), 'Already started: inspect journal before any retry.'
    with (AUDIT/'file_mapping.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    moving = [r for r in rows if r['action']=='PROPOSE_MOVE']
    assert len({r['proposed'].casefold() for r in rows})==len(rows)
    errors=[]
    for i,r in enumerate(moving):
        source,target=checked(r['original']),checked(r['proposed'])
        if not source.is_file() or target.exists():
            errors.append(r['original']+': source missing or destination exists');continue
        if source.stat().st_size != int(r['bytes']):errors.append(r['original']+': size changed')
        current=digest(source)
        if r['sha256'] and current!=r['sha256']:errors.append(r['original']+': content changed')
        r['sha256']=current
        if i%1000==0:print(f'Preflight {i}/{len(moving)}',flush=True)
    assert not errors, errors
    with (OUT/'approved_moves_with_hashes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(moving[0]));w.writeheader();w.writerows(moving)
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    for name,args in [('git_status_before.txt',['status','--porcelain=v1','-uall']),('worktree_before.patch',['diff','--binary']),('index_before.patch',['diff','--cached','--binary']),('unmerged_before.txt',['ls-files','--unmerged'])]:
        result=subprocess.run(['git','-c','core.quotepath=false',*args],cwd=ROOT,env=env,capture_output=True)
        (OUT/name).write_bytes(result.stdout)
    BACKUP.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'.git/index',BACKUP/'root_git_index.before')
    snapshot=[r for r in rows if r['original'].startswith('server_results/DAST_test/')]
    archive=BACKUP/'server_dast_test_before.zip'
    assert not archive.exists()
    print(f'Backing up server checkout ({len(snapshot)} files)...',flush=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for r in snapshot:
            assert r['kind']=='file'
            z.write(ROOT/r['original'],r['original'])
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(snapshot)
        assert z.testzip() is None
    (OUT/'backup.json').write_text(json.dumps({'path':archive.relative_to(ROOT).as_posix(),'files':len(snapshot),'sha256':digest(archive),'zip_crc_verified':True},indent=2),encoding='utf-8')
    print('Backup verified; moving files.',flush=True)
    with (OUT/'move_journal.jsonl').open('x',encoding='utf-8') as journal:
        for i,r in enumerate(moving,1):
            source,target=checked(r['original']),checked(r['proposed'])
            assert not target.exists()
            target.parent.mkdir(parents=True,exist_ok=True)
            record={'original':r['original'],'current':r['proposed'],'sha256_before_edit':r['sha256'],'bytes':int(r['bytes'])}
            journal.write(json.dumps(dict(record,status='intent'),ensure_ascii=False)+'\n');journal.flush()
            # Same-volume, no-overwrite rename on Windows; no recursive shell moves.
            source.rename(target)
            assert digest(target)==r['sha256'],str(target)
            journal.write(json.dumps(dict(record,status='done'),ensure_ascii=False)+'\n');journal.flush()
            if i%100==0:os.fsync(journal.fileno())
            if i%1000==0:print(f'Moved and hash-verified {i}/{len(moving)}',flush=True)
    # Only remove empty original directories; never touch protected paths or links.
    parents=set()
    for r in moving:
        for p in (ROOT/r['original']).parents:
            if p==ROOT:break
            parents.add(p)
    removed=[]
    for p in sorted(parents,key=lambda x:len(x.parts),reverse=True):
        if not p.exists() or any(x in PROTECTED for x in p.relative_to(ROOT).parts):continue
        if p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:continue
        assert p.resolve().is_relative_to(ROOT)
        try:p.rmdir();removed.append(p.relative_to(ROOT).as_posix())
        except OSError:pass  # Nonempty directories and protected children remain.
    (OUT/'removed_empty_directories.json').write_text(json.dumps(removed,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'migration_result.json').write_text(json.dumps({'moved':len(moving),'bytes':sum(int(r['bytes']) for r in moving),'all_destination_hashes_verified':True,'files_deleted':0,'removed_empty_directories':len(removed),'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2),encoding='utf-8')
    print('MIGRATION_OK',len(moving),flush=True)

if __name__=='__main__':main()
