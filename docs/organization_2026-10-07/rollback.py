"""Restore original files and locations. Default is a read-only precheck.

New indexes, derived configs, reports and backups are retained. Git is never reset.
Run with --apply only when restoration is intended. Changed files stop the restore.
"""
from pathlib import Path
import argparse,csv,hashlib,json,stat

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'execution'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def checked(rel):
    p=ROOT/rel
    assert p.resolve().is_relative_to(ROOT)
    assert not any(x in {'.git','.agents','.claude','.codex','.aris','.github'} for x in p.relative_to(ROOT).parts)
    for q in [p,*p.parents]:
        if q==ROOT:break
        if q.exists():assert not q.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    return p
args=argparse.ArgumentParser(description=__doc__)
args.add_argument('--apply',action='store_true');args=args.parse_args()
with (OUT/'approved_moves_with_hashes.csv').open(encoding='utf-8-sig',newline='') as f:moves=list(csv.DictReader(f))
edits=json.loads((OUT/'edited_files.json').read_text(encoding='utf-8'));byedit={r['path']:r for r in edits}
for r in moves:
    source,target=checked(r['proposed']),checked(r['original'])
    assert source.is_file() and not target.exists(),r['original']
    if args.apply:assert sha(source)==byedit.get(r['proposed'],{}).get('after_sha256',r['sha256']),r['proposed']
for r in edits:
    assert checked(r['path']).is_file() and checked(r['backup']).is_file()
    if args.apply:
        assert sha(checked(r['path']))==r['after_sha256'],r['path']
        assert sha(checked(r['backup']))==r['before_sha256'],r['backup']
print(f'Precheck OK: {len(moves)} moves, {len(edits)} edits. New artifacts and Git will be retained.')
if not args.apply:
    print('No files changed. --apply additionally verifies content hashes before restoring.')
else:
    assert not (OUT/'rollback_journal.jsonl').exists(),'Inspect prior rollback journal before retry.'
    with (OUT/'rollback_journal.jsonl').open('x',encoding='utf-8') as journal:
        for r in edits:
            checked(r['path']).write_bytes(checked(r['backup']).read_bytes())
            assert sha(checked(r['path']))==r['before_sha256']
            journal.write(json.dumps({'restored_content':r['path']},ensure_ascii=False)+'\n');journal.flush()
        for r in reversed(moves):
            source,target=checked(r['proposed']),checked(r['original'])
            assert not target.exists()
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
            assert sha(target)==r['sha256']
            journal.write(json.dumps({'restored_location':r['original']},ensure_ascii=False)+'\n');journal.flush()
    print('Original files restored; audit, new navigation/derived files and backups retained.')
