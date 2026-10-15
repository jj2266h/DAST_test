"""Surgical path changes; preserve each edited file before the first change."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'execution'
assert (OUT/'migration_result.json').exists(), 'Finish migration first.'
BACK=OUT/'before_edits'
changes=json.loads((OUT/'edited_files.json').read_text(encoding='utf-8')) if (OUT/'edited_files.json').exists() else []

def save(relative, content):
    p=ROOT/relative
    original=p.read_bytes()
    updated=content.encode('utf-8')
    if original==updated:return
    backup=BACK/relative
    backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():backup.write_bytes(original)
    p.write_bytes(updated)
    changes.append({'path':relative,'before_sha256':hashlib.sha256(original).hexdigest(),'after_sha256':hashlib.sha256(updated).hexdigest(),'backup':backup.relative_to(ROOT).as_posix()})
    (OUT/'edited_files.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')

def replace(relative,pairs):
    previous=next((r for r in changes if r['path']==relative),None)
    if previous and hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==previous['after_sha256']:
        return
    content=(ROOT/relative).read_bytes().decode('utf-8')
    nl='\r\n' if '\r\n' in content else '\n'
    for old,new in pairs:
        old=old.replace('\n',nl);new=new.replace('\n',nl)
        assert old in content,(relative,old)
        content=content.replace(old,new)
    save(relative,content)

replace('config.json',[
    ('"data_path": "Cmapss_data"','"data_path": "data/raw/cmapss/local"'),
    ('"output_path": "cmapss_dataset"','"output_path": "data/processed/cmapss_current/local"'),
    ('"output_path": "femto_dataset/"','"output_path": "data/processed/femto/local"'),
    ('"model_save_path": "model"','"experiment_name": "cmapss_no_clip_no_warmup",\n    "model_save_path": "experiments/cmapss_no_clip_no_warmup/checkpoints"')])

replace('DAST_test.py',[
    ('    log_dir       = "紀錄"\n    history_csv   = os.path.join(log_dir, f"training_history_{run_suffix}.csv")\n    summary_csv   = os.path.join(log_dir, "experiment_log.csv")',
     '    experiment_name = _tr.get("experiment_name", "cmapss_training")\n    experiment_dir = os.path.join("experiments", experiment_name)\n    log_dir       = os.path.join("logs", experiment_name, run_id[:10], run_suffix)\n    history_csv   = os.path.join(log_dir, "metrics.csv")\n    summary_csv   = os.path.join(experiment_dir, "results", "experiment_log.csv")\n    figure_dir    = os.path.join(experiment_dir, "figures", run_suffix)\n    config_dir    = os.path.join(experiment_dir, "configs", "runs")\n    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)\n    os.makedirs(config_dir, exist_ok=True)\n    config_path = os.path.join(config_dir, f"{run_suffix}.json")\n    with open(config_path, "x", encoding="utf-8") as f:\n        json.dump(config, f, ensure_ascii=False, indent=2)'),
    ('        save_dir="plots",','        save_dir=figure_dir,')])

replace('run_cmapss_sweep.py',[
    ('                    config["training"]["model_save_path"] = str(Path("experiments/results/models"))\n',''),
    ('                    clip_label = _format_number(clip_threshold)',
     '                    experiment_name = ("clip_warmup_smoke_test"\n                                       if config["training"]["epochs"] == 1\n                                       else "clip_warmup_hyperparameter_search")\n                    config["training"]["experiment_name"] = experiment_name\n                    config["training"]["model_save_path"] = str(\n                        Path("experiments") / experiment_name / "checkpoints"\n                    )\n                    clip_label = _format_number(clip_threshold)'),
    ('        default="experiments/generated_configs",','        default=None,'),
    ('    jobs = create_sweep_jobs(','    experiment_name = ("clip_warmup_smoke_test"\n                       if (args.epochs or base_config["training"]["epochs"]) == 1\n                       else "clip_warmup_hyperparameter_search")\n    output_dir = args.generated_config_dir or str(\n        Path("experiments") / experiment_name / "configs" / "generated"\n    )\n    jobs = create_sweep_jobs('),
    ('        output_dir=Path(args.generated_config_dir),','        output_dir=Path(output_dir),')])
replace('tests/test_run_cmapss_sweep.py',[
    ('from experiments.run_cmapss_sweep import','from run_cmapss_sweep import'),
    ('str(Path("experiments/results/models"))','str(Path("experiments/clip_warmup_smoke_test/checkpoints"))')])

replace('DAST_FEMTO_train.py',[
    ('    model_dir  = _tr["model_save_path"]',
     '    experiment_dir = os.path.join("experiments", "femto_condition_training")\n    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")\n    run_suffix = f"FEMTO_cond{CONDITION}_{run_id}"\n    model_dir = os.path.join(experiment_dir, "checkpoints")\n    best_ckpt = os.path.join(model_dir, f"dast_{run_suffix}_best.pth")\n    figure_dir = os.path.join(experiment_dir, "figures", run_suffix)\n    log_dir = os.path.join("logs", "femto_condition_training", run_id[:10], run_suffix)\n    config_dir = os.path.join(experiment_dir, "configs", "runs")\n    for directory in (model_dir, log_dir, config_dir):\n        os.makedirs(directory, exist_ok=True)\n    with open(os.path.join(config_dir, f"{run_suffix}.json"), "x", encoding="utf-8") as f:\n        json.dump(config, f, ensure_ascii=False, indent=2)'),
    ('            torch.save(model.state_dict(), f"dast_FEMTO_cond{CONDITION}_best.pth")','            torch.save(model.state_dict(), best_ckpt)'),
    ('    best_ckpt = f"dast_FEMTO_cond{CONDITION}_best.pth"\n',''),
    ('f\'dast_FEMTO_cond{CONDITION}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth\'','f"dast_{run_suffix}.pth"'),
    ('        loss_history["mae"].append(mae)','        loss_history["mae"].append(mae)\n        append_experiment_log(os.path.join(log_dir, "metrics.csv"), {\n            "epoch": epoch, "train_loss": train_loss_avg, "test_loss": test_loss,\n            "rmse": rmse, "mae": mae,\n        })'),
    ('save_dir="plots", condition=CONDITION','save_dir=figure_dir, condition=CONDITION')])

replace('DAST_final.py',[
    ('DATA_ROOT   = r"CMAPSSData"','DATA_ROOT   = r"data/raw/cmapss/local"'),
    ('    best_path = os.path.join(root, f"best_DAST_{fd}_PAPER_DROPOUT.pth")','    checkpoint_dir = os.path.join("experiments", "dast_paper_dropout", "checkpoints")\n    os.makedirs(checkpoint_dir, exist_ok=True)\n    best_path = os.path.join(checkpoint_dir, f"best_DAST_{fd}_PAPER_DROPOUT.pth")')])
replace('DAST_main.py',[
    ("File_Path = '' + '\\\\' + 'DAST' + '\\\\' + 'F001' + '\\\\'",'File_Path = os.path.join("experiments", "dast_original_training", "checkpoints")')])
# Preserve the legacy placeholder input semantics; only route its writes.
replace('Statistical features process .py',[
    ('import pandas as pd','import os\nimport pandas as pd'),
    ("sio.savemat('F001_window_size_trainX_new.mat'",'output_dir = "experiments/cmapss_preprocessing_exploration/data"\nos.makedirs(output_dir, exist_ok=True)\nsio.savemat(os.path.join(output_dir, "F001_window_size_trainX_new.mat")'),
    ("sio.savemat('F001_window_size_testX_new.mat'",'sio.savemat(os.path.join(output_dir, "F001_window_size_testX_new.mat")')])

replace('FEMTO_datapreprocess.py',[
    ('plots/hi_overview.png','experiments/phm2012_health_indicator/figures/hi_overview.png')])
replace('detect_train_psp.py',[
    ('SAVE_DIR        = "plots/psp_rms"','SAVE_DIR        = "experiments/phm2012_degradation_detection/figures"'),
    ('    csv_path = os.path.join(SAVE_DIR, "psp_rms_summary.csv")','    result_dir = "experiments/phm2012_degradation_detection/results"\n    os.makedirs(result_dir, exist_ok=True)\n    csv_path = os.path.join(result_dir, "psp_rms_summary.csv")')])
replace('detect_degradation_start.py',[
    ('os.makedirs("plots", exist_ok=True)','os.makedirs("experiments/phm2012_degradation_detection/figures", exist_ok=True)'),
    ('f"plots/psp_detection_{bearing_name}.png"','f"experiments/phm2012_degradation_detection/figures/psp_detection_{bearing_name}.png"')])

replace('analysis/utils.py',[
    ('DATA_DIR     = r"C:\\Users\\hjs92\\Desktop\\DAST_test\\Cmapss_data"\nANALYSIS_DIR = r"C:\\Users\\hjs92\\Desktop\\DAST_test\\analysis"\nFIG_DIR      = os.path.join(ANALYSIS_DIR, "figures")\nSTATS_FILE   = os.path.join(ANALYSIS_DIR, "stats.json")',
     'ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\nDATA_DIR = os.path.join(ROOT, "data", "raw", "cmapss", "local")\nEXPERIMENT_DIR = os.path.join(ROOT, "experiments", "fd004_operating_condition_analysis")\nANALYSIS_DIR = os.path.join(EXPERIMENT_DIR, "notes", "current")\nFIG_DIR = os.path.join(EXPERIMENT_DIR, "figures", "current")\nSTATS_FILE = os.path.join(EXPERIMENT_DIR, "results", "current", "stats.json")\nos.makedirs(ANALYSIS_DIR, exist_ok=True)\nos.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)')])
for filename in ['analysis/08_dast_inference.py','analysis/09_dast_feature_importance.py']:
    replace(filename,[
        ('r"C:\\Users\\hjs92\\Desktop\\DAST_test"','os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'),
        ('os.path.join(ROOT, "train_dataset")','os.path.join(ROOT, "data", "processed", "cmapss_legacy", "local")'),
        ('os.path.join(ROOT, f"dast_{DATASET}_best.pth")','os.path.join(ROOT, "experiments", "cmapss_legacy_runs", "checkpoints", "local", f"dast_{DATASET}_best.pth")')])
replace('analysis/07_generate_report.py',[
    ('figures/','../../figures/current/'),
    ('`analysis/../../figures/current/`','`experiments/fd004_operating_condition_analysis/figures/current/`')])
replace('analysis/run_all.py',[
    ('Figures : analysis/figures/','Figures : experiments/fd004_operating_condition_analysis/figures/current/'),
    ('Report  : analysis/analysis_report.md','Report  : experiments/fd004_operating_condition_analysis/notes/current/analysis_report.md')])

for name in ['read_pdf.py','read_pdf2.py']:
    relative='docs/literature/dast_paper/local/'+name
    replace(relative,[
        ('import sys','import sys\nfrom pathlib import Path'),
        ('r\'C:\\Users\\hjs92\\Desktop\\DAST_test\\Cmapss_data\\fbf0a9f2-20ce-49a8-87a7-53204af19338-author.pdf\'',
         'Path(__file__).resolve().parents[4] / "docs/literature/local/fbf0a9f2-20ce-49a8-87a7-53204af19338-author.pdf"')])

# Retain historical JSON byte-for-byte; create five explicitly derived local configs.
with (OUT.parent/'file_mapping.csv').open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
derived=[]
for row in manifest:
    if not row['original'].startswith(('experiments/run_once_configs/','experiments/generated_configs/')):continue
    cfg=json.loads((ROOT/row['proposed']).read_text(encoding='utf-8'))
    cfg['cmapss']['data_path']='data/raw/cmapss/local'
    cfg['cmapss']['output_path']='data/processed/cmapss_legacy/local'
    if 'femto' in cfg:cfg['femto']['output_path']='data/processed/femto/local'
    cfg['training']['experiment_name']=row['experiment']
    cfg['training']['model_save_path']=f'experiments/{row["experiment"]}/checkpoints'
    target=ROOT/f'experiments/{row["experiment"]}/configs/relocated/{Path(row["original"]).name}'
    text=json.dumps(cfg,ensure_ascii=False,indent=2)+'\n'
    if target.exists():assert target.read_text(encoding='utf-8')==text
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text,encoding='utf-8')
    derived.append({'original':row['original'],'preserved_source':row['proposed'],'derived':target.relative_to(ROOT).as_posix(),'scope':'Path-only derivative; legacy MAT generation compatibility not established.'})
(OUT/'derived_configs.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
print('PATH_UPDATES_OK',len(changes),'edited;',len(derived),'derived configs')
