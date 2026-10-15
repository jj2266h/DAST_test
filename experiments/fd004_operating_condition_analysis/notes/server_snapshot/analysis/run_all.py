"""
Master runner — executes all 7 analysis phases in sequence.
Usage (in conda activate dast):
    python analysis/run_all.py
"""
import subprocess
import sys
import os
import time

# Use the dast conda env Python if running from a different interpreter
_DAST_PYTHON = r"C:\Users\hjs92\.conda\envs\dast\python.exe"
PYTHON = _DAST_PYTHON if os.path.exists(_DAST_PYTHON) else sys.executable
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    ("01_data_preparation.py",      "Phase 1: Data Preparation & Basic EDA"),
    ("02_operating_conditions.py",  "Phase 2: Operating Condition Analysis"),
    ("03_sensor_eda.py",            "Phase 3: Sensor EDA"),
    ("04_oc_sensor_interaction.py", "Phase 4: OC × Sensor Interaction"),
    ("05_degradation_analysis.py",  "Phase 5: Degradation Trend Analysis"),
    ("06_models.py",                "Phase 6: Random Forest RUL Prediction"),
    ("07_generate_report.py",       "Phase 7: Generate Report"),
]

total_start = time.time()

for script, desc in scripts:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, os.path.join(SCRIPTS_DIR, script)],
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  Done in {elapsed:.1f}s")

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"  All phases complete in {total:.1f}s")
print(f"  Figures : analysis/figures/")
print(f"  Report  : analysis/analysis_report.md")
print(f"{'='*60}")
