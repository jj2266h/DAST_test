import csv
import json
import os
from datetime import datetime


# Build file and folder paths for a single experiment run.
def build_run_paths(dataset_name, batch_size, model_dir="Model_trained", log_dir="experiment_logs"):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dast_{dataset_name}_bs{batch_size}_{run_timestamp}"
    model_path = os.path.join(model_dir, f"{run_name}.pth")
    history_path = os.path.join(log_dir, f"{run_name}.json")
    summary_csv_path = os.path.join(log_dir, "experiment_summary.csv")
    return {
        "run_timestamp": run_timestamp,
        "run_name": run_name,
        "model_dir": model_dir,
        "log_dir": log_dir,
        "model_path": model_path,
        "history_path": history_path,
        "summary_csv_path": summary_csv_path,
    }


# Create the output folders if they do not exist yet.
def ensure_log_dirs(model_dir, log_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


# Append one experiment summary row to the CSV log file.
def append_experiment_log(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# Save the full experiment metadata and all epoch metrics to JSON.
def save_experiment_history(history_path, experiment_record, epoch_history):
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": experiment_record,
                "epoch_history": epoch_history,
            },
            f,
            indent=2,
        )
