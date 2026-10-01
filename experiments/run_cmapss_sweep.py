import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLIP_THRESHOLDS = [0.5, 1.0, 2.0, 5.0]
DEFAULT_WARMUP_EPOCHS = [5, 10, 15]
DEFAULT_SEEDS = [42, 20, 100]


@dataclass(frozen=True)
class SweepJob:
    run_label: str
    config_path: Path


def _format_number(value):
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def create_sweep_jobs(
    base_config,
    output_dir,
    datasets,
    clip_thresholds,
    warmup_epochs,
    seeds,
    epochs=None,
    limit=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for dataset in datasets:
        for clip_threshold in clip_thresholds:
            for warmup_epoch in warmup_epochs:
                for seed in seeds:
                    if limit is not None and len(jobs) >= limit:
                        return jobs

                    config = copy.deepcopy(base_config)
                    config["cmapss"]["dataset"] = dataset
                    config["training"]["grad_clip_enabled"] = True
                    config["training"]["grad_clip_max_norm"] = float(clip_threshold)
                    config["training"]["lr_warmup_enabled"] = True
                    config["training"]["lr_warmup_epochs"] = int(warmup_epoch)
                    config["training"]["model_save_path"] = str(Path("experiments/results/models"))
                    config["training"]["seed"] = int(seed)
                    if epochs is not None:
                        config["training"]["epochs"] = int(epochs)

                    clip_label = _format_number(clip_threshold)
                    run_label = f"{dataset}_clip{clip_label}_warmup{warmup_epoch}_seed{seed}"
                    config_path = output_dir / f"{run_label}.json"
                    config_path.write_text(
                        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    jobs.append(SweepJob(run_label=run_label, config_path=config_path))

    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description="Run CMAPSS DAST sweep experiments.")
    parser.add_argument("--config", default="config.json", help="Base config JSON.")
    parser.add_argument("--datasets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--clip-thresholds", nargs="+", type=float, default=DEFAULT_CLIP_THRESHOLDS)
    parser.add_argument("--warmup-epochs", nargs="+", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, help="Override training epochs for every run.")
    parser.add_argument("--limit", type=int, help="Run only the first N generated jobs.")
    parser.add_argument(
        "--generated-config-dir",
        default="experiments/generated_configs",
        help="Directory for generated per-run configs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    jobs = create_sweep_jobs(
        base_config=base_config,
        output_dir=Path(args.generated_config_dir),
        datasets=args.datasets,
        clip_thresholds=args.clip_thresholds,
        warmup_epochs=args.warmup_epochs,
        seeds=args.seeds,
        epochs=args.epochs,
        limit=args.limit,
    )

    for job in jobs:
        print(f"=== {job.run_label} ===", flush=True)
        subprocess.run(
            [sys.executable, "data_process.py", "--config", str(job.config_path)],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "DAST_test.py",
                "--config",
                str(job.config_path),
                "--run-label",
                job.run_label,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
