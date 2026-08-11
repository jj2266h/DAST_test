import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_cmapss_sweep import create_sweep_jobs


class SweepConfigTests(unittest.TestCase):
    def test_create_sweep_jobs_writes_limited_configs(self):
        base_config = {
            "dataset_type": "cmapss",
            "cmapss": {
                "dataset": "FD003",
                "data_path": "Cmapss_data",
                "output_path": "train_dataset",
                "window_size": 40,
                "rul_max": 125.0,
                "feature_len": 14,
            },
            "training": {
                "epochs": 100,
                "grad_clip_enabled": False,
                "grad_clip_max_norm": 1.0,
                "lr_warmup_enabled": False,
                "lr_warmup_epochs": 5,
                "model_save_path": "model",
                "seed": 42,
            },
            "model": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            jobs = create_sweep_jobs(
                base_config=base_config,
                output_dir=Path(tmp),
                datasets=["FD004"],
                clip_thresholds=[0.5, 2.0],
                warmup_epochs=[5],
                seeds=[42, 20],
                epochs=1,
                limit=1,
            )

            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual(job.run_label, "FD004_clip0.5_warmup5_seed42")
            self.assertTrue(job.config_path.exists())

            generated = json.loads(job.config_path.read_text(encoding="utf-8"))
            self.assertEqual(generated["cmapss"]["dataset"], "FD004")
            self.assertEqual(generated["training"]["epochs"], 1)
            self.assertTrue(generated["training"]["grad_clip_enabled"])
            self.assertEqual(generated["training"]["grad_clip_max_norm"], 0.5)
            self.assertTrue(generated["training"]["lr_warmup_enabled"])
            self.assertEqual(generated["training"]["lr_warmup_epochs"], 5)
            self.assertEqual(generated["training"]["seed"], 42)
            self.assertEqual(
                generated["training"]["model_save_path"],
                str(Path("experiments/results/models")),
            )


if __name__ == "__main__":
    unittest.main()
