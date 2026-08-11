import csv
import tempfile
import unittest
from pathlib import Path

from DAST_test import append_experiment_log


class ExperimentLogTests(unittest.TestCase):
    def test_append_experiment_log_expands_existing_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "experiment_log.csv"
            csv_path.write_text("run_id,dataset\nold,FD003\n", encoding="utf-8")

            append_experiment_log(
                str(csv_path),
                {"run_id": "new", "dataset": "FD004", "seed": 42},
            )

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]["run_id"], "old")
            self.assertEqual(rows[0]["seed"], "")
            self.assertEqual(rows[1]["run_id"], "new")
            self.assertEqual(rows[1]["seed"], "42")

    def test_append_experiment_log_handles_existing_extra_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "experiment_log.csv"
            csv_path.write_text("run_id,dataset\nold,FD003,extra\n", encoding="utf-8")

            append_experiment_log(
                str(csv_path),
                {"run_id": "new", "dataset": "FD004", "seed": 42},
            )

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]["run_id"], "old")
            self.assertEqual(rows[0]["dataset"], "FD003")
            self.assertEqual(rows[1]["seed"], "42")


if __name__ == "__main__":
    unittest.main()
