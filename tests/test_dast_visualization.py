import os
import tempfile
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from DAST_visualization import plot_training_results


class PlotTrainingResultsTests(unittest.TestCase):
    def test_writes_all_six_current_plot_files(self):
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "rmse": [12.0, 10.0, 11.0],
            "score": [30.0, 20.0, 25.0],
        }
        pred = np.array([0.2, 0.5, 0.8, 0.9])
        true = np.array([0.1, 0.4, 0.7, 1.0])
        oc_clusters = np.array([0, 0, 1, 1])
        expected_suffixes = [
            "01_train_loss.png",
            "02_val_test_rmse.png",
            "03_nasa_score.png",
            "04_best_pred_vs_true_rul.png",
            "05_sorted_prediction_curve.png",
            "06_oc_cluster_rmse_score.png",
        ]

        with tempfile.TemporaryDirectory() as save_dir:
            paths = plot_training_results(
                history,
                pred,
                true,
                125,
                "FD_TEST",
                save_dir,
                oc_clusters=oc_clusters,
            )

            self.assertEqual(6, len(paths))
            self.assertTrue(all(os.path.isfile(path) for path in paths))
            self.assertTrue(
                all(path.endswith(suffix) for path, suffix in zip(paths, expected_suffixes))
            )


if __name__ == "__main__":
    unittest.main()
