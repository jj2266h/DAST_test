import unittest

import numpy as np

from experiments.prepare_splits import SENSOR_COLS, scale_sensors, split_stem


def make_raw(n=200, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 26))
    labels = rng.integers(0, 3, size=n)
    raw[:, SENSOR_COLS] += labels[:, None] * 50.0  # condition offsets
    return raw, labels


class ScaleSensorsTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.labels = make_raw()
        self.train = np.arange(len(self.raw)) < 140

    def test_statistics_come_from_train_rows_only(self):
        for norm in ("oc_z", "global_z", "global_minmax"):
            perturbed = self.raw.copy()
            perturbed[~self.train, SENSOR_COLS] += 1000.0
            a = scale_sensors(self.raw, self.train, self.labels, 3, norm)
            b = scale_sensors(perturbed, self.train, self.labels, 3, norm)
            np.testing.assert_allclose(a[self.train], b[self.train], err_msg=norm)

    def test_global_z_is_standardized_on_train(self):
        s = scale_sensors(self.raw, self.train, self.labels, 3, "global_z")[self.train, SENSOR_COLS]
        np.testing.assert_allclose(s.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(s.std(axis=0), 1, atol=1e-9)

    def test_global_minmax_spans_unit_interval_on_train(self):
        s = scale_sensors(self.raw, self.train, self.labels, 3, "global_minmax")[self.train, SENSOR_COLS]
        np.testing.assert_allclose(s.min(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(s.max(axis=0), 1, atol=1e-9)

    def test_oc_z_removes_condition_offsets(self):
        s = scale_sensors(self.raw, self.train, self.labels, 3, "oc_z")
        for c in range(3):
            m = self.train & (self.labels == c)
            np.testing.assert_allclose(s[m, SENSOR_COLS].mean(axis=0), 0, atol=1e-9)

    def test_global_norms_keep_condition_offsets(self):
        s = scale_sensors(self.raw, self.train, self.labels, 3, "global_z")
        means = [s[self.labels == c, SENSOR_COLS].mean() for c in range(3)]
        self.assertGreater(means[2] - means[0], 1.0)

    def test_non_sensor_columns_untouched(self):
        s = scale_sensors(self.raw, self.train, self.labels, 3, "oc_z")
        np.testing.assert_array_equal(s[:, :5], self.raw[:, :5])


class SplitStemTests(unittest.TestCase):
    def test_oc_z_keeps_original_file_names(self):
        self.assertEqual(split_stem("FD004", "oc_z", "global"), "FD004_global")
        self.assertEqual(split_stem("FD004", "global_z", "global"), "FD004_global_z_global")


if __name__ == "__main__":
    unittest.main()
