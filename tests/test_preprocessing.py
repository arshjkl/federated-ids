"""
Unit tests for the CIC-IDS2017 preprocessing pipeline.

Runs against the saved processed arrays (no full re-preprocessing needed, so
tests stay fast) and exercises the integrity helpers in
src.data.preprocessing_validation.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from data.preprocessing_validation import (
    count_rows,
    discover_csv_files,
    validate_dataset_integrity,
)

PROJECT_ROOT = "/Users/admin/Desktop/federated-ids"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "MachineLearningCVE")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


class TestPreprocessingDiscovery(unittest.TestCase):
    """Raw CSV discovery and row counting."""

    def test_discover_csv_files(self):
        files = discover_csv_files(DATA_DIR)
        self.assertEqual(len(files), 8, "Expected 8 CIC-IDS2017 CSV files")
        names = [f.name for f in files]
        self.assertTrue(any("Monday-WorkingHours" in n for n in names))
        self.assertTrue(any("Friday-WorkingHours-Morning" in n for n in names))

    def test_count_rows_positive(self):
        files = discover_csv_files(DATA_DIR)
        for fp in files:
            rows = count_rows(fp)
            self.assertGreater(rows, 0, f"{fp.name} should contain data rows")


class TestPreprocessingIntegrity(unittest.TestCase):
    """End-to-end integrity checks over raw data + processed arrays."""

    def setUp(self):
        self.results = validate_dataset_integrity(
            data_dir=DATA_DIR, processed_dir=PROCESSED_DIR
        )

    def test_all_checks_pass(self):
        self.assertTrue(
            self.results["all_checks_passed"],
            f"Failed checks: {self.results['failed_checks']}",
        )

    def test_files_readable(self):
        self.assertTrue(self.results["checks"]["files_readable"]["passed"])

    def test_label_mapping_binary(self):
        self.assertTrue(self.results["checks"]["binary_mapping_consistent"]["passed"])
        self.assertEqual(set(self.results["label_mapping"].values()), {0, 1})

    def test_classes_present(self):
        self.assertTrue(self.results["checks"]["classes_present"]["passed"])

    def test_constant_features_removed(self):
        self.assertEqual(self.results["num_constant_features"], 8)

    def test_processed_arrays_present(self):
        self.assertTrue(self.results["checks"]["processed_arrays_present"]["passed"])

    def test_processed_arrays_clean(self):
        self.assertTrue(self.results["checks"]["processed_arrays_clean"]["passed"])

    def test_stratified_split(self):
        self.assertTrue(self.results["checks"]["stratified_split"]["passed"])

    def test_scaler_fit_on_train_only(self):
        self.assertTrue(self.results["checks"]["scaler_fit_on_train_only"]["passed"])


class TestProcessedArrayShapes(unittest.TestCase):
    """Saved processed arrays must have consistent, expected shapes."""

    def test_array_shapes(self):
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"), mmap_mode="r")
        X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"), mmap_mode="r")
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"), mmap_mode="r")
        y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"), mmap_mode="r")

        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_train.shape[0], len(y_train))
        self.assertEqual(X_test.shape[0], len(y_test))
        self.assertEqual(X_train.shape[1], 70)


if __name__ == "__main__":
    unittest.main()
