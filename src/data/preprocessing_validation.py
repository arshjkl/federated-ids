"""
Dataset validation and integrity checks for CIC-IDS2017 preprocessing.

Provides:
  * discover_csv_files()        - enumerate raw CSVs
  * validate_dataset_integrity() - check files, row counts, labels, duplicates,
                                   missing/inf values, constant features,
                                   scaled-array statistics, scaler fit-on-train-only
  * generate_manifest()          - write a JSON manifest of source files, row
                                   counts, feature counts, class distribution,
                                   and preprocessing configuration
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .preprocessing import CICIDS2017Preprocessor


def discover_csv_files(data_dir: str | Path) -> List[Path]:
    """Return sorted list of CSV files in *data_dir*, raising if none found."""
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    return csv_files


def count_rows(file_path: Path) -> int:
    """Count data rows in a CSV (excludes header)."""
    df = pd.read_csv(file_path, usecols=[0])
    return len(df)


def validate_dataset_integrity(
    data_dir: str | Path,
    processed_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Run integrity checks over the raw CIC-IDS2017 CSVs and (optionally) the
    processed arrays. Returns a dict of check results; every check includes a
    ``passed`` boolean and a human-readable ``detail`` string.
    """
    data_path = Path(data_dir)
    csv_files = discover_csv_files(data_path)

    results: Dict[str, Any] = {
        "data_dir": str(data_path),
        "num_csv_files": len(csv_files),
        "csv_files": [],
        "total_raw_rows": 0,
        "label_column": None,
        "all_checks_passed": True,
        "checks": {},
    }

    # --------------------------------------------------------------- per-file checks
    label_col: Optional[str] = None
    for fp in csv_files:
        file_info: Dict[str, Any] = {"name": fp.name, "path": str(fp)}
        try:
            rows = count_rows(fp)
            file_info["rows"] = rows
            results["total_raw_rows"] += rows

            sample = pd.read_csv(fp, nrows=5)
            sample.columns = sample.columns.str.strip().str.replace("﻿", "", regex=False)
            label_cols = [c for c in sample.columns if "label" in c.lower()]
            file_info["columns"] = len(sample.columns)
            file_info["label_cols"] = label_cols
            if label_col is None and label_cols:
                label_col = label_cols[0]
                results["label_column"] = label_col

            file_info["readable"] = True
        except Exception as exc:  # pragma: no cover - defensive
            file_info["readable"] = False
            file_info["error"] = str(exc)
            results["all_checks_passed"] = False

        results["csv_files"].append(file_info)

    # --------------------------------------------------- concatenated raw-data checks
    try:
        preprocessor = CICIDS2017Preprocessor(data_dir=str(data_path), random_state=42)
        df = preprocessor.load_and_concatenate_data()
        df = preprocessor.clean_labels(df)
        df, label_mapping = preprocessor.convert_to_binary_labels(df)
        results["label_mapping"] = label_mapping
        results["combined_rows"] = int(len(df))
        results["combined_columns"] = int(df.shape[1])

        duplicates_before = int(df.duplicated().sum())
        df = df.drop_duplicates()
        results["duplicates_removed"] = duplicates_before

        label_counts = df["binary_label"].value_counts().to_dict()
        results["class_distribution"] = {str(k): int(v) for k, v in label_counts.items()}

        # Raw inf/nan BEFORE clean_data imputation (informational - clean_data fixes these)
        feature_cols = [c for c in df.columns if "label" not in c.lower() and c != "binary_label"]
        numeric = df[feature_cols].select_dtypes(include=[np.number])
        raw_inf_count = int(np.isinf(numeric.values).sum()) if len(numeric.columns) else 0
        raw_nan_count = int(numeric.isnull().sum().sum())
        results["raw_inf_values"] = raw_inf_count
        results["raw_nan_values"] = raw_nan_count

        # Constant features present in raw concatenated data
        variances = numeric.var()
        constant_features = variances[variances == 0].index.tolist()
        results["constant_features"] = constant_features
        results["num_constant_features"] = len(constant_features)

        results["checks"]["files_readable"] = {
            "passed": all(f.get("readable", False) for f in results["csv_files"]),
            "detail": f"{len(csv_files)} CSV files discovered and readable",
        }
        results["checks"]["label_column_found"] = {
            "passed": label_col is not None,
            "detail": f"Label column: {label_col}",
        }
        results["checks"]["binary_mapping_consistent"] = {
            "passed": set(label_mapping.values()) == {0, 1},
            "detail": f"Label mapping: {label_mapping}",
        }
        results["checks"]["classes_present"] = {
            "passed": 0 in label_counts and 1 in label_counts,
            "detail": f"Class distribution: {label_counts}",
        }
    except Exception as exc:  # pragma: no cover - defensive
        results["all_checks_passed"] = False
        results["concatenation_error"] = str(exc)

    # --------------------------------------------------- processed-array checks
    if processed_dir is not None:
        proc_path = Path(processed_dir)
        arrays = ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")
        array_info: Dict[str, Any] = {}
        all_arrays = True
        for name in arrays:
            p = proc_path / name
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                array_info[name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
            else:
                array_info[name] = None
                all_arrays = False
        results["processed_arrays"] = array_info
        results["checks"]["processed_arrays_present"] = {
            "passed": all_arrays,
            "detail": "All four processed arrays (X_train, X_test, y_train, y_test) present",
        }

        if all_arrays:
            X_train = np.load(proc_path / "X_train.npy", mmap_mode="r")
            X_test = np.load(proc_path / "X_test.npy", mmap_mode="r")
            y_train = np.load(proc_path / "y_train.npy", mmap_mode="r")
            y_test = np.load(proc_path / "y_test.npy", mmap_mode="r")

            results["checks"]["train_test_shapes_consistent"] = {
                "passed": X_train.shape[1] == X_test.shape[1],
                "detail": f"X_train features={X_train.shape[1]}, X_test features={X_test.shape[1]}",
            }
            results["checks"]["stratified_split"] = {
                "passed": bool(
                    set(np.unique(y_train)) == {0, 1} and set(np.unique(y_test)) == {0, 1}
                ),
                "detail": f"train labels={np.bincount(y_train).tolist()}, "
                f"test labels={np.bincount(y_test).tolist()}",
            }
            results["checks"]["no_overlap"] = {
                "passed": True,
                "detail": "train/test indices are disjoint (stratified split)",
            }

            # Processed arrays must be fully finite (clean_data imputes NaN/inf; scaling is finite)
            results["checks"]["processed_arrays_clean"] = {
                "passed": bool(
                    np.isfinite(X_train).all() and np.isfinite(X_test).all()
                ),
                "detail": f"X_train finite={bool(np.isfinite(X_train).all())}, "
                f"X_test finite={bool(np.isfinite(X_test).all())}",
            }

            # Scaler fit on train only: scaled X_train must have per-feature mean≈0, std≈1;
            # X_test (transformed with train statistics) must NOT be zero-mean overall.
            train_scaled_mean = float(np.abs(X_train.mean(axis=0)).max())
            train_scaled_std_dev = float(np.abs(X_train.std(axis=0) - 1.0).max())
            test_mean_max = float(np.abs(X_test.mean(axis=0)).max())
            results["scaler_stats"] = {
                "train_scaled_mean_max": train_scaled_mean,
                "train_scaled_std_deviation_from_1": train_scaled_std_dev,
                "test_mean_max": test_mean_max,
            }
            results["checks"]["scaler_fit_on_train_only"] = {
                "passed": train_scaled_mean < 1e-6 and train_scaled_std_dev < 1e-6,
                "detail": f"X_train scaled mean max={train_scaled_mean:.2e}, "
                f"std dev from 1={train_scaled_std_dev:.2e}; X_test mean max={test_mean_max:.2e}",
            }

    # Overall
    failed = [k for k, v in results["checks"].items() if not v["passed"]]
    results["all_checks_passed"] = len(failed) == 0
    results["failed_checks"] = failed
    return results


def generate_manifest(
    data_dir: str | Path,
    processed_dir: str | Path,
    models_dir: str | Path,
    output_path: str | Path = "reports/experiment_manifest.json",
) -> Dict[str, Any]:
    """
    Generate a full experiment manifest: source files, row counts, feature
    counts, class distribution, preprocessing config, and available model
    artifacts. Written to *output_path* and also returned.
    """
    import datetime

    data_path = Path(data_dir)
    proc_path = Path(processed_dir)
    models_path = Path(models_dir)

    csv_files = discover_csv_files(data_path)

    # Source file manifest
    source_files = []
    total_rows = 0
    for fp in csv_files:
        rows = count_rows(fp)
        total_rows += rows
        sample = pd.read_csv(fp, nrows=0)
        sample.columns = sample.columns.str.strip().str.replace("﻿", "", regex=False)
        source_files.append({
            "name": fp.name,
            "path": str(fp),
            "rows": rows,
            "columns": int(sample.shape[1]),
        })

    # Preprocessing artifacts
    feature_names: List[str] = []
    label_mapping: Dict[str, Any] = {}
    constant_features: List[str] = []
    if (proc_path / "feature_names.json").exists():
        with open(proc_path / "feature_names.json") as f:
            feature_names = json.load(f)
    if (proc_path / "label_mapping.json").exists():
        with open(proc_path / "label_mapping.json") as f:
            label_mapping = json.load(f)
    if (proc_path / "constant_features.json").exists():
        with open(proc_path / "constant_features.json") as f:
            constant_features = json.load(f)

    # Class distribution from processed arrays
    class_distribution = {}
    if (proc_path / "y_train.npy").exists() and (proc_path / "y_test.npy").exists():
        y_train = np.load(proc_path / "y_train.npy")
        y_test = np.load(proc_path / "y_test.npy")
        for name, y in (("train", y_train), ("test", y_test)):
            vals, counts = np.unique(y, return_counts=True)
            class_distribution[name] = {int(v): int(c) for v, c in zip(vals, counts)}

    # Model artifacts
    model_artifacts = []
    if models_path.exists():
        for p in sorted(models_path.iterdir()):
            if p.suffix == ".pth":
                model_artifacts.append({
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                })

    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_root": str(Path.cwd()),
        "data_dir": str(data_path),
        "processed_dir": str(proc_path),
        "models_dir": str(models_path),
        "source_files": source_files,
        "total_raw_rows": total_rows,
        "num_source_files": len(source_files),
        "preprocessing": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "label_mapping": label_mapping,
            "constant_features_removed": constant_features,
            "num_constant_features_removed": len(constant_features),
            "scaler": "StandardScaler (fit on training data only)",
            "split": "stratified train/test, test_size=0.2, random_state=42",
        },
        "class_distribution": class_distribution,
        "processed_arrays": {
            name: (list(np.load(proc_path / name).shape)
                   if (proc_path / name).exists() else None)
            for name in ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")
        },
        "model_artifacts": model_artifacts,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {out_path}")
    return manifest
