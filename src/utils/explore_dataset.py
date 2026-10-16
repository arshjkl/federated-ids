#!/usr/bin/env python3
"""
Utility script to explore the CIC-IDS2017 dataset structure.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def explore_dataset():
    """Explore the CIC-IDS2017 dataset."""
    data_dir = Path("/Users/admin/Desktop/federated-ids/data/raw/MachineLearningCVE")

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return

    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:")

    total_rows = 0
    for i, file in enumerate(csv_files, start=1):
        print(f"\n{i}. {file.name}")
        try:
            # Read just the header and a few rows to get basic info
            df_sample = pd.read_csv(file, nrows=5)
            print(f"   Columns: {len(df_sample.columns)}")

            # Get row count efficiently
            df_rows = pd.read_csv(file, usecols=[0])  # Just first column to count rows
            rows = len(df_rows)
            total_rows += rows
            print(f"   Rows: {rows:,}")

            # Show column names (first 10)
            cols = list(df_sample.columns)
            print(f"   First 10 columns: {cols[:10]}")

            # Check for Label column
            label_cols = [col for col in df_sample.columns if 'label' in col.lower()]
            print(f"   Label-related columns: {label_cols}")

        except Exception as e:
            print(f"   Error reading file: {e}")

    print(f"\nTotal rows across all files: {total_rows:,}")

if __name__ == "__main__":
    explore_dataset()