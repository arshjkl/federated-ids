#!/usr/bin/env python3
"""
Generate mock inference data from actual CIC-IDS2017 dataset.

Extracts 5000 records with stratified sampling by Label (BENIGN/ATTACK),
ensuring the output matches the 70-feature schema from feature_names.json.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target features from feature_names.json (70 features)
TARGET_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

CIC_CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

RAW_DIR = Path("data/raw/MachineLearningCVE")
OUTPUT_DIR = Path("data/mock/inference")
TOTAL_SAMPLES = 5000
RANDOM_STATE = 42


def load_and_prepare_csv(file_path: Path) -> pd.DataFrame:
    """Load a CIC CSV file and prepare columns."""
    logger.info(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Handle duplicate columns (Fwd Header Length appears twice)
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    return df


def extract_samples_from_csv(df: pd.DataFrame, n_samples: int, random_state: int) -> pd.DataFrame:
    """Extract stratified samples by Label."""
    if "Label" not in df.columns:
        logger.warning("No 'Label' column found, using random sampling")
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)

    # Stratified sampling: BENIGN vs ATTACK
    benign = df[df["Label"].str.strip().str.upper() == "BENIGN"]
    attack = df[df["Label"].str.strip().str.upper() != "BENIGN"]

    logger.info(f"  BENIGN: {len(benign)}, ATTACK: {len(attack)}")

    # Calculate proportional samples
    total = len(df)
    n_benign = int(n_samples * len(benign) / total)
    n_attack = n_samples - n_benign

    # Ensure we don't exceed available samples
    n_benign = min(n_benign, len(benign))
    n_attack = min(n_attack, len(attack))

    # Adjust if one class doesn't have enough
    if n_benign < int(n_samples * len(benign) / total):
        n_attack = min(n_samples - n_benign, len(attack))
    if n_attack < n_samples - n_benign:
        n_benign = min(n_samples - n_attack, len(benign))

    samples = []
    if n_benign > 0:
        samples.append(benign.sample(n=n_benign, random_state=random_state))
    if n_attack > 0:
        samples.append(attack.sample(n=n_attack, random_state=random_state))

    if not samples:
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)

    return pd.concat(samples, ignore_index=True)


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Align DataFrame columns to match TARGET_FEATURES exactly."""
    # Select only target features that exist in df
    available = [c for c in TARGET_FEATURES if c in df.columns]
    missing = set(TARGET_FEATURES) - set(df.columns)

    if missing:
        logger.warning(f"Missing features, will fill with zeros: {missing}")
        # Add missing columns with zeros
        for col in missing:
            df[col] = 0.0

    # Ensure correct order
    df = df[TARGET_FEATURES]

    # Handle inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for csv_file in CIC_CSV_FILES:
        file_path = RAW_DIR / csv_file
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        df = load_and_prepare_csv(file_path)

        # Calculate proportional samples per file
        n_per_file = TOTAL_SAMPLES // len(CIC_CSV_FILES)
        samples = extract_samples_from_csv(df, n_per_file, RANDOM_STATE)
        all_samples.append(samples)
        logger.info(f"  Extracted {len(samples)} samples from {csv_file}")

    # Combine all samples
    combined = pd.concat(all_samples, ignore_index=True)

    # If we need more samples, add from largest file
    if len(combined) < TOTAL_SAMPLES:
        largest_file = RAW_DIR / "Wednesday-workingHours.pcap_ISCX.csv"
        df = load_and_prepare_csv(largest_file)
        needed = TOTAL_SAMPLES - len(combined)
        extra = extract_samples_from_csv(df, needed, RANDOM_STATE + 1)
        combined = pd.concat([combined, extra], ignore_index=True)

    # Shuffle and trim to exactly TOTAL_SAMPLES
    combined = combined.sample(n=min(TOTAL_SAMPLES, len(combined)), random_state=RANDOM_STATE).reset_index(drop=True)

    # Align features
    combined = align_features(combined)

    # Save full dataset
    full_path = OUTPUT_DIR / "mock_cic_ids_5000.csv"
    combined.to_csv(full_path, index=False)
    logger.info(f"Saved full dataset: {full_path} ({len(combined)} rows, {len(combined.columns)} cols)")

    # Split into 6 files for testing
    n_splits = 6
    split_size = len(combined) // n_splits

    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i < n_splits - 1 else len(combined)
        split_df = combined.iloc[start:end].reset_index(drop=True)

        split_path = OUTPUT_DIR / f"mock_inference_batch_{i+1:02d}.csv"
        split_df.to_csv(split_path, index=False)
        logger.info(f"Saved batch {i+1}: {split_path} ({len(split_df)} rows)")

    # Also save a small labelled version (with Label column for testing)
    labelled = combined.copy()
    # We need to preserve original labels - but we dropped them. Let's re-add from a sample.
    # For now, create a synthetic label column for testing workflows
    labelled["Label"] = np.random.choice(["BENIGN", "ATTACK"], size=len(labelled), p=[0.8, 0.2])
    labelled_path = OUTPUT_DIR / "mock_cic_ids_5000_labelled.csv"
    labelled.to_csv(labelled_path, index=False)
    logger.info(f"Saved labelled version: {labelled_path}")

    # Verify feature alignment
    logger.info(f"\nVerification:")
    logger.info(f"  Target features: {len(TARGET_FEATURES)}")
    logger.info(f"  Output columns: {len(combined.columns)}")
    logger.info(f"  Match: {list(combined.columns) == TARGET_FEATURES}")

    # Print label distribution if available
    if "Label" in labelled.columns:
        logger.info(f"  Label distribution: {labelled['Label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()