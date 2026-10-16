"""
Preprocessing module for CIC-IDS2017 dataset.
Handles loading, cleaning, and preparing data for federated learning.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CICIDS2017Preprocessor:
    """
    Preprocessor for CIC-IDS2017 dataset.
    Handles loading multiple CSV files, cleaning, and preparing data.
    """

    def __init__(self, data_dir: str, random_state: int = 42):
        """
        Initialize the preprocessor.

        Args:
            data_dir: Path to directory containing CSV files
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
        self.label_mapping = None
        self.constant_features = None

        # Set random seeds for reproducibility
        np.random.seed(random_state)

    def discover_csv_files(self) -> List[Path]:
        """
        Discover all CSV files in the data directory.

        Returns:
            List of paths to CSV files
        """
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        return csv_files

    def load_and_concatenate_data(self) -> pd.DataFrame:
        """
        Load all CSV files and concatenate them into a single DataFrame.

        Returns:
            Concatenated DataFrame with all data
        """
        csv_files = self.discover_csv_files()
        dataframes = []

        print(f"Loading {len(csv_files)} CSV files...")

        for i, file_path in enumerate(csv_files, 1):
            print(f"  Loading file {i}/{len(csv_files)}: {file_path.name}")

            try:
                # Read CSV with proper encoding to handle BOM
                df = pd.read_csv(file_path, encoding='utf-8')

                # Strip whitespace from column names
                df.columns = df.columns.str.strip()

                # Handle BOM in column names (remove any BOM characters)
                df.columns = df.columns.str.replace('﻿', '', regex=False)

                dataframes.append(df)
                print(f"    -> Shape: {df.shape}")

            except Exception as e:
                print(f"    -> Error loading {file_path.name}: {e}")
                # Try with different encoding if utf-8 fails
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    df.columns = df.columns.str.strip()
                    df.columns = df.columns.str.replace('﻿', '', regex=False)
                    dataframes.append(df)
                    print(f"    -> Shape: {df.shape} (loaded with utf-8-sig)")
                except Exception as e2:
                    print(f"    -> Failed to load {file_path.name} with both encodings: {e2}")
                    continue

        if not dataframes:
            raise ValueError("No CSV files could be loaded successfully")

        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nCombined dataset shape: {combined_df.shape}")

        return combined_df

    def clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize labels.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned labels
        """
        # Find the label column (case-insensitive)
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if not label_cols:
            raise ValueError("No label column found in dataset")

        label_col = label_cols[0]
        print(f"Using '{label_col}' as label column")

        # Strip whitespace from labels
        df[label_col] = df[label_col].astype(str).str.strip()

        return df

    def convert_to_binary_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Convert multiclass labels to binary: BENIGN=0, ATTACK=1.

        Args:
            df: Input DataFrame with labels

        Returns:
            Tuple of (DataFrame with binary labels, label mapping dictionary)
        """
        # Find the label column
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        label_col = label_cols[0]

        # Create label mapping: BENIGN -> 0, everything else -> 1
        unique_labels = df[label_col].unique()
        print(f"Unique labels found: {unique_labels}")

        label_mapping = {}
        for label in unique_labels:
            label_str = str(label).strip().upper()
            if label_str == 'BENIGN':
                label_mapping[label] = 0
            else:
                label_mapping[label] = 1

        print(f"Label mapping: {label_mapping}")

        # Apply mapping
        df['binary_label'] = df[label_col].map(label_mapping)

        # Check for any unmapped labels
        null_count = df['binary_label'].isnull().sum()
        if null_count > 0:
            print(f"Warning: {null_count} labels could not be mapped and will be set to 0 (BENIGN)")
            df['binary_label'] = df['binary_label'].fillna(0).astype(int)
        else:
            df['binary_label'] = df['binary_label'].astype(int)

        # Store the label mapping
        self.label_mapping = label_mapping

        return df, label_mapping

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, infinities, duplicates, etc.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_shape = df.shape
        print(f"Initial shape: {initial_shape}")

        # Replace +/- infinity with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        print("Replaced infinity values with NaN")

        # Remove duplicate rows
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        duplicates_after = df.duplicated().sum()
        print(f"Removed {duplicates_before} duplicate rows")

        # Separate features and label for processing
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if 'binary_label' in df.columns:
            label_cols.append('binary_label')

        feature_cols = [col for col in df.columns if col not in label_cols]

        # Handle missing values in features only
        print(f"Missing values in features before cleaning: {df[feature_cols].isnull().sum().sum()}")

        # For numeric columns, fill NaN with median (more robust than mean)
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # For non-numeric columns that should be numeric, try to convert
        # Then remove rows that still can't be converted
        non_numeric_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_features:
            # Try to convert to numeric, coercing errors to NaN
            original_col = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Report conversion issues
            na_count = df[col].isnull().sum()
            if na_count > 0:
                print(f"Column '{col}': {na_count} non-numeric values converted to NaN")

        # Now handle any remaining NaN values (from conversion or original data)
        # Fill with median for numeric features
        for col in df[feature_cols].select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Drop any rows that still have NaN in features (should be few after above processing)
        nan_before = df[feature_cols].isnull().sum().sum()
        df = df.dropna(subset=feature_cols)
        nan_after = df[feature_cols].isnull().sum().sum()
        print(f"Removed {nan_before - nan_after} rows with NaN in features")

        # Remove constant features (features with zero variance)
        numeric_feature_df = df[feature_cols].select_dtypes(include=[np.number])
        if len(numeric_feature_df.columns) > 0:
            variances = numeric_feature_df.var()
            constant_features = variances[variances == 0].index.tolist()

            if constant_features:
                print(f"Removing {len(constant_features)} constant features: {constant_features[:10]}{'...' if len(constant_features) > 10 else ''}")
                self.constant_features = constant_features
                df = df.drop(columns=constant_features)
                # Update feature cols after removing constants
                feature_cols = [col for col in df.columns if col not in label_cols]
            else:
                self.constant_features = []
                print("No constant features found")
        else:
            self.constant_features = []
            print("No numeric features found")

        final_shape = df.shape
        print(f"Final shape after cleaning: {final_shape}")
        print(f"Removed {initial_shape[0] - final_shape[0]} rows and {initial_shape[1] - final_shape[1]} columns")

        return df

    def prepare_features_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features (X) and labels (y) for machine learning.

        Args:
            df: Cleaned DataFrame

        Returns:
            Tuple of (features array, labels array, feature names list)
        """
        # Identify feature and label columns
        label_cols = [col for col in df.columns if 'label' in col.lower()]
        if 'binary_label' in df.columns:
            # Use our binary label for training
            label_col = 'binary_label'
        else:
            label_col = label_cols[0]

        feature_cols = [col for col in df.columns if col not in label_cols]

        # Ensure all features are numeric
        feature_df = df[feature_cols].select_dtypes(include=[np.number])

        # If we lost columns during selection, warn about it
        if len(feature_df.columns) < len(feature_cols):
            dropped_cols = set(feature_cols) - set(feature_df.columns)
            print(f"Warning: Dropped non-numeric columns: {dropped_cols}")

        X = feature_df.values
        y = df[label_col].values
        feature_names = list(feature_df.columns)

        # Store feature names
        self.feature_names = feature_names

        print(f"Prepared features: {X.shape}")
        print(f"Prepared labels: {y.shape}")
        print(f"Feature names: {len(feature_names)} features")

        return X, y, feature_names

    def split_and_scale_data(self, X: np.ndarray, y: np.ndarray,
                           test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets and scale features.

        CRITICAL: Fit scaler only on training data to prevent data leakage.

        Args:
            X: Feature array
            y: Label array
            test_size: Proportion of data for test set

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )

        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Test label distribution: {np.bincount(y_test)}")

        # Fit scaler on training data only
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Features standardized using training data statistics")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_artifacts(self, output_dir: str):
        """
        Save preprocessing artifacts for later use.

        Args:
            output_dir: Directory to save artifacts
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save scaler
        if self.scaler is not None:
            with open(output_path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        # Save feature names
        if self.feature_names is not None:
            with open(output_path / "feature_names.json", "w") as f:
                json.dump(self.feature_names, f, indent=2)

        # Save label mapping
        if self.label_mapping is not None:
            with open(output_path / "label_mapping.json", "w") as f:
                json.dump(self.label_mapping, f, indent=2)

        # Save constant features
        if self.constant_features is not None:
            with open(output_path / "constant_features.json", "w") as f:
                json.dump(self.constant_features, f, indent=2)

        print(f"Preprocessing artifacts saved to {output_path}")

    def load_artifacts(self, input_dir: str):
        """
        Load preprocessing artifacts.

        Args:
            input_dir: Directory containing artifacts
        """
        input_path = Path(input_dir)

        # Load scaler
        scaler_path = input_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Load feature names
        feature_names_path = input_path / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, "r") as f:
                self.feature_names = json.load(f)

        # Load label mapping
        label_mapping_path = input_path / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, "r") as f:
                self.label_mapping = json.load(f)

        # Load constant features
        constant_features_path = input_path / "constant_features.json"
        if constant_features_path.exists():
            with open(constant_features_path, "r") as f:
                self.constant_features = json.load(f)

        print(f"Preprocessing artifacts loaded from {input_path}")

    def preprocess_pipeline(self, test_size: float = 0.2,
                          artifacts_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete preprocessing pipeline.

        Args:
            test_size: Proportion of data for test set
            artifacts_dir: Directory to save/load artifacts (if None, only saves)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("=" * 60)
        print("Starting CIC-IDS2017 Preprocessing Pipeline")
        print("=" * 60)

        # Step 1: Load data
        print("\n1. Loading and concatenating CSV files...")
        df = self.load_and_concatenate_data()

        # Step 2: Clean labels
        print("\n2. Cleaning labels...")
        df = self.clean_labels(df)

        # Step 3: Convert to binary labels
        print("\n3. Converting to binary labels (BENIGN=0, ATTACK=1)...")
        df, label_mapping = self.convert_to_binary_labels(df)

        # Step 4: Clean data
        print("\n4. Cleaning data (NaN, inf, duplicates, constant features)...")
        df = self.clean_data(df)

        # Step 5: Prepare features and labels
        print("\n5. Preparing features and labels...")
        X, y, feature_names = self.prepare_features_and_labels(df)

        # Step 6: Split and scale
        print("\n6. Splitting data and scaling features...")
        X_train, X_test, y_train, y_test = self.split_and_scale_data(X, y, test_size)

        # Step 7: Save artifacts if directory provided
        if artifacts_dir:
            print(f"\n7. Saving preprocessing artifacts to {artifacts_dir}...")
            self.save_artifacts(artifacts_dir)

        print("\n" + "=" * 60)
        print("Preprocessing Pipeline Completed Successfully!")
        print("=" * 60)

        return X_train, X_test, y_train, y_test


def main():
    """Main function to run preprocessing."""
    # Initialize preprocessor
    preprocessor = CICIDS2017Preprocessor(
        data_dir="/Users/admin/Desktop/federated-ids/data/raw/MachineLearningCVE",
        random_state=42
    )

    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        test_size=0.2,
        artifacts_dir="/Users/admin/Desktop/federated-ids/data/processed"
    )

    # Save processed data as numpy arrays
    processed_dir = Path("/Users/admin/Desktop/federated-ids/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_test.npy", y_test)

    print(f"\nSaved processed data to {processed_dir}:")
    print(f"  X_train.npy: {X_train.shape}")
    print(f"  X_test.npy: {X_test.shape}")
    print(f"  y_train.npy: {y_train.shape}")
    print(f"  y_test.npy: {y_test.shape}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()