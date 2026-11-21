"""
Preprocessing service for loading scaler, feature names, and label mappings.
Loads artifacts from data/processed/ (READ-ONLY).
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PreprocessingLoader:
    """Loads and applies preprocessing artifacts."""

    def __init__(self, artifacts_dir: str = "data/processed"):
        self.artifacts_dir = Path(artifacts_dir)
        if not self.artifacts_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            if not self.artifacts_dir.exists() and (project_root / self.artifacts_dir).exists():
                self.artifacts_dir = project_root / self.artifacts_dir
        self._scaler = None
        self._feature_names: Optional[List[str]] = None
        self._label_mapping: Optional[Dict[str, int]] = None
        self._constant_features: Optional[List[str]] = None

    def load_all(self) -> Dict[str, Any]:
        """Load all preprocessing artifacts."""
        self._load_scaler()
        self._load_feature_names()
        self._load_label_mapping()
        self._load_constant_features()

        return {
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "label_mapping": self._label_mapping,
            "constant_features": self._constant_features,
        }

    def _load_scaler(self):
        scaler_path = self.artifacts_dir / "scaler.pkl"
        if not scaler_path.exists():
            from sklearn.preprocessing import StandardScaler
            logger.warning(f"Scaler not found at {scaler_path}, synthesizing standard 70-feature fallback scaler")
            scaler = StandardScaler()
            scaler.fit(np.zeros((2, 70)))
            self._scaler = scaler
            return
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")

    def _load_feature_names(self):
        path = self.artifacts_dir / "feature_names.json"
        if not path.exists():
            logger.warning(f"Feature names not found at {path}, using default CICIDS 70 features")
            from src.inference.feature_derivation import CICIDS70_FEATURES
            self._feature_names = [f.canonical_id for f in CICIDS70_FEATURES if f.cicids_index is not None]
            return
        with open(path) as f:
            self._feature_names = json.load(f)
        logger.info(f"Loaded {len(self._feature_names)} feature names")

    def _load_label_mapping(self):
        path = self.artifacts_dir / "label_mapping.json"
        if not path.exists():
            logger.warning(f"Label mapping not found at {path}, using standard binary labels")
            self._label_mapping = {"BENIGN": 0, "ATTACK": 1}
            return
        with open(path) as f:
            self._label_mapping = json.load(f)
        logger.info(f"Loaded label mapping: {self._label_mapping}")

    def _load_constant_features(self):
        path = self.artifacts_dir / "constant_features.json"
        if path.exists():
            with open(path) as f:
                self._constant_features = json.load(f)
            logger.info(f"Loaded {len(self._constant_features)} constant features")
        else:
            self._constant_features = []
            logger.warning("Constant features file not found")

    @property
    def scaler(self):
        if self._scaler is None:
            self._load_scaler()
        return self._scaler

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            self._load_feature_names()
        return self._feature_names

    @property
    def label_mapping(self) -> Dict[str, int]:
        if self._label_mapping is None:
            self._load_label_mapping()
        return self._label_mapping

    @property
    def constant_features(self) -> List[str]:
        if self._constant_features is None:
            self._load_constant_features()
        return self._constant_features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw dataframe to scaled features."""
        # Normalize column names (strip whitespace) so real CSVs match feature_names.json
        df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

        # De-duplicate column names by appending .1, .2, ... so CIC CSVs that repeat a
        # column (e.g. 'Fwd Header Length' twice) still match feature_names.json
        # (which lists 'Fwd Header Length' and 'Fwd Header Length.1').
        seen: Dict[str, int] = {}
        new_cols: List[str] = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}.{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols

        # Ensure all features present
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Select and order features
        X = df[self.feature_names].values

        # Handle inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Transform raw numpy array (already in correct feature order)."""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X)

    def inverse_label(self, label: int) -> str:
        """Convert numeric label to string."""
        inv = {v: k for k, v in self.label_mapping.items()}
        return inv.get(label, "UNKNOWN")


# Global instance
_preprocessing_loader: Optional[PreprocessingLoader] = None


def get_preprocessing_loader(artifacts_dir: str = "data/processed") -> PreprocessingLoader:
    """Get or create global preprocessing loader."""
    global _preprocessing_loader
    if _preprocessing_loader is None:
        _preprocessing_loader = PreprocessingLoader(artifacts_dir)
    return _preprocessing_loader