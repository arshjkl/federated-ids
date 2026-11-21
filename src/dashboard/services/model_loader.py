"""
Model loading service for existing trained models.
Loads checkpoints from models/ directory (READ-ONLY).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Import model architectures from existing code (READ-ONLY)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.mlp import MLPIDS, create_mlp_model
from src.models.improved_mlp import ImprovedMLPIDS, create_improved_mlp_model, get_optimal_threshold


class ModelLoader:
    """Loads and manages existing trained models for inference."""

    # Known model paths (READ-ONLY artifacts)
    MODEL_PATHS = {
        "centralized_baseline": "models/centralized_mlp.pth",
        "centralized_improved": "models/centralized_mlp_improved.pth",
        "federated_manual": "models/federated_mlp_manual.pth",
        "federated_manual_v2": "models/manual_federated_mlp.pth",
        "dp_federated": "models/federated_mlp_dp.pth",
        "federated_v2_round_0": "models/checkpoints/federated_v2_round_0.pth",
        "federated_v2_round_10": "models/checkpoints/federated_v2_round_10.pth",
        "federated_v2_round_20": "models/checkpoints/federated_v2_round_20.pth",
    }

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self._models: Dict[str, Tuple[nn.Module, Dict[str, Any]]] = {}
        self._thresholds: Dict[str, float] = {}

    def load_model(self, model_key: str, force_reload: bool = False) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a model by key. Returns (model, metadata)."""
        if model_key in self._models and not force_reload:
            return self._models[model_key]

        if model_key not in self.MODEL_PATHS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(self.MODEL_PATHS.keys())}")

        model_path = Path(self.MODEL_PATHS[model_key])
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            if not model_path.exists() and (project_root / model_path).exists():
                model_path = project_root / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model {model_key} from {model_path}")

        # Load checkpoint (weights_only=False needed for checkpoints with numpy scalar metadata in PyTorch 2.6+)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Determine architecture from model key or checkpoint metadata
        is_improved = (
            "improved" in model_key
            or "federated_v2" in model_key
            or (
                isinstance(checkpoint, dict)
                and (
                    checkpoint.get("use_residual")
                    or checkpoint.get("hidden_dims") == [256, 128, 64, 32]
                )
            )
        )
        if is_improved:
            hidden_dims = (
                checkpoint.get("hidden_dims", [256, 128, 64, 32])
                if isinstance(checkpoint, dict)
                else [256, 128, 64, 32]
            )
            model = create_improved_mlp_model(input_dim=70, hidden_dims=hidden_dims)
            arch = "ImprovedMLPIDS"
        elif "dp" in model_key:
            model = create_mlp_model(input_dim=70, use_batchnorm=False)
            arch = "MLPIDS (no BatchNorm)"
        elif "manual_v2" in model_key:
            model = create_mlp_model(input_dim=70, use_batchnorm=True)
            arch = "MLPIDS"
        else:
            model = create_mlp_model(input_dim=70)
            arch = "MLPIDS"

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata = {
                'epoch': checkpoint.get('epoch'),
                'metrics': checkpoint.get('metrics', {}),
                'config': checkpoint.get('config', {}),
            }
        else:
            state_dict = checkpoint
            metadata = {}

        # Remap state_dict keys if they don't have "network." prefix
        # Some checkpoints were saved with just the Sequential network
        first_key = next(iter(state_dict.keys())) if state_dict else ""
        if first_key and not first_key.startswith("network."):
            # Remap keys: "0.weight" -> "network.0.weight"
            new_state_dict = {}
            for key, value in state_dict.items():
                new_state_dict[f"network.{key}"] = value
            state_dict = new_state_dict
            logger.info(f"Remapped state_dict keys for {model_key}")

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Get optimal threshold if available
        threshold = self._get_threshold(model_key, metadata)

        self._models[model_key] = (model, metadata)
        self._thresholds[model_key] = threshold

        logger.info(f"Loaded {arch} model: {model_key}, threshold={threshold:.4f}")
        return model, metadata

    def _get_threshold(self, model_key: str, metadata: Dict) -> float:
        """Get optimal threshold for model."""
        # Check metadata first
        if 'optimal_threshold' in metadata:
            return metadata['optimal_threshold']
        if 'threshold' in metadata:
            return metadata['threshold']

        # Default thresholds from evaluation results
        defaults = {
            "centralized_baseline": 0.5,
            "centralized_improved": 0.35,
            "federated_manual": 0.5,
            "federated_manual_v2": 0.5,
            "dp_federated": 0.5,
            "federated_v2_round_0": 0.5,
            "federated_v2_round_10": 0.5,
            "federated_v2_round_20": 0.5,
        }
        return defaults.get(model_key, 0.5)

    def get_threshold(self, model_key: str) -> float:
        """Get threshold for a loaded model."""
        if model_key not in self._thresholds:
            self.load_model(model_key)
        return self._thresholds[model_key]

    def predict(self, model_key: str, X: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor."""
        model, _ = self.load_model(model_key)
        with torch.no_grad():
            X = X.to(self.device)
            logits = model(X)
            probs = torch.sigmoid(logits)
        return probs.cpu()

    def predict_binary(self, model_key: str, X: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """Run inference and return binary predictions."""
        probs = self.predict(model_key, X)
        if threshold is None:
            threshold = self.get_threshold(model_key)
        return (probs >= threshold).int()

    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get model metadata."""
        _, metadata = self.load_model(model_key)
        return {
            "key": model_key,
            "path": self.MODEL_PATHS[model_key],
            "threshold": self._thresholds.get(model_key, 0.5),
            "metadata": metadata,
        }

    def list_available_models(self) -> Dict[str, str]:
        """List all available model keys and paths."""
        return {k: v for k, v in self.MODEL_PATHS.items() if Path(v).exists()}


# Global instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader(device: str = "cpu") -> ModelLoader:
    """Get or create global model loader."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(device)
    return _model_loader