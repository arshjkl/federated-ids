"""
MLP model for binary classification.
Implements a simple feedforward neural network for intrusion detection.
"""

import numpy as np
from typing import Tuple

# Try to import torch, but provide a fallback for testing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type annotations
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class BatchNorm1d:
            pass
        class ReLU:
            pass
        class Dropout:
            pass
        class Sequential:
            pass

    class F:
        @staticmethod
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))


class MLPIDS:
    """
    Multi-Layer Perceptron for Intrusion Detection System.
    Binary classification: BENIGN (0) vs ATTACK (1).
    """

    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.3, use_batchnorm: bool = True):
        """
        Initialize the MLP.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batchnorm: Whether to use batch normalization
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.TORCH_AVAILABLE = TORCH_AVAILABLE

        if TORCH_AVAILABLE:
            # PyTorch version
            super().__init__()  # Initialize nn.Module

            # Build layers
            layers = []
            prev_dim = input_dim

            for i, hidden_dim in enumerate(hidden_dims):
                # Linear layer
                layers.append(nn.Linear(prev_dim, hidden_dim))

                # Batch normalization (if enabled)
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                # Activation function
                layers.append(nn.ReLU())

                # Dropout
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                prev_dim = hidden_dim

            # Output layer (binary classification)
            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)
        else:
            # Fallback numpy version for testing
            self.layers = []  # Store layer information for forward pass
            prev_dim = input_dim

            for i, hidden_dim in enumerate(hidden_dims):
                # Store layer info: (in_dim, out_dim, has_batchnorm, has_dropout)
                self.layers.append({
                    'type': 'linear',
                    'in_dim': prev_dim,
                    'out_dim': hidden_dim,
                    'weight': np.random.randn(hidden_dim, prev_dim) * np.sqrt(2. / prev_dim),  # He init
                    'bias': np.zeros(hidden_dim),
                    'batchnorm': use_batchnorm,
                    'dropout': dropout_rate > 0
                })
                prev_dim = hidden_dim

            # Output layer
            self.layers.append({
                'type': 'linear',
                'in_dim': prev_dim,
                'out_dim': 1,
                'weight': np.random.randn(1, prev_dim) * np.sqrt(2. / prev_dim),
                'bias': np.zeros(1),
                'batchnorm': False,
                'dropout': False
            })

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor or array of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, 1)
        """
        if TORCH_AVAILABLE and hasattr(self, 'network'):
            # PyTorch version
            return self.network(x)
        else:
            # Fallback numpy version
            # Convert to numpy if needed
            if hasattr(x, 'cpu'):  # torch tensor
                x = x.detach().cpu().numpy()
            elif not isinstance(x, np.ndarray):
                x = np.array(x)

            # Forward pass through layers
            for layer in self.layers:
                if layer['type'] == 'linear':
                    # Linear transformation: y = xW^T + b
                    x = x @ layer['weight'].T + layer['bias']

                    # Batch normalization (simplified)
                    if layer['batchnorm'] and x.shape[0] > 1:
                        mean = np.mean(x, axis=0, keepdims=True)
                        var = np.var(x, axis=0, keepdims=True)
                        x = (x - mean) / np.sqrt(var + 1e-8)

                    # ReLU activation
                    x = np.maximum(0, x)

                    # Dropout (simplified - during training we'd randomly zero some units)
                    # For simplicity in inference/test, we'll skip dropout or scale
                    if layer['dropout']:
                        # During inference, we typically scale by (1 - dropout_rate)
                        # or just don't apply dropout. For simplicity, we'll skip.
                        pass

            return x

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Args:
            x: Input tensor or array of shape (batch_size, input_dim)

        Returns:
            Probabilities of shape (batch_size, 2) for [BENIGN, ATTACK]
        """
        logits = self.forward(x)
        if TORCH_AVAILABLE:
            # PyTorch version
            probs = torch.sigmoid(logits)
            # Return [P(BENIGN), P(ATTACK)]
            return torch.cat([1 - probs, probs], dim=1)
        else:
            # Fallback numpy version
            # Sigmoid function
            probs = 1 / (1 + np.exp(-logits))
            # Return [P(BENIGN), P(ATTACK)] as numpy array
            return np.hstack([1 - probs, probs])

    def predict(self, x, threshold: float = 0.5):
        """
        Predict class labels.

        Args:
            x: Input tensor or array of shape (batch_size, input_dim)
            threshold: Decision threshold for positive class

        Returns:
            Predicted labels of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        if TORCH_AVAILABLE:
            # PyTorch version
            return (probs[:, 1] >= threshold).long()
        else:
            # Fallback numpy version
            return (probs[:, 1] >= threshold).astype(int)


def create_mlp_model(input_dim: int, **kwargs) -> MLPIDS:
    """
    Factory function to create an MLP model.

    Args:
        input_dim: Number of input features
        **kwargs: Additional arguments to pass to MLPIDS constructor

    Returns:
        Initialized MLPIDS model
    """
    return MLPIDS(input_dim=input_dim, **kwargs)


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: MLPIDS model

    Returns:
        Number of trainable parameters
    """
    if TORCH_AVAILABLE and hasattr(model, 'network'):
        # PyTorch version
        return sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    else:
        # Fallback numpy version
        total_params = 0
        for layer in model.layers:
            if layer['type'] == 'linear':
                # Weight matrix: output_dim x input_dim
                total_params += layer['weight'].size
                # Bias vector: output_dim
                total_params += layer['bias'].size
        return total_params


if __name__ == "__main__":
    # Example usage
    input_dim = 70  # From our preprocessing
    model = create_mlp_model(input_dim=input_dim)
    print(f"MLP Model Architecture:")
    print(model)
    print(f"\nNumber of trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 128
    if TORCH_AVAILABLE:
        import torch
        x = torch.randn(batch_size, input_dim)
    else:
        x = np.random.randn(batch_size, input_dim)
    logits = model(x)
    probs = model.predict_proba(x)
    preds = model.predict(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Sample probabilities: {probs[:5]}")