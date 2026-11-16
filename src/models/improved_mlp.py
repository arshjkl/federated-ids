"""
Improved MLP model for binary classification with advanced architecture.
Implements residual connections, better regularization, and multiple improvements
for higher F1 score and precision.
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass
        class Linear: pass
        class BatchNorm1d: pass
        class LayerNorm: pass
        class ReLU: pass
        class GELU: pass
        class Dropout: pass
        class Sequential: pass
        class ModuleList: pass
        class Identity: pass
    class F:
        @staticmethod
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        @staticmethod
        def gelu(x): return x * 0.5 * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class ResidualBlock(nn.Module):
    """
    Residual block with optional layer normalization and dropout.
    Helps with gradient flow in deeper networks.
    """
    def __init__(self, dim: int, dropout_rate: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class ImprovedMLPIDS(nn.Module):
    """
    Improved Multi-Layer Perceptron for Intrusion Detection System.
    Features:
    - Residual connections for better gradient flow
    - GELU activation (better than ReLU for deep networks)
    - Layer Normalization (more stable than BatchNorm for small batches)
    - Configurable architecture
    - Better weight initialization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rate: float = 0.2,
        use_residual: bool = True,
        use_layernorm: bool = True,
        residual_every: int = 2,
        output_dim: int = 1
    ):
        """
        Initialize the Improved MLP.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_residual: Whether to use residual connections
            use_layernorm: Use LayerNorm instead of BatchNorm
            residual_every: Add residual block every N layers
            output_dim: Output dimension (1 for binary classification)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.residual_every = residual_every
        self.output_dim = output_dim

        # Input projection
        layers = []
        prev_dim = input_dim

        # Initial projection layer
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dims[0]))
        else:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.GELU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        prev_dim = hidden_dims[0]

        # Hidden layers with optional residual blocks
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            # Add residual block periodically
            if use_residual and i % residual_every == 0:
                layers.append(ResidualBlock(prev_dim, dropout_rate, use_layernorm))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                else:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He/Kaiming initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def predict_proba(self, x):
        """Predict class probabilities."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return torch.cat([1 - probs, probs], dim=1)

    def predict(self, x, threshold: float = 0.5):
        """Predict class labels."""
        probs = self.predict_proba(x)
        return (probs[:, 1] >= threshold).long()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy examples and focuses on hard-to-classify samples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: binary labels (0 or 1)
        bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for class imbalance.
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        return F.binary_cross_entropy_with_logits(
            inputs.squeeze(),
            targets.float(),
            pos_weight=torch.tensor(self.pos_weight, device=inputs.device)
        )


def create_improved_mlp_model(input_dim: int, **kwargs) -> ImprovedMLPIDS:
    """Factory function to create an improved MLP model."""
    return ImprovedMLPIDS(input_dim=input_dim, **kwargs)


def count_parameters(model) -> int:
    """Count the number of trainable parameters in a model."""
    if TORCH_AVAILABLE:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return 0


def get_optimal_threshold(model, X_val, y_val, device='cpu'):
    """
    Find the optimal threshold that maximizes F1 score on validation data.
    """
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        logits = model(X_val_tensor)
        probs = torch.sigmoid(logits.squeeze()).cpu().numpy()

    from sklearn.metrics import f1_score, precision_score, recall_score

    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    # Search thresholds from 0.1 to 0.9
    for threshold in np.arange(0.1, 0.95, 0.01):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_f1, best_precision, best_recall


if __name__ == "__main__":
    # Example usage
    input_dim = 70
    model = create_improved_mlp_model(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64, 32],
        dropout_rate=0.2,
        use_residual=True,
        use_layernorm=True,
        residual_every=2
    )
    print(f"Improved MLP Model Architecture:")
    print(model)
    print(f"\nNumber of trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    import torch
    batch_size = 128
    x = torch.randn(batch_size, input_dim)
    logits = model(x)
    probs = model.predict_proba(x)
    preds = model.predict(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Sample probabilities: {probs[:5]}")