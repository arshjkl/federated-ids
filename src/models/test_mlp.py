"""
Test script to verify MLP model works with preprocessed data.
"""

import os
import numpy as np
import torch
from src.models.mlp import MLPIDS, count_parameters


def test_mlp_with_data():
    """Test MLP model with actual preprocessed data."""
    # Load preprocessed data
    processed_dir = "/Users/admin/Desktop/federated-ids/data/processed"

    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"Data shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    # Get input dimension
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Create model
    model = MLPIDS(input_dim=input_dim, hidden_dims=[64, 32], dropout_rate=0.2)
    print(f"\nModel:")
    print(model)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train[:100])  # Test with first 100 samples
        y_tensor = torch.LongTensor(y_train[:100])

        # Forward pass
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        print(f"\nForward pass test:")
        print(f"  Input shape: {X_tensor.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Predictions shape: {preds.shape}")

        # Calculate accuracy on this batch
        accuracy = (preds.squeeze() == y_tensor).float().mean().item()
        print(f"  Batch accuracy: {accuracy:.4f}")

        # Show some predictions
        print(f"\nSample predictions:")
        for i in range(min(5, len(preds))):
            print(f"  Sample {i}: True={y_tensor[i].item()}, "
                  f"Pred={preds[i].item()}, Prob_ATTACK={probs[i].item():.4f}")

    return model


if __name__ == "__main__":
    test_mlp_with_data()