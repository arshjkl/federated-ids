"""
Quick training script to verify MLP training works.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from src.models.mlp import MLPIDS, count_parameters


def quick_train():
    """Quick training test."""
    # Load preprocessed data
    processed_dir = "/Users/admin/Desktop/federated-ids/data/processed"

    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    # Use smaller subsets for quick testing
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]

    print(f"Using subset:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    model = MLPIDS(
        input_dim=input_dim,
        hidden_dims=[64, 32],
        dropout_rate=0.2
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.float())
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = torch.sigmoid(test_outputs.squeeze()).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)

        print(f"\nResults on test set:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

    return model


if __name__ == "__main__":
    quick_train()