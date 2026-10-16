"""
Centralized MLP training script.
Trains a standard MLP on the preprocessed CIC-IDS2017 dataset.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time
import json
from src.data.preprocessing import CICIDS2017Preprocessor
from src.models.mlp import MLPIDS, count_parameters


def train_centralized_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    hidden_dims: list = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    num_epochs: int = 20,
    device: str = 'auto'
):
    """
    Train a centralized MLP model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Trained model and training history
    """
    # Determine device
    if device == 'auto':
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = MLPIDS(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)

    print(f"Model architecture:\n{model}")
    print(f"Number of trainable parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary classification with logits
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': []
    }

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.float())

                val_loss += loss.item()

                probs = torch.sigmoid(outputs.squeeze())
                predicted = (probs >= 0.5).long()

                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                # Store for metrics calculation
                all_val_preds.append(predicted.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(batch_y.cpu().numpy())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total

        # Concatenate all batches for metric calculation
        val_preds = np.concatenate(all_val_preds)
        val_probs = np.concatenate(all_val_probs)
        val_labels = np.concatenate(all_val_labels)

        # Calculate additional metrics
        try:
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_roc_auc = roc_auc_score(val_labels, val_probs)
        except Exception as e:
            print(f"Warning: Could not calculate some metrics: {e}")
            val_precision = val_recall = val_f1 = val_roc_auc = 0.0

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} '
                  f'Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}')

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = torch.sigmoid(test_outputs.squeeze()).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)

        # Calculate final metrics
        accuracy = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds, zero_division=0)
        recall = recall_score(y_test, test_preds, zero_division=0)
        f1 = f1_score(y_test, test_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_test, test_probs)
        except:
            roc_auc = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_test, test_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print("\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"FNR:       {fnr:.4f}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print("="*50)

        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'confusion_matrix': cm.tolist(),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'training_time': training_time,
            'history': history
        }

    return model, results


def main():
    """Main function to train centralized MLP."""
    # Load preprocessed data
    print("Loading preprocessed data...")

    processed_dir = "/Users/admin/Desktop/federated-ids/data/processed"

    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"Loaded data:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")

    # Get input dimension
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Train the model
    model, results = train_centralized_mlp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=256,
        num_epochs=20
    )

    # Save the model
    model_dir = "/Users/admin/Desktop/federated-ids/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "centralized_mlp.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': [128, 64, 32],
        'dropout_rate': 0.3,
        'results': results
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save results
    results_path = os.path.join(model_dir, "centralized_mlp_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    return model, results


if __name__ == "__main__":
    main()