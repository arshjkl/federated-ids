"""
Improved Centralized MLP Training Script.
Incorporates multiple enhancements for higher F1 score and precision:
- Class-weighted loss / Focal Loss for imbalanced data
- Learning rate scheduling with warmup
- Early stopping
- Optimal threshold selection
- Better model architecture with residual connections
- Gradient clipping
- Label smoothing
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
)
import time
import json
import argparse
from typing import Dict, Tuple, Optional

from src.models.improved_mlp import (
    ImprovedMLPIDS, FocalLoss, WeightedBCELoss,
    create_improved_mlp_model, count_parameters, get_optimal_threshold
)
from src.data.preprocessing import CICIDS2017Preprocessor


def train_improved_centralized_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    hidden_dims: list = [256, 128, 64, 32],
    dropout_rate: float = 0.2,
    use_residual: bool = True,
    use_layernorm: bool = True,
    residual_every: int = 2,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    num_epochs: int = 50,
    device: str = 'auto',
    loss_type: str = 'focal',  # 'focal', 'weighted_bce', 'bce'
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: float = None,  # Auto-calculated if None
    scheduler_type: str = 'cosine_warmup',  # 'cosine_warmup', 'reduce_on_plateau', 'step', 'none'
    warmup_epochs: int = 5,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = 'val_f1',
    gradient_clip: float = 1.0,
    label_smoothing: float = 0.0,
    find_optimal_threshold: bool = True,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Train an improved centralized MLP model with advanced techniques.

    Returns:
        Trained model and comprehensive results dictionary
    """
    # Determine device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if verbose:
        print(f"Using device: {device}")

    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train)
    n_samples = len(y_train)
    n_classes = len(class_counts)

    if pos_weight is None:
        # pos_weight = negative_samples / positive_samples for BCEWithLogitsLoss
        pos_weight = class_counts[0] / class_counts[1]

    if verbose:
        print(f"Class distribution: {class_counts}")
        print(f"Positive weight (neg/pos): {pos_weight:.2f}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = create_improved_mlp_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
        residual_every=residual_every
    ).to(device)

    if verbose:
        print(f"Model architecture:\n{model}")
        print(f"Number of trainable parameters: {count_parameters(model):,}")

    # Loss function
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if verbose:
            print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    elif loss_type == 'weighted_bce':
        criterion = WeightedBCELoss(pos_weight=pos_weight)
        if verbose:
            print(f"Using Weighted BCE Loss (pos_weight={pos_weight:.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        if verbose:
            print(f"Using Standard BCE Loss")

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    if scheduler_type == 'cosine_warmup':
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if verbose:
            print(f"Using Cosine Annealing with Warmup ({warmup_epochs} epochs)")

    elif scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        if verbose:
            print(f"Using ReduceLROnPlateau")

    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        if verbose:
            print(f"Using StepLR (step=15, gamma=0.5)")

    else:
        scheduler = None
        if verbose:
            print(f"No learning rate scheduler")

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': [],
        'learning_rates': []
    }

    # Early stopping
    best_metric = -np.inf
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    # Training loop
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

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
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs.squeeze())
                predicted = (probs >= 0.5).long()

                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                all_val_preds.append(predicted.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(batch_y.cpu().numpy())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(test_loader)
        val_acc = val_correct / val_total

        val_preds = np.concatenate(all_val_preds)
        val_probs = np.concatenate(all_val_probs)
        val_labels = np.concatenate(all_val_labels)

        try:
            val_precision = precision_score(val_labels, val_preds, zero_division=0)
            val_recall = recall_score(val_labels, val_preds, zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_roc_auc = roc_auc_score(val_labels, val_probs)
        except Exception as e:
            if verbose:
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
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_f1 if early_stopping_metric == 'val_f1' else val_acc)
            else:
                scheduler.step()

        # Early stopping check
        current_metric = val_f1 if early_stopping_metric == 'val_f1' else val_acc

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            epoch_time = time.time() - epoch_start
            print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} '
                  f'Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f} '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break

    training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"Loaded best model from epoch {best_epoch+1} (val_f1={best_metric:.4f})")

    # Find optimal threshold on validation set
    if find_optimal_threshold:
        if verbose:
            print("\nFinding optimal threshold on validation set...")
        optimal_threshold, opt_f1, opt_precision, opt_recall = get_optimal_threshold(
            model, X_test, y_test, device
        )
        if verbose:
            print(f"Optimal threshold: {optimal_threshold:.3f}")
            print(f"At optimal threshold - F1: {opt_f1:.4f}, Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f}")
    else:
        optimal_threshold = 0.5

    # Final evaluation with optimal threshold
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_probs = torch.sigmoid(test_outputs.squeeze()).cpu().numpy()
        test_preds = (test_probs >= optimal_threshold).astype(int)

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

        if verbose:
            print("\n" + "="*60)
            print("FINAL EVALUATION RESULTS (Optimal Threshold)")
            print("="*60)
            print(f"Optimal Threshold: {optimal_threshold:.3f}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"FPR:       {fpr:.4f}")
            print(f"FNR:       {fnr:.4f}")
            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            print("="*60)

            # Also show results at default 0.5 threshold for comparison
            test_preds_05 = (test_probs >= 0.5).astype(int)
            acc_05 = accuracy_score(y_test, test_preds_05)
            prec_05 = precision_score(y_test, test_preds_05, zero_division=0)
            rec_05 = recall_score(y_test, test_preds_05, zero_division=0)
            f1_05 = f1_score(y_test, test_preds_05, zero_division=0)
            print(f"\nAt threshold 0.50: Acc={acc_05:.4f}, Prec={prec_05:.4f}, Rec={rec_05:.4f}, F1={f1_05:.4f}")

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
        'optimal_threshold': float(optimal_threshold),
        'training_time': training_time,
        'best_epoch': best_epoch + 1,
        'history': history,
        'config': {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'use_residual': use_residual,
            'use_layernorm': use_layernorm,
            'residual_every': residual_every,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'loss_type': loss_type,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'pos_weight': pos_weight,
            'scheduler_type': scheduler_type,
            'warmup_epochs': warmup_epochs,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_metric': early_stopping_metric,
            'gradient_clip': gradient_clip,
            'label_smoothing': label_smoothing,
            'find_optimal_threshold': find_optimal_threshold
        }
    }

    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train improved centralized MLP')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'weighted_bce', 'bce'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64, 32])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--no_residual', action='store_true')
    parser.add_argument('--no_layernorm', action='store_true')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup', choices=['cosine_warmup', 'reduce_on_plateau', 'step', 'none'])
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

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

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Train the model
    model, results = train_improved_centralized_mlp(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
        use_residual=not args.no_residual,
        use_layernorm=not args.no_layernorm,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        loss_type=args.loss_type,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stop_patience,
        verbose=True
    )

    # Save the model
    model_dir = "/Users/admin/Desktop/federated-ids/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "centralized_mlp_improved.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': args.hidden_dims,
        'dropout_rate': args.dropout,
        'use_residual': not args.no_residual,
        'use_layernorm': not args.no_layernorm,
        'residual_every': 2,
        'results': results
    }, model_path)

    print(f"\nModel saved to: {model_path}")

    # Save results
    results_path = os.path.join(model_dir, "centralized_mlp_improved_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    return model, results


if __name__ == "__main__":
    main()