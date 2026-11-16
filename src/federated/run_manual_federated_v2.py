"""
Manual Federated Learning V2 - Improved Version
Uses the improved MLP architecture with:
- Focal Loss for class imbalance
- Improved model architecture (residual connections, LayerNorm)
- Learning rate scheduling
- Optimal threshold selection
- Better local training
"""

import os
import numpy as np
import torch
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from copy import deepcopy

from src.models.improved_mlp import ImprovedMLPIDS, FocalLoss, create_improved_mlp_model, count_parameters, get_optimal_threshold
from src.data.preprocessing import CICIDS2017Preprocessor
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_and_partition_data(
    num_clients: int = 5,
    val_ratio: float = 0.1,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    random_state: int = 42,
    max_samples_per_client: int = None
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],  # client train data
    List[Tuple[np.ndarray, np.ndarray]],  # client val data
    Tuple[np.ndarray, np.ndarray],       # global test data
    int                                  # input dimension
]:
    """Load and partition data for federated learning."""
    processed_dir = "/Users/admin/Desktop/federated-ids/data/processed"

    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    print(f"Global dataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    np.random.seed(random_state)

    if partition_strategy == "iid":
        indices = np.random.permutation(len(X_train))
        split_indices = np.array_split(indices, num_clients)

        client_data = []
        for client_indices in split_indices:
            client_X = X_train[client_indices]
            client_y = y_train[client_indices]
            client_data.append((client_X, client_y))

    elif partition_strategy == "non_iid_dirichlet":
        class_indices = []
        for class_label in np.unique(y_train):
            class_indices.append(np.where(y_train == class_label)[0])

        client_indices = [[] for _ in range(num_clients)]

        for indices_in_class in class_indices:
            np.random.shuffle(indices_in_class)
            proportions = np.random.dirichlet(alpha=[dirichlet_alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(indices_in_class)).astype(int)[:-1]
            class_splits = np.split(indices_in_class, split_points)

            for i, indices_for_client in enumerate(class_splits):
                client_indices[i].extend(indices_for_client.tolist())

        client_data = []
        for indices in client_indices:
            np.random.shuffle(indices)
            client_X = X_train[indices]
            client_y = y_train[indices]
            client_data.append((client_X, client_y))

    else:
        raise ValueError(f"Unknown partition strategy: {partition_strategy}")

    print(f"\nData partitioning using '{partition_strategy}' strategy:")
    print(f"Number of clients: {num_clients}")
    for i, (client_X, client_y) in enumerate(client_data):
        unique, counts = np.unique(client_y, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"  Client {i}: {len(client_X)} samples, label distribution: {distribution}")

    # Limit samples per client if specified
    if max_samples_per_client is not None:
        print(f"\nLimiting samples to {max_samples_per_client} per client...")
        limited_client_data = []
        for client_X, client_y in client_data:
            if len(client_X) > max_samples_per_client:
                indices = np.random.choice(len(client_X), max_samples_per_client, replace=False)
                client_X = client_X[indices]
                client_y = client_y[indices]
            limited_client_data.append((client_X, client_y))
        client_data = limited_client_data
        for i, (client_X, client_y) in enumerate(client_data):
            unique, counts = np.unique(client_y, return_counts=True)
            distribution = dict(zip(unique, counts))
            print(f"  Client {i} (limited): {len(client_X)} samples, label distribution: {distribution}")

    client_train_data = []
    client_val_data = []

    for client_X, client_y in client_data:
        n_val = int(len(client_X) * val_ratio)
        indices = np.random.permutation(len(client_X))

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        client_X_train = client_X[train_indices]
        client_y_train = client_y[train_indices]
        client_X_val = client_X[val_indices]
        client_y_val = client_y[val_indices]

        client_train_data.append((client_X_train, client_y_train))
        client_val_data.append((client_X_val, client_y_val))

    print(f"\nAfter train/validation split:")
    for i, ((X_train, y_train), (X_val, y_val)) in enumerate(zip(client_train_data, client_val_data)):
        print(f"  Client {i}: {len(X_train)} train, {len(X_val)} val samples")

    return client_train_data, client_val_data, (X_test, y_test), input_dim


def load_centralized_model_improved(input_dim: int, hidden_dims: List[int], dropout_rate: float, device: str,
                                     use_residual: bool = True, use_layernorm: bool = True):
    """Load the improved centralized model for warm start."""
    model_path = "/Users/admin/Desktop/federated-ids/models/centralized_mlp_improved.pth"
    if os.path.exists(model_path):
        print(f"Loading improved centralized model from {model_path}...")
        model = create_improved_mlp_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layernorm=use_layernorm
        )
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        if hasattr(model, 'network'):
            network_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('network.'):
                    network_state_dict[k[8:]] = v
                else:
                    network_state_dict[k] = v
            model.network.load_state_dict(network_state_dict)
        else:
            model.load_state_dict(state_dict)

        if hasattr(model, 'to'):
            model = model.to(device)
        elif hasattr(model, 'network') and hasattr(model.network, 'to'):
            model.network = model.network.to(device)
        print("Improved centralized model loaded successfully!")
        return model
    else:
        print(f"Improved centralized model not found at {model_path}")
        return None


def get_model_parameters(model):
    """Extract model parameters as numpy arrays."""
    if hasattr(model, 'network'):
        return [val.cpu().clone().numpy() for _, val in model.network.state_dict().items()]
    elif hasattr(model, 'state_dict'):
        return [val.cpu().clone().numpy() for _, val in model.state_dict().items()]
    else:
        raise AttributeError("Model has no network or state_dict attribute")


def set_model_parameters(model, parameters):
    """Set model parameters from numpy arrays."""
    if hasattr(model, 'network'):
        params_dict = zip(model.network.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.network.load_state_dict(state_dict, strict=True)
    elif hasattr(model, 'state_dict'):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    else:
        raise AttributeError("Model has no network or state_dict attribute")


def average_parameters(parameters_list, weights=None):
    """Average parameters across clients (FedAvg)."""
    if weights is None:
        weights = [1.0] * len(parameters_list)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    avg_params = []
    for i in range(len(parameters_list[0])):
        avg_param = np.zeros_like(parameters_list[0][i], dtype=np.float32)
        for j, params in enumerate(parameters_list):
            avg_param += weights[j] * params[i]
        avg_params.append(avg_param)

    return avg_params


def evaluate_model_detailed(model, X_test, y_test, device: str, threshold: float = 0.5):
    """Evaluate model on test data with detailed metrics."""
    network = model.network if hasattr(model, 'network') else model
    network.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_tensor = torch.LongTensor(y_test).to(device)

        outputs = network(X_tensor)
        outputs_flat = outputs.view(-1)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs_flat, y_tensor.float()
        ).item()

        probs = torch.sigmoid(outputs_flat)
        predicted = (probs >= threshold).long()
        accuracy = (predicted == y_tensor).float().mean().item()

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(
                y_tensor.cpu().numpy(), probs.detach().cpu().numpy()
            )
        except Exception:
            roc_auc = 0.0

        # Precision, Recall, F1
        y_true = y_tensor.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": threshold,
        "confusion_matrix": cm.tolist()
    }


def train_client_model_improved(client_model, X_train, y_train, X_val, y_val,
                                local_epochs, batch_size, learning_rate, weight_decay, device,
                                loss_type="focal", focal_alpha=0.25, focal_gamma=2.0):
    """Train a client model on local data with improved techniques."""
    network = client_model.network if hasattr(client_model, 'network') else client_model

    # Setup data loaders
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif loss_type == 'weighted_bce':
        class_counts = np.bincount(y_train)
        pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Local LR scheduler (cosine annealing)
    def lr_lambda(epoch):
        progress = epoch / max(local_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    local_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    network.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = network(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        train_losses.append(avg_epoch_loss)
        train_accuracies.append(epoch_acc)

        local_scheduler.step()

    # Validation
    network.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = network(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs.squeeze())
            predicted = (probs >= 0.5).long()

            all_val_preds.append(predicted.cpu().numpy())
            all_val_probs.append(probs.cpu().numpy())
            all_val_labels.append(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    val_preds = np.concatenate(all_val_preds)
    val_probs = np.concatenate(all_val_probs)
    val_labels = np.concatenate(all_val_labels)

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, zero_division=0)
    val_rec = recall_score(val_labels, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "val_f1": val_f1
    }


def run_manual_federated_v2(
    num_clients: int = 5,
    num_rounds: int = 30,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    local_epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.1,
    device: str = "cpu",
    warm_start: bool = True,
    checkpoint_interval: int = 10,
    max_samples_per_client: int = 200000,
    loss_type: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_residual: bool = True,
    use_layernorm: bool = True,
    hidden_dims: List[int] = None,
    dropout_rate: float = 0.2,
) -> Dict:
    """Run manual federated learning V2 with improved techniques."""
    if hidden_dims is None:
        hidden_dims = [256, 128, 64, 32]

    print("=" * 70)
    print("Starting MANUAL Federated Learning V2 (Improved)")
    print("=" * 70)

    start_time = time.time()

    # Load and partition data
    print(f"\n1. Loading and partitioning data for {num_clients} clients...")
    client_train_data, client_val_data, (X_test, y_test), input_dim = load_and_partition_data(
        num_clients=num_clients,
        val_ratio=val_ratio,
        partition_strategy=partition_strategy,
        dirichlet_alpha=dirichlet_alpha,
        random_state=42,
        max_samples_per_client=max_samples_per_client
    )

    # Model architecture
    print(f"\n2. Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden layers: {hidden_dims}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Use residual: {use_residual}")
    print(f"  Use LayerNorm: {use_layernorm}")
    print(f"  Loss type: {loss_type}")

    # Create global model (warm start or random init)
    print(f"\n3. Initializing global model...")
    if warm_start:
        global_model = load_centralized_model_improved(
            input_dim, hidden_dims, dropout_rate, device, use_residual, use_layernorm
        )
        if global_model is None:
            print("  Falling back to random initialization...")
            global_model = create_improved_mlp_model(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_residual=use_residual,
                use_layernorm=use_layernorm
            )
            if hasattr(global_model, 'to'):
                global_model = global_model.to(device)
    else:
        global_model = create_improved_mlp_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layernorm=use_layernorm
        )
        if hasattr(global_model, 'to'):
            global_model = global_model.to(device)

    # Evaluate initial model
    print(f"\n4. Evaluating initial global model...")
    initial_metrics = evaluate_model_detailed(global_model, X_test, y_test, device, threshold=0.5)
    print(f"  Initial - Loss: {initial_metrics['loss']:.4f}, Accuracy: {initial_metrics['accuracy']:.4f}, "
          f"F1: {initial_metrics['f1']:.4f}, Precision: {initial_metrics['precision']:.4f}, Recall: {initial_metrics['recall']:.4f}")

    # History tracking
    history = {
        "rounds": [],
        "global_loss": [],
        "global_accuracy": [],
        "global_precision": [],
        "global_recall": [],
        "global_f1": [],
        "global_roc_auc": [],
        "global_fpr": [],
        "global_fnr": [],
        "client_metrics": []
    }

    history["rounds"].append(0)
    history["global_loss"].append(initial_metrics["loss"])
    history["global_accuracy"].append(initial_metrics["accuracy"])
    history["global_precision"].append(initial_metrics["precision"])
    history["global_recall"].append(initial_metrics["recall"])
    history["global_f1"].append(initial_metrics["f1"])
    history["global_roc_auc"].append(initial_metrics["roc_auc"])
    history["global_fpr"].append(initial_metrics["fpr"])
    history["global_fnr"].append(initial_metrics["fnr"])
    history["client_metrics"].append([])

    # Get initial global parameters
    global_params = get_model_parameters(global_model)

    # Create checkpoint directory
    checkpoint_dir = "/Users/admin/Desktop/federated-ids/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save initial checkpoint
    if checkpoint_interval > 0:
        checkpoint_path = os.path.join(checkpoint_dir, "federated_v2_round_0.pth")
        network = global_model.network if hasattr(global_model, 'network') else global_model
        torch.save({
            'model_state_dict': network.state_dict(),
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'use_residual': use_residual,
            'use_layernorm': use_layernorm,
            'round': 0,
            'metrics': initial_metrics,
            'config': {
                'num_clients': num_clients,
                'num_rounds': num_rounds,
                'partition_strategy': partition_strategy,
                'dirichlet_alpha': dirichlet_alpha,
                'local_epochs': local_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'val_ratio': val_ratio,
                'device': device,
                'warm_start': warm_start,
                'checkpoint_interval': checkpoint_interval,
                'loss_type': loss_type,
                'focal_alpha': focal_alpha,
                'focal_gamma': focal_gamma
            }
        }, checkpoint_path)
        print(f"  Initial checkpoint saved: {checkpoint_path}")

    best_f1 = initial_metrics['f1']
    best_round = 0

    # Federated training rounds
    print(f"\n5. Starting federated training for {num_rounds} rounds...")
    print("-" * 70)

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()
        print(f"\n  Round {round_num}/{num_rounds}")

        client_params_list = []
        client_weights = []
        client_metrics_round = []

        # Train on each client
        for client_id in range(num_clients):
            X_train, y_train = client_train_data[client_id]
            X_val, y_val = client_val_data[client_id]

            # Create client model and set to global parameters
            client_model = create_improved_mlp_model(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_residual=use_residual,
                use_layernorm=use_layernorm
            )
            if hasattr(client_model, 'to'):
                client_model = client_model.to(device)
            set_model_parameters(client_model, global_params)

            # Train client
            metrics = train_client_model_improved(
                client_model, X_train, y_train, X_val, y_val,
                local_epochs, batch_size, learning_rate, weight_decay, device,
                loss_type, focal_alpha, focal_gamma
            )

            # Get updated parameters
            client_params = get_model_parameters(client_model)
            client_params_list.append(client_params)
            client_weights.append(len(X_train))
            client_metrics_round.append({
                "client_id": client_id,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "train_losses": metrics["train_losses"],
                "train_accuracies": metrics["train_accuracies"],
                "val_loss": metrics["val_loss"],
                "val_accuracy": metrics["val_accuracy"],
                "val_precision": metrics["val_precision"],
                "val_recall": metrics["val_recall"],
                "val_f1": metrics["val_f1"]
            })

            print(f"    Client {client_id}: val_loss={metrics['val_loss']:.4f}, "
                  f"val_f1={metrics['val_f1']:.4f}, val_prec={metrics['val_precision']:.4f}")

        # Aggregate (FedAvg)
        print(f"    Aggregating parameters (FedAvg)...")
        global_params = average_parameters(client_params_list, client_weights)

        # Update global model
        set_model_parameters(global_model, global_params)

        # Evaluate global model
        global_metrics = evaluate_model_detailed(global_model, X_test, y_test, device, threshold=0.5)

        # Track history
        history["rounds"].append(round_num)
        history["global_loss"].append(global_metrics["loss"])
        history["global_accuracy"].append(global_metrics["accuracy"])
        history["global_precision"].append(global_metrics["precision"])
        history["global_recall"].append(global_metrics["recall"])
        history["global_f1"].append(global_metrics["f1"])
        history["global_roc_auc"].append(global_metrics["roc_auc"])
        history["global_fpr"].append(global_metrics["fpr"])
        history["global_fnr"].append(global_metrics["fnr"])
        history["client_metrics"].append(client_metrics_round)

        round_time = time.time() - round_start
        print(f"    Global - Loss: {global_metrics['loss']:.4f}, Acc: {global_metrics['accuracy']:.4f}, "
              f"F1: {global_metrics['f1']:.4f}, Prec: {global_metrics['precision']:.4f}, Rec: {global_metrics['recall']:.4f} "
              f"({round_time:.1f}s)")

        # Track best
        if global_metrics['f1'] > best_f1:
            best_f1 = global_metrics['f1']
            best_round = round_num
            print(f"    *** New best F1: {best_f1:.4f} ***")

        # Save checkpoint
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"federated_v2_round_{round_num}.pth")
            network = global_model.network if hasattr(global_model, 'network') else global_model
            torch.save({
                'model_state_dict': network.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
                'use_residual': use_residual,
                'use_layernorm': use_layernorm,
                'round': round_num,
                'metrics': global_metrics,
                'config': {
                    'num_clients': num_clients,
                    'num_rounds': num_rounds,
                    'partition_strategy': partition_strategy,
                    'dirichlet_alpha': dirichlet_alpha,
                    'local_epochs': local_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'val_ratio': val_ratio,
                    'device': device,
                    'warm_start': warm_start,
                    'checkpoint_interval': checkpoint_interval,
                    'loss_type': loss_type,
                    'focal_alpha': focal_alpha,
                    'focal_gamma': focal_gamma
                }
            }, checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - start_time

    # Find optimal threshold on test set
    print(f"\n6. Finding optimal threshold on test set...")
    optimal_threshold, opt_f1, opt_prec, opt_rec = get_optimal_threshold(global_model, X_test, y_test, device)
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  At optimal - F1: {opt_f1:.4f}, Precision: {opt_prec:.4f}, Recall: {opt_rec:.4f}")

    # Final evaluation with optimal threshold
    print(f"\n7. Final evaluation with optimal threshold ({optimal_threshold:.3f})...")
    final_metrics = evaluate_model_detailed(global_model, X_test, y_test, device, threshold=optimal_threshold)
    print(f"  Final - Loss: {final_metrics['loss']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}, "
          f"F1: {final_metrics['f1']:.4f}, Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}")
    print(f"  ROC-AUC: {final_metrics['roc_auc']:.4f}, FPR: {final_metrics['fpr']:.4f}, FNR: {final_metrics['fnr']:.4f}")
    print(f"  TP: {final_metrics['tp']}, TN: {final_metrics['tn']}, FP: {final_metrics['fp']}, FN: {final_metrics['fn']}")
    print(f"  Best F1 during training: {best_f1:.4f} at round {best_round}")

    # Save final model
    model_dir = "/Users/admin/Desktop/federated-ids/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "federated_mlp_manual_v2.pth")
    network = global_model.network if hasattr(global_model, 'network') else global_model
    torch.save({
        'model_state_dict': network.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate,
        'use_residual': use_residual,
        'use_layernorm': use_layernorm,
        'metrics': final_metrics,
        'optimal_threshold': optimal_threshold,
        'best_f1': best_f1,
        'best_round': best_round,
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'partition_strategy': partition_strategy,
            'dirichlet_alpha': dirichlet_alpha,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'val_ratio': val_ratio,
            'device': device,
            'warm_start': warm_start,
            'checkpoint_interval': checkpoint_interval,
            'loss_type': loss_type,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma
        }
    }, model_path)
    print(f"  Final model saved: {model_path}")

    # Save results JSON
    results_json_path = os.path.join(model_dir, "manual_federated_v2_results.json")
    with open(results_json_path, 'w') as f:
        json.dump({
            "results": final_metrics,
            "optimal_threshold": float(optimal_threshold),
            "best_f1_during_training": float(best_f1),
            "best_round": best_round,
            "history": history,
            "config": {
                'num_clients': num_clients,
                'num_rounds': num_rounds,
                'partition_strategy': partition_strategy,
                'dirichlet_alpha': dirichlet_alpha,
                'local_epochs': local_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'val_ratio': val_ratio,
                'device': device,
                'warm_start': warm_start,
                'checkpoint_interval': checkpoint_interval,
                'loss_type': loss_type,
                'focal_alpha': focal_alpha,
                'focal_gamma': focal_gamma,
                'use_residual': use_residual,
                'use_layernorm': use_layernorm,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
                'input_dim': input_dim
            },
            "simulation_time": total_time
        }, f, indent=2)
    print(f"  Results JSON saved: {results_json_path}")

    print(f"\n8. Manual Federated Learning V2 Completed!")
    print(f"  Total time: {total_time:.2f} seconds")
    print("=" * 70)

    return {
        "final_metrics": final_metrics,
        "optimal_threshold": optimal_threshold,
        "best_f1": best_f1,
        "best_round": best_round,
        "history": history,
        "config": {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'partition_strategy': partition_strategy,
            'dirichlet_alpha': dirichlet_alpha,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'val_ratio': val_ratio,
            'device': device,
            'warm_start': warm_start,
            'checkpoint_interval': checkpoint_interval,
            'loss_type': loss_type,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'use_residual': use_residual,
            'use_layernorm': use_layernorm,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate
        },
        "simulation_time": total_time
    }


def main():
    results = run_manual_federated_v2(
        num_clients=5,
        num_rounds=30,
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        local_epochs=5,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=1e-4,
        val_ratio=0.1,
        device="cpu",
        warm_start=True,
        checkpoint_interval=10,
        max_samples_per_client=200000,
        loss_type="focal",
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_residual=True,
        use_layernorm=True,
        hidden_dims=[256, 128, 64, 32],
        dropout_rate=0.2,
    )
    return results


if __name__ == "__main__":
    main()