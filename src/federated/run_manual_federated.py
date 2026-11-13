"""
Manual Federated Learning for Intrusion Detection System.
Runs federated averaging manually without requiring Ray/Flower simulation.
"""

import os
import numpy as np
import torch
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from copy import deepcopy

from src.models.mlp import MLPIDS, count_parameters
from src.data.preprocessing import CICIDS2017Preprocessor
from sklearn.metrics import roc_auc_score


def load_and_partition_data(
    num_clients: int = 5,
    val_ratio: float = 0.2,
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
                # Randomly subsample
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


def load_centralized_model(input_dim: int, hidden_dims: List[int], dropout_rate: float, device: str):
    """Load the centralized model for warm start."""
    model_path = "/Users/admin/Desktop/federated-ids/models/centralized_mlp.pth"
    if os.path.exists(model_path):
        print(f"Loading centralized model from {model_path}...")
        model = MLPIDS(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
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
        print("Centralized model loaded successfully!")
        return model
    else:
        print(f"Centralized model not found at {model_path}")
        return None


def get_model_parameters(model):
    """Extract model parameters as numpy arrays.

    Uses ``.clone()`` so the returned arrays are independent copies of the
    parameters. Without it, ``tensor.numpy()`` shares memory with the live
    parameter, and an in-place ``optimizer.step()`` would mutate the
    "before" snapshot too (making before==after).
    """
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


def evaluate_model(model, X_test, y_test, device: str):
    """Evaluate model on test data."""
    # Get the actual network module
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
        predicted = (probs >= 0.5).long()
        accuracy = (predicted == y_tensor).float().mean().item()

        # ROC-AUC (requires both classes present)
        try:
            roc_auc = roc_auc_score(
                y_tensor.cpu().numpy(), probs.detach().cpu().numpy()
            )
        except Exception:
            roc_auc = 0.0

        # Also compute precision, recall, F1
        tp = ((predicted == 1) & (y_tensor == 1)).sum().item()
        tn = ((predicted == 0) & (y_tensor == 0)).sum().item()
        fp = ((predicted == 1) & (y_tensor == 0)).sum().item()
        fn = ((predicted == 0) & (y_tensor == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def train_client_model(client_model, X_train, y_train, X_val, y_val,
                       local_epochs, batch_size, learning_rate, device):
    """Train a client model on local data."""
    # Get the actual network module
    network = client_model.network if hasattr(client_model, 'network') else client_model

    # Setup data loaders
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

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
            outputs_flat = outputs.view(-1)
            loss = criterion(outputs_flat, batch_y.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predicted = (torch.sigmoid(outputs_flat) >= 0.5).long()
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        train_losses.append(avg_epoch_loss)
        train_accuracies.append(epoch_acc)

    # Validation
    network.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = network(batch_X)
            outputs_flat = outputs.view(-1)
            loss = criterion(outputs_flat, batch_y.float())
            val_loss += loss.item()
            predicted = (torch.sigmoid(outputs_flat) >= 0.5).long()
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()

    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc
    }


def run_manual_federated(
    num_clients: int = 5,
    num_rounds: int = 50,            # Full federated training (was reduced to 20 for speed)
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    local_epochs: int = 5,           # Proper local training per client (was reduced to 3)
    batch_size: int = 128,           # Optimal: power-of-2, balances gradient stability & speed
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    device: str = "cpu",
    warm_start: bool = True,
    checkpoint_interval: int = 10,   # Checkpoint every 10 rounds
    max_samples_per_client: int = 100000  # Use more data per client for proper training
) -> Dict:
    """Run manual federated learning with warm start and convergence monitoring."""
    print("=" * 70)
    print("Starting MANUAL Federated Learning (No Ray/Flower Required)")
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
    hidden_dims = [128, 64, 32]
    dropout_rate = 0.3

    print(f"\n2. Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden layers: {hidden_dims}")
    print(f"  Dropout rate: {dropout_rate}")

    # Create global model (warm start or random init)
    print(f"\n3. Initializing global model...")
    if warm_start:
        global_model = load_centralized_model(input_dim, hidden_dims, dropout_rate, device)
        if global_model is None:
            print("  Falling back to random initialization...")
            global_model = MLPIDS(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
            if hasattr(global_model, 'to'):
                global_model = global_model.to(device)
    else:
        global_model = MLPIDS(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
        if hasattr(global_model, 'to'):
            global_model = global_model.to(device)

    # Evaluate initial model
    print(f"\n4. Evaluating initial global model...")
    initial_metrics = evaluate_model(global_model, X_test, y_test, device)
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
        "client_metrics": []
    }

    history["rounds"].append(0)
    history["global_loss"].append(initial_metrics["loss"])
    history["global_accuracy"].append(initial_metrics["accuracy"])
    history["global_precision"].append(initial_metrics["precision"])
    history["global_recall"].append(initial_metrics["recall"])
    history["global_f1"].append(initial_metrics["f1"])
    history["client_metrics"].append([])

    # Get initial global parameters
    global_params = get_model_parameters(global_model)

    # Create checkpoint directory
    checkpoint_dir = "/Users/admin/Desktop/federated-ids/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save initial checkpoint
    if checkpoint_interval > 0:
        checkpoint_path = os.path.join(checkpoint_dir, "federated_round_0.pth")
        network = global_model.network if hasattr(global_model, 'network') else global_model
        torch.save({
            'model_state_dict': network.state_dict(),
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
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
                'val_ratio': val_ratio,
                'device': device,
                'warm_start': warm_start,
                'checkpoint_interval': checkpoint_interval
            }
        }, checkpoint_path)
        print(f"  Initial checkpoint saved: {checkpoint_path}")

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
            client_model = MLPIDS(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
            if hasattr(client_model, 'to'):
                client_model = client_model.to(device)
            set_model_parameters(client_model, global_params)

            # Train client
            metrics = train_client_model(
                client_model, X_train, y_train, X_val, y_val,
                local_epochs, batch_size, learning_rate, device
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
                "val_accuracy": metrics["val_accuracy"]
            })

            print(f"    Client {client_id}: train_loss={metrics['train_losses'][-1]:.4f}, "
                  f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_accuracy']:.4f}")

        # Aggregate (FedAvg)
        print(f"    Aggregating parameters (FedAvg)...")
        global_params = average_parameters(client_params_list, client_weights)

        # Update global model
        set_model_parameters(global_model, global_params)

        # Evaluate global model
        global_metrics = evaluate_model(global_model, X_test, y_test, device)

        # Track history
        history["rounds"].append(round_num)
        history["global_loss"].append(global_metrics["loss"])
        history["global_accuracy"].append(global_metrics["accuracy"])
        history["global_precision"].append(global_metrics["precision"])
        history["global_recall"].append(global_metrics["recall"])
        history["global_f1"].append(global_metrics["f1"])
        history["client_metrics"].append(client_metrics_round)

        round_time = time.time() - round_start
        print(f"    Global - Loss: {global_metrics['loss']:.4f}, Accuracy: {global_metrics['accuracy']:.4f}, "
              f"F1: {global_metrics['f1']:.4f}, Precision: {global_metrics['precision']:.4f}, Recall: {global_metrics['recall']:.4f} "
              f"({round_time:.1f}s)")

        # Save checkpoint
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"federated_round_{round_num}.pth")
            network = global_model.network if hasattr(global_model, 'network') else global_model
            torch.save({
                'model_state_dict': network.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': hidden_dims,
                'dropout_rate': dropout_rate,
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
                    'val_ratio': val_ratio,
                    'device': device,
                    'warm_start': warm_start,
                    'checkpoint_interval': checkpoint_interval
                }
            }, checkpoint_path)
            print(f"    Checkpoint saved: {checkpoint_path}")

        # Early stopping check (if loss is increasing consistently)
        if round_num >= 10:
            recent_losses = history["global_loss"][-5:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                print(f"    Warning: Loss increasing for 5 consecutive rounds. Consider early stopping.")

    total_time = time.time() - start_time

    # Final evaluation
    print(f"\n6. Final evaluation...")
    final_metrics = evaluate_model(global_model, X_test, y_test, device)
    print(f"  Final - Loss: {final_metrics['loss']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}, "
          f"F1: {final_metrics['f1']:.4f}, Precision: {final_metrics['precision']:.4f}, Recall: {final_metrics['recall']:.4f}")
    print(f"  Best accuracy: {max(history['global_accuracy']):.4f} at round {history['global_accuracy'].index(max(history['global_accuracy']))}")

    # Save final model
    model_dir = "/Users/admin/Desktop/federated-ids/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "federated_mlp_manual.pth")
    network = global_model.network if hasattr(global_model, 'network') else global_model
    torch.save({
        'model_state_dict': network.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate,
        'metrics': final_metrics,
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'partition_strategy': partition_strategy,
            'dirichlet_alpha': dirichlet_alpha,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_ratio': val_ratio,
            'device': device,
            'warm_start': warm_start,
            'checkpoint_interval': checkpoint_interval
        }
    }, model_path)
    print(f"  Final model saved: {model_path}")

    # Save results JSON
    results_json_path = os.path.join(model_dir, "manual_federated_improved_results.json")
    with open(results_json_path, 'w') as f:
        json.dump({
            "results": final_metrics,
            "history": history,
            "config": {
                'num_clients': num_clients,
                'num_rounds': num_rounds,
                'partition_strategy': partition_strategy,
                'dirichlet_alpha': dirichlet_alpha,
                'local_epochs': local_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'val_ratio': val_ratio,
                'device': device,
                'warm_start': warm_start,
                'checkpoint_interval': checkpoint_interval
            },
            "simulation_time": total_time
        }, f, indent=2)
    print(f"  Results JSON saved: {results_json_path}")

    print(f"\n7. Manual Federated Learning Completed!")
    print(f"  Total time: {total_time:.2f} seconds")
    print("=" * 70)

    return {
        "final_metrics": final_metrics,
        "history": history,
        "config": {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'partition_strategy': partition_strategy,
            'dirichlet_alpha': dirichlet_alpha,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'val_ratio': val_ratio,
            'device': device,
            'warm_start': warm_start,
            'checkpoint_interval': checkpoint_interval
        },
        "simulation_time": total_time
    }


def main():
    """Main function to run manual federated learning."""
    results = run_manual_federated(
        num_clients=5,
        num_rounds=50,
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        local_epochs=5,
        batch_size=128,
        learning_rate=0.001,
        val_ratio=0.2,
        device="cpu",
        warm_start=True,
        checkpoint_interval=10,
        max_samples_per_client=100000
    )
    return results


if __name__ == "__main__":
    main()