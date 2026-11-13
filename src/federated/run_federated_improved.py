"""
Improved Federated Learning Simulation for Intrusion Detection System.
Runs with more rounds, warm start from centralized model, and convergence monitoring.
"""

import os
import numpy as np
import flwr as fl
import torch
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from src.data.preprocessing import CICIDS2017Preprocessor
from src.federated.client import FlowerClient
from src.federated.server import get_evaluate_fn, get_on_fit_config_fn, get_on_evaluate_config_fn
from src.models.mlp import MLPIDS, count_parameters


def load_and_partition_data(
    num_clients: int = 5,
    val_ratio: float = 0.2,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    random_state: int = 42
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray]],  # client train data
    List[Tuple[np.ndarray, np.ndarray]],  # client val data
    Tuple[np.ndarray, np.ndarray],       # global test data
    int                                  # input dimension
]:
    """
    Load and partition data for federated learning.
    """
    # Load preprocessed data
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

    # Set random seed
    np.random.seed(random_state)

    # Partition training data among clients
    if partition_strategy == "iid":
        # IID partitioning: randomly shuffle and split evenly
        indices = np.random.permutation(len(X_train))
        split_indices = np.array_split(indices, num_clients)

        client_data = []
        for client_indices in split_indices:
            client_X = X_train[client_indices]
            client_y = y_train[client_indices]
            client_data.append((client_X, client_y))

    elif partition_strategy == "non_iid_dirichlet":
        # Non-IID partitioning using Dirichlet distribution
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

    # Print partition statistics
    print(f"\nData partitioning using '{partition_strategy}' strategy:")
    print(f"Number of clients: {num_clients}")
    for i, (client_X, client_y) in enumerate(client_data):
        unique, counts = np.unique(client_y, return_counts=True)
        distribution = dict(zip(unique, counts))
        print(f"  Client {i}: {len(client_X)} samples, label distribution: {distribution}")

    # Split each client's data into train and validation
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

        # Handle both cases: state_dict with or without "network." prefix
        # The centralized model was saved as the full MLPIDS model
        if hasattr(model, 'network'):
            # Try loading into network directly (stripping "network." prefix if present)
            network_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('network.'):
                    network_state_dict[k[8:]] = v  # Remove "network." prefix
                else:
                    network_state_dict[k] = v
            model.network.load_state_dict(network_state_dict)
        else:
            # Fallback for numpy version
            model.load_state_dict(state_dict)

        # Move to device if torch is available
        if hasattr(model, 'to'):
            model = model.to(device)
        elif hasattr(model, 'network') and hasattr(model.network, 'to'):
            model.network = model.network.to(device)
        print("Centralized model loaded successfully!")
        return model
    else:
        print(f"Centralized model not found at {model_path}")
        return None


def run_federated_simulation(
    num_clients: int = 5,
    num_rounds: int = 50,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    local_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    device: str = "cpu",
    save_model: bool = True,
    warm_start: bool = True,
    checkpoint_interval: int = 10
) -> Dict:
    """
    Run improved federated learning simulation with warm start and convergence monitoring.
    """
    print("=" * 70)
    print("Starting IMPROVED Federated Learning Simulation")
    print("=" * 70)

    start_time = time.time()

    # Load and partition data
    print(f"\n1. Loading and partitioning data for {num_clients} clients...")
    client_train_data, client_val_data, (X_test, y_test), input_dim = load_and_partition_data(
        num_clients=num_clients,
        val_ratio=val_ratio,
        partition_strategy=partition_strategy,
        dirichlet_alpha=dirichlet_alpha,
        random_state=42
    )

    # Define model architecture
    hidden_dims = [128, 64, 32]
    dropout_rate = 0.3

    print(f"\n2. Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden layers: {hidden_dims}")
    print(f"  Dropout rate: {dropout_rate}")

    # Create initial model (for parameter counting)
    temp_model = MLPIDS(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    print(f"  Model parameters: {count_parameters(temp_model):,}")

    # Load centralized model for warm start
    initial_parameters = None
    if warm_start:
        print(f"\n3. Loading centralized model for warm start...")
        central_model = load_centralized_model(input_dim, hidden_dims, dropout_rate, device)
        if central_model is not None:
            # Extract parameters from the network
            if hasattr(central_model, 'network'):
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in central_model.network.state_dict().items()]
                )
            elif hasattr(central_model, 'state_dict'):
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in central_model.state_dict().items()]
                )
            else:
                # Fallback
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
                )
            print("  Warm start enabled with centralized model weights!")
        else:
            print("  Using random initialization (centralized model not found)")
            if hasattr(temp_model, 'network'):
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in temp_model.network.state_dict().items()]
                )
            elif hasattr(temp_model, 'state_dict'):
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
                )
            else:
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
                )
    else:
        print(f"\n3. Using random initialization (warm_start=False)")
        if hasattr(temp_model, 'network'):
            initial_parameters = fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in temp_model.network.state_dict().items()]
            )
        elif hasattr(temp_model, 'state_dict'):
            initial_parameters = fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
            )
        else:
            initial_parameters = fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
            )

    # Prepare evaluation function for server-side evaluation
    print(f"\n4. Setting up evaluation function...")
    evaluate_fn = get_evaluate_fn(
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        device=device
    )

    # Prepare configuration functions
    on_fit_config_fn = get_on_fit_config_fn()
    on_evaluate_config_fn = get_on_evaluate_config_fn()

    # Track history for convergence monitoring
    history_tracker = {
        "losses_distributed": [],
        "losses_centralized": [],
        "metrics_distributed": {"accuracy": [], "val_accuracy": []},
        "metrics_centralized": {"accuracy": [], "loss": []},
    }

    # Define client function that captures the data
    def client_fn(cid: str) -> FlowerClient:
        client_id = int(cid)
        X_train, y_train = client_train_data[client_id]
        X_val, y_val = client_val_data[client_id]

        return FlowerClient(
            client_id=cid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            local_epochs=local_epochs,
            device=device
        )

    # Custom strategy with checkpointing
    class CheckpointingFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, *args, checkpoint_dir: str = None, checkpoint_interval: int = 10, **kwargs):
            super().__init__(*args, **kwargs)
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_interval = checkpoint_interval
            self.round_count = 0

        def aggregate_fit(self, server_round, results, failures):
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            self.round_count = server_round

            # Save checkpoint periodically
            if self.checkpoint_dir and server_round % self.checkpoint_interval == 0:
                self._save_checkpoint(server_round, aggregated_parameters)

            return aggregated_parameters, aggregated_metrics

        def _save_checkpoint(self, server_round, parameters):
            if parameters is None:
                return
            try:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"federated_round_{server_round}.pth")
                # Convert parameters to model state dict
                parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
                state_dict = OrderedDict({
                    k: torch.tensor(v) for k, v in zip(temp_model.state_dict().keys(), parameters_ndarrays)
                })
                torch.save({
                    'model_state_dict': state_dict,
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims,
                    'dropout_rate': dropout_rate,
                    'round': server_round,
                    'config': {
                        'num_clients': num_clients,
                        'num_rounds': num_rounds,
                        'partition_strategy': partition_strategy,
                        'dirichlet_alpha': dirichlet_alpha,
                        'local_epochs': local_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'val_ratio': val_ratio,
                        'device': device
                    }
                }, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"  Warning: Failed to save checkpoint: {e}")

    # Create strategy with checkpointing
    checkpoint_dir = "/Users/admin/Desktop/federated-ids/models/checkpoints" if save_model else None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n5. Creating federated learning strategy with checkpointing...")
    strategy = CheckpointingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        initial_parameters=initial_parameters,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
    )

    # Print configuration
    print(f"\n6. Simulation configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Partition strategy: {partition_strategy}")
    print(f"  Dirichlet alpha: {dirichlet_alpha}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Validation ratio: {val_ratio}")
    print(f"  Device: {device}")
    print(f"  Warm start: {warm_start}")
    print(f"  Checkpoint interval: {checkpoint_interval}")

    # Start simulation
    print(f"\n7. Starting federated learning simulation...")
    flwr_start_time = time.time()

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if device == "cpu" else 0.5},
        ray_init_args={"_temp_dir": "/tmp/claude", "include_dashboard": False},
    )

    flwr_time = time.time() - flwr_start_time
    total_time = time.time() - start_time

    print(f"\n8. Simulation completed!")
    print(f"  Flower simulation time: {flwr_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")

    # Process and display results
    print(f"\n9. Results:")
    if history.losses_distributed:
        print(f"  Losses distributed: {len(history.losses_distributed)} rounds")
        for round_num, loss in history.losses_distributed[-5:]:
            print(f"    Round {round_num}: {loss:.4f}")
    if history.losses_centralized:
        print(f"  Losses centralized: {len(history.losses_centralized)} rounds")
        for round_num, loss in history.losses_centralized[-5:]:
            print(f"    Round {round_num}: {loss:.4f}")
    if history.metrics_centralized:
        print(f"  Metrics centralized: {list(history.metrics_centralized.keys())}")
        if "accuracy" in history.metrics_centralized:
            accuracies = [acc for _, acc in history.metrics_centralized["accuracy"]]
            if accuracies:
                print(f"  Final accuracy: {accuracies[-1]:.4f}")
                print(f"  Best accuracy: {max(accuracies):.4f}")
                print(f"  Accuracy progression: {[f'{a:.4f}' for a in accuracies[-10:]]}")
        if "loss" in history.metrics_centralized:
            losses = [loss for _, loss in history.metrics_centralized["loss"]]
            if losses:
                print(f"  Final loss: {losses[-1]:.4f}")
                print(f"  Best loss: {min(losses):.4f}")

    # Extract final evaluation results
    final_results = {}
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        accuracies = [acc for _, acc in history.metrics_centralized["accuracy"]]
        if accuracies:
            final_results["accuracy"] = accuracies[-1]
            final_results["best_accuracy"] = max(accuracies)
            final_results["accuracy_history"] = accuracies

    if history.metrics_centralized and "loss" in history.metrics_centralized:
        losses = [loss for _, loss in history.metrics_centralized["loss"]]
        if losses:
            final_results["loss"] = losses[-1]
            final_results["best_loss"] = min(losses)
            final_results["loss_history"] = losses

    # Save final model if requested
    if save_model and hasattr(strategy, 'round_count') and strategy.round_count > 0:
        model_dir = "/Users/admin/Desktop/federated-ids/models"
        os.makedirs(model_dir, exist_ok=True)

        # Get final parameters from the last round (we need to reconstruct from history or server state)
        # Since we can't easily get final params from strategy, we'll evaluate and note
        model_path = os.path.join(model_dir, "federated_mlp_improved.pth")
        # We'll save a model with the final evaluated metrics
        # Note: In practice, we'd extract the final parameters from the server
        # For now, we save the config and results
        torch.save({
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'results': final_results,
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
        print(f"  Final results saved to: {model_path}")

        # Also save results as JSON
        results_json_path = os.path.join(model_dir, "federated_improved_results.json")
        with open(results_json_path, 'w') as f:
            json.dump({
                "results": final_results,
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
                "simulation_time": total_time,
                "flower_time": flwr_time
            }, f, indent=2)
        print(f"  Results JSON saved to: {results_json_path}")

    # Prepare final results dictionary
    results = {
        "history": history,
        "final_results": final_results,
        "simulation_time": total_time,
        "flower_time": flwr_time,
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "partition_strategy": partition_strategy,
            "dirichlet_alpha": dirichlet_alpha,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_ratio": val_ratio,
            "device": device,
            "warm_start": warm_start,
            "checkpoint_interval": checkpoint_interval,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate
        }
    }

    print(f"\n10. Improved Federated Learning Simulation Completed!")
    print("=" * 70)

    return results


def main():
    """Main function to run improved federated learning simulation."""
    # Run the simulation with improved settings
    results = run_federated_simulation(
        num_clients=5,
        num_rounds=50,  # Increased from 5 to 50
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        local_epochs=5,  # Increased from 2 to 5
        batch_size=64,
        learning_rate=0.001,
        val_ratio=0.2,
        device="cpu",
        save_model=True,
        warm_start=True,  # Enable warm start from centralized model
        checkpoint_interval=10
    )

    return results


if __name__ == "__main__":
    main()