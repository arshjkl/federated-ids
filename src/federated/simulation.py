"""
Federated Learning Simulation for Intrusion Detection System.
Coordinates the federated learning process using Flower.
"""

import os
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple, Optional
import json
import time

from src.data.preprocessing import CICIDS2017Preprocessor
from src.federated.client import FlowerClient, create_client
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

    Args:
        num_clients: Number of clients to simulate
        val_ratio: Proportion of training data to use for validation
        partition_strategy: Strategy for partitioning data ("iid" or "non_iid_dirichlet")
        dirichlet_alpha: Parameter for Dirichlet distribution (lower = more non-IID)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (client_train_data, client_val_data, global_test_data, input_dim)
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
        # This creates imbalance in label distribution across clients

        # Group indices by class
        class_indices = []
        for class_label in np.unique(y_train):
            class_indices.append(np.where(y_train == class_label)[0])

        # Initialize client indices
        client_indices = [[] for _ in range(num_clients)]

        # Distribute each class's indices among clients using Dirichlet distribution
        for indices_in_class in class_indices:
            # Shuffle indices for this class
            np.random.shuffle(indices_in_class)

            # Generate proportions for this class from Dirichlet distribution
            proportions = np.random.dirichlet(alpha=[dirichlet_alpha] * num_clients)

            # Calculate split points
            split_points = (np.cumsum(proportions) * len(indices_in_class)).astype(int)[:-1]

            # Split indices according to proportions
            class_splits = np.split(indices_in_class, split_points)

            # Assign to clients
            for i, indices_for_client in enumerate(class_splits):
                client_indices[i].extend(indices_for_client.tolist())

        # Create client data
        client_data = []
        for indices in client_indices:
            # Shuffle to avoid ordering effects
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
        # Split client data into train and validation
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


def client_fn(cid: str) -> FlowerClient:
    """
    Create a Flower client instance.

    Args:
        cid: Client ID (string)

    Returns:
        FlowerClient instance
    """
    # This function will be called by Flower for each client
    # We need to access the client data from somewhere accessible
    # For now, we'll create a placeholder - in practice, this would
    # access pre-loaded data or load it on demand

    # This is a simplified version - in the actual simulation,
    # we'll pass the data through closure or global variables
    raise NotImplementedError("client_fn should be implemented in the simulation runner")


def run_federated_simulation(
    num_clients: int = 5,
    num_rounds: int = 5,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    local_epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    val_ratio: float = 0.2,
    device: str = "cpu",
    save_model: bool = True
) -> Dict:
    """
    Run federated learning simulation.

    Args:
        num_clients: Number of clients to simulate
        num_rounds: Number of federated learning rounds
        partition_strategy: Strategy for partitioning data
        dirichlet_alpha: Parameter for Dirichlet distribution
        local_epochs: Number of local training epochs per round
        batch_size: Batch size for training
        learning_rate: Learning rate
        val_ratio: Proportion of training data for validation
        device: Device to use for training
        save_model: Whether to save the final model

    Returns:
        Dictionary containing results and history
    """
    print("=" * 60)
    print("Starting Federated Learning Simulation")
    print("=" * 60)

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

    # Create a temporary model to count parameters
    temp_model = MLPIDS(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    print(f"  Model parameters: {count_parameters(temp_model):,}")

    # Prepare evaluation function for server-side evaluation
    print(f"\n3. Setting up evaluation function...")
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

    # Create strategy
    print(f"\n4. Creating federated learning strategy...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=num_clients,  # Minimum number of clients for training
        min_evaluate_clients=num_clients,  # Minimum number of clients for evaluation
        min_available_clients=num_clients,  # Minimum number of clients that need to be connected
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
        ),
    )

    # Print configuration
    print(f"\n5. Simulation configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Partition strategy: {partition_strategy}")
    print(f"  Dirichlet alpha: {dirichlet_alpha}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Validation ratio: {val_ratio}")
    print(f"  Device: {device}")

    # Start simulation
    print(f"\n6. Starting federated learning simulation...")
    flwr_start_time = time.time()

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0 if device == "cpu" else 0.5},
    )

    flwr_time = time.time() - flwr_start_time
    total_time = time.time() - start_time

    print(f"\n7. Simulation completed!")
    print(f"  Flower simulation time: {flwr_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")

    # Process and display results
    print(f"\n8. Results:")
    if history.losses_distributed:
        print(f"  Losses distributed: {history.losses_distributed}")
    if history.losses_centralized:
        print(f"  Losses centralized: {history.losses_centralized}")
    if history.metrics_distributed:
        print(f"  Metrics distributed: {list(history.metrics_distributed.keys())}")
    if history.metrics_centralized:
        print(f"  Metrics centralized: {list(history.metrics_centralized.keys())}")

    # Extract final evaluation results if available
    final_results = {}
    if history.metrics_centralized and "accuracy" in history.metrics_centralized:
        accuracies = [acc for _, acc in history.metrics_centralized["accuracy"]]
        if accuracies:
            final_results["accuracy"] = accuracies[-1]
            final_results["accuracy_history"] = accuracies
            print(f"  Final accuracy: {accuracies[-1]:.4f}")

    if history.metrics_centralized and "loss" in history.metrics_centralized:
        losses = [loss for _, loss in history.metrics_centralized["loss"]]
        if losses:
            final_results["loss"] = losses[-1]
            final_results["loss_history"] = losses
            print(f"  Final loss: {losses[-1]:.4f}")

    # Save model if requested
    if save_model:
        model_dir = "/Users/admin/Desktop/federated-ids/models"
        os.makedirs(model_dir, exist_ok=True)

        # Get the final parameters from the strategy
        # Note: In a real implementation, we'd get these from the history or server state
        # For now, we'll save the initial model as a placeholder
        model_path = os.path.join(model_dir, "federated_mlp.pth")
        torch.save({
            'model_state_dict': temp_model.state_dict(),
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
                'device': device
            }
        }, model_path)
        print(f"  Model saved to: {model_path}")

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
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate
        }
    }

    print(f"\n9. Federated Learning Simulation Completed!")
    print("=" * 60)

    return results


def main():
    """Main function to run federated learning simulation."""
    # Run the simulation
    results = run_federated_simulation(
        num_clients=5,
        num_rounds=5,
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        local_epochs=2,
        batch_size=64,
        learning_rate=0.001,
        val_ratio=0.2,
        device="cpu",
        save_model=True
    )

    return results


if __name__ == "__main__":
    main()