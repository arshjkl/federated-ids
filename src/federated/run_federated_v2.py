"""
Federated Learning V2 - Improved Version
Uses the improved MLP architecture with:
- Focal Loss for class imbalance
- Optimal threshold selection
- Better model architecture (residual connections, LayerNorm)
- Learning rate scheduling
- More local epochs for better convergence
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
from src.models.improved_mlp import ImprovedMLPIDS, FocalLoss, create_improved_mlp_model, count_parameters, get_optimal_threshold


def load_and_partition_data(
    num_clients: int = 5,
    val_ratio: float = 0.1,
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


class ImprovedFlowerClient(fl.client.NumPyClient):
    """
    Improved Flower client with Focal Loss, better optimization, and LR scheduling.
    """

    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rate: float = 0.2,
        use_residual: bool = True,
        use_layernorm: bool = True,
        residual_every: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        local_epochs: int = 5,
        device: str = "cpu",
        loss_type: str = "focal",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        self.client_id = client_id
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.residual_every = residual_every
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.device = device
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Initialize model
        self.model = create_improved_mlp_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
            residual_every=residual_every
        ).to(device)

        # Calculate class weights
        class_counts = np.bincount(y_train)
        self.pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0

        # Loss function
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_type == 'weighted_bce':
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(self.pos_weight, device=device)
            )
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare PyTorch data loaders."""
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
        y_val_tensor = torch.LongTensor(self.y_val).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        # Set parameters from server
        self.set_parameters(parameters)

        # Local LR scheduler (cosine annealing within local epochs)
        def lr_lambda(epoch):
            progress = epoch / max(self.local_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        local_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Train model
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()

            train_loss += epoch_loss
            train_correct += epoch_correct
            train_total += epoch_total

            local_scheduler.step()

            if epoch == 0 and self.local_epochs > 1:
                print(
                    f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs} "
                    f"Loss: {epoch_loss/len(self.train_loader):.4f} "
                    f"Acc: {epoch_correct/epoch_total:.4f}"
                )

        avg_train_loss = train_loss / (self.local_epochs * len(self.train_loader))
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Evaluate on validation set
        val_loss, val_acc, val_f1, val_prec, val_rec = self._evaluate_detailed()

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(avg_train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "local_epochs": self.local_epochs,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val)
        }

        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        loss, accuracy, f1, precision, recall = self._evaluate_detailed()

        metrics = {
            "client_id": self.client_id,
            "val_loss": float(loss),
            "val_acc": float(accuracy),
            "val_f1": float(f1),
            "val_precision": float(precision),
            "val_recall": float(recall),
            "val_samples": len(self.X_val)
        }

        return float(loss), len(self.X_val), metrics

    def _evaluate_detailed(self) -> Tuple[float, float, float, float, float]:
        """Evaluate the model on validation data with detailed metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                probs = torch.sigmoid(outputs.squeeze())
                predicted = (probs >= 0.5).long()

                all_preds.append(predicted.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        return avg_loss, accuracy, f1, precision, recall


def get_evaluate_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    hidden_dims: List[int],
    dropout_rate: float,
    use_residual: bool,
    use_layernorm: bool,
    residual_every: int,
    device: str
):
    """Create evaluation function for server-side evaluation."""

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, str]):
        # Create model
        model = create_improved_mlp_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
            residual_every=residual_every
        ).to(device)

        # Set parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)

        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

        # Use Focal Loss for evaluation consistency
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                probs = torch.sigmoid(outputs.squeeze())
                predicted = (probs >= 0.5).long()

                all_preds.append(predicted.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        avg_loss = total_loss / len(test_loader)

        preds = np.concatenate(all_preds)
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(labels, probs)
        except:
            roc_auc = 0.0

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        print(f"\n[Server Round {server_round}] Evaluation:")
        print(f"  Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "loss": float(avg_loss)
        }

        return avg_loss, metrics

    return evaluate


def run_federated_v2(
    num_clients: int = 5,
    num_rounds: int = 50,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    local_epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.1,
    device: str = "cpu",
    save_model: bool = True,
    loss_type: str = "focal",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_residual: bool = True,
    use_layernorm: bool = True,
    hidden_dims: List[int] = None,
    dropout_rate: float = 0.2,
) -> Dict:
    """
    Run improved federated learning simulation (V2).
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64, 32]

    print("=" * 70)
    print("Starting IMPROVED Federated Learning V2")
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

    print(f"\n2. Model configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden layers: {hidden_dims}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Use residual: {use_residual}")
    print(f"  Use LayerNorm: {use_layernorm}")
    print(f"  Loss type: {loss_type}")

    # Create initial model (for parameter counting)
    temp_model = create_improved_mlp_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        use_layernorm=use_layernorm
    )
    print(f"  Model parameters: {count_parameters(temp_model):,}")

    # Initial parameters
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in temp_model.state_dict().items()]
    )

    # Evaluation function
    print(f"\n3. Setting up evaluation function...")
    evaluate_fn = get_evaluate_fn(
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
        residual_every=2,
        device=device
    )

    # Config functions
    def fit_config(server_round: int):
        return {
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }

    def evaluate_config(server_round: int):
        return {"batch_size": batch_size}

    # Custom strategy with better aggregation
    class FedAvgWithMetrics(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_f1 = 0
            self.best_round = 0

        def aggregate_evaluate(self, server_round, results, failures):
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
                server_round, results, failures
            )

            # Track best F1
            if aggregated_metrics and "f1" in aggregated_metrics:
                f1 = aggregated_metrics["f1"]
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_round = server_round
                    print(f"\n*** New best F1: {f1:.4f} at round {server_round} ***")

            return aggregated_loss, aggregated_metrics

    print(f"\n4. Creating federated learning strategy...")
    strategy = FedAvgWithMetrics(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
    )

    print(f"\n5. Simulation configuration:")
    print(f"  Number of clients: {num_clients}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Partition strategy: {partition_strategy}")
    print(f"  Dirichlet alpha: {dirichlet_alpha}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Validation ratio: {val_ratio}")
    print(f"  Device: {device}")
    print(f"  Loss type: {loss_type}")
    print(f"  Focal alpha: {focal_alpha}, gamma: {focal_gamma}")

    # Define client function
    def client_fn(cid: str) -> ImprovedFlowerClient:
        client_id = int(cid)
        X_train, y_train = client_train_data[client_id]
        X_val, y_val = client_val_data[client_id]

        return ImprovedFlowerClient(
            client_id=cid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            local_epochs=local_epochs,
            device=device,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    # Start simulation
    print(f"\n6. Starting federated learning simulation...")
    flwr_start_time = time.time()

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

    print(f"\n7. Simulation completed!")
    print(f"  Flower simulation time: {flwr_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")

    # Extract final results
    final_results = {}
    if history.metrics_centralized:
        if "f1" in history.metrics_centralized:
            f1s = [m for _, m in history.metrics_centralized["f1"]]
            if f1s:
                final_results["f1"] = f1s[-1]
                final_results["best_f1"] = max(f1s)
                final_results["f1_history"] = f1s

        if "accuracy" in history.metrics_centralized:
            accs = [m for _, m in history.metrics_centralized["accuracy"]]
            if accs:
                final_results["accuracy"] = accs[-1]
                final_results["best_accuracy"] = max(accs)
                final_results["accuracy_history"] = accs

        if "precision" in history.metrics_centralized:
            precs = [m for _, m in history.metrics_centralized["precision"]]
            if precs:
                final_results["precision"] = precs[-1]
                final_results["best_precision"] = max(precs)

        if "recall" in history.metrics_centralized:
            recs = [m for _, m in history.metrics_centralized["recall"]]
            if recs:
                final_results["recall"] = recs[-1]
                final_results["best_recall"] = max(recs)

        if "loss" in history.metrics_centralized:
            losses = [m for _, m in history.metrics_centralized["loss"]]
            if losses:
                final_results["loss"] = losses[-1]
                final_results["best_loss"] = min(losses)
                final_results["loss_history"] = losses

    if hasattr(strategy, 'best_f1'):
        final_results["server_best_f1"] = strategy.best_f1
        final_results["server_best_round"] = strategy.best_round

    # Save results
    if save_model:
        model_dir = "/Users/admin/Desktop/federated-ids/models"
        os.makedirs(model_dir, exist_ok=True)

        results_path = os.path.join(model_dir, "federated_v2_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "results": final_results,
                "config": {
                    "num_clients": num_clients,
                    "num_rounds": num_rounds,
                    "partition_strategy": partition_strategy,
                    "dirichlet_alpha": dirichlet_alpha,
                    "local_epochs": local_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "val_ratio": val_ratio,
                    "device": device,
                    "loss_type": loss_type,
                    "focal_alpha": focal_alpha,
                    "focal_gamma": focal_gamma,
                    "use_residual": use_residual,
                    "use_layernorm": use_layernorm,
                    "hidden_dims": hidden_dims,
                    "dropout_rate": dropout_rate,
                    "input_dim": input_dim
                },
                "simulation_time": total_time,
                "flower_time": flwr_time
            }, f, indent=2)
        print(f"  Results saved to: {results_path}")

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
            "weight_decay": weight_decay,
            "val_ratio": val_ratio,
            "device": device,
            "loss_type": loss_type,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "use_residual": use_residual,
            "use_layernorm": use_layernorm,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "input_dim": input_dim
        }
    }

    print(f"\n8. Federated Learning V2 Completed!")
    print("=" * 70)

    return results


def main():
    results = run_federated_v2(
        num_clients=5,
        num_rounds=30,  # Start with fewer rounds for testing
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        local_epochs=5,
        batch_size=256,
        learning_rate=0.001,
        weight_decay=1e-4,
        val_ratio=0.1,
        device="cpu",
        save_model=True,
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