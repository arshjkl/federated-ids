"""
Federated Learning Client for Intrusion Detection System.
Implements a Flower client that trains the MLP model on local data.
"""

import os
import numpy as np
import torch
import flwr as fl
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

from src.models.mlp import MLPIDS
from src.data.preprocessing import CICIDS2017Preprocessor


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning with MLP model.
    """

    def __init__(
        self,
        client_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        local_epochs: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize the Flower client.

        Args:
            client_id: Unique identifier for this client
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            local_epochs: Number of local training epochs
            device: Device to use for training
        """
        self.client_id = client_id
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.device = device

        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Initialize model
        self.model = MLPIDS(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(device)

        # Loss function and optimizer
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare PyTorch data loaders."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.LongTensor(self.y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
        y_val_tensor = torch.LongTensor(self.y_val).to(self.device)

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            X_train_tensor, y_train_tensor
        )
        val_dataset = torch.utils.data.TensorDataset(
            X_val_tensor, y_val_tensor
        )

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Get model parameters as a list of NumPy arrays.

        Args:
            config: Configuration (unused)

        Returns:
            Model parameters as list of NumPy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of NumPy arrays.

        Args:
            parameters: Model parameters as list of NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in params_dict
        })
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.

        Args:
            parameters: Current model parameters
            config: Configuration (unused)

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)

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

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y.float())

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
                epoch_total += batch_y.size(0)
                epoch_correct += (predicted == batch_y).sum().item()

            train_loss += epoch_loss
            train_correct += epoch_correct
            train_total += epoch_total

            # Print progress for first epoch only to avoid too much output
            if epoch == 0 and self.local_epochs > 1:
                print(
                    f"[Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs} "
                    f"Loss: {epoch_loss/len(self.train_loader):.4f} "
                    f"Acc: {epoch_correct/epoch_total:.4f}"
                )

        # Calculate average metrics
        avg_train_loss = train_loss / (self.local_epochs * len(self.train_loader))
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Evaluate on validation set
        val_loss, val_acc = self._evaluate()

        # Prepare metrics
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(avg_train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "local_epochs": self.local_epochs,
            "train_samples": len(self.X_train),
            "val_samples": len(self.X_val)
        }

        return self.get_parameters(config), len(self.X_train), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local validation data.

        Args:
            parameters: Model parameters to evaluate
            config: Configuration (unused)

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)

        # Evaluate
        loss, accuracy = self._evaluate()

        # Prepare metrics
        metrics = {
            "client_id": self.client_id,
            "val_loss": float(loss),
            "val_acc": float(accuracy),
            "val_samples": len(self.X_val)
        }

        return float(loss), len(self.X_val), metrics

    def _evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.

        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y.float())

                total_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy


def create_client(
    client_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    **kwargs
) -> FlowerClient:
    """
    Factory function to create a Flower client.

    Args:
        client_id: Client identifier
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        input_dim: Number of input features
        **kwargs: Additional arguments for FlowerClient

    Returns:
        Configured FlowerClient instance
    """
    return FlowerClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage (for testing)
    print("Testing FlowerClient creation...")

    # Create dummy data for testing
    input_dim = 70
    num_samples = 1000

    X_dummy = np.random.randn(num_samples, input_dim).astype(np.float32)
    y_dummy = np.random.randint(0, 2, num_samples).astype(np.int64)

    # Split into train/val
    split_idx = int(0.8 * num_samples)
    X_train = X_dummy[:split_idx]
    y_train = y_dummy[:split_idx]
    X_val = X_dummy[split_idx:]
    y_val = y_dummy[split_idx:]

    # Create client
    client = FlowerClient(
        client_id="test_client",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        local_epochs=1
    )

    # Test getting parameters
    params = client.get_parameters({})
    print(f"Retrieved {len(params)} parameter arrays")

    # Test setting parameters
    client.set_parameters(params)
    print("Successfully set parameters")

    # Test fit (will train for 1 epoch)
    params_out, num_examples, metrics = client.fit(params, {})
    print(f"Fit completed: {num_examples} examples, metrics: {metrics}")

    # Test evaluate
    loss, num_examples, metrics = client.evaluate(params, {})
    print(f"Evaluation: loss={loss:.4f}, examples={num_examples}, metrics={metrics}")

    print("FlowerClient test completed successfully!")