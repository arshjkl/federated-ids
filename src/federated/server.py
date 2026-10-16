"""
Federated Learning Server for Intrusion Detection System.
Implements a Flower server that coordinates federated averaging.
"""

import flwr as fl
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import OrderedDict
import torch

from src.models.mlp import MLPIDS, count_parameters
from src.data.preprocessing import CICIDS2017Preprocessor


def get_evaluate_fn(
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    hidden_dims: List[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
    device: str = "cpu"
):
    """
    Return an evaluation function for server-side evaluation.

    Args:
        X_test: Test features
        y_test: Test labels
        input_dim: Number of input features
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        device: Device to use for evaluation

    Returns:
        Evaluation function
    """
    def evaluate(
        server_round: int,
        parameters: fl.common.Parameters,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Create model
        model = MLPIDS(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        ).to(device)

        # Convert parameters to model state dict
        parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            y_tensor = torch.LongTensor(y_test).to(device)

            outputs = model(X_tensor)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.squeeze(), y_tensor.float()
            ).item()

            predicted = (torch.sigmoid(outputs.squeeze()) >= 0.5).long()
            accuracy = (predicted == y_tensor).float().mean().item()

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """
    Return training configuration for each round.

    Args:
        server_round: Current round of federated learning

    Returns:
        Configuration dictionary
    """
    config = {
        "server_round": server_round,
        "local_epochs": 2,  # Constant for now, could be made adaptive
    }
    return config


def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """
    Return evaluation configuration for each round.

    Args:
        server_round: Current round of federated learning

    Returns:
        Configuration dictionary
    """
    config = {
        "server_round": server_round,
    }
    return config


def get_on_fit_config_fn() -> Callable[[int], Dict[str, fl.common.Scalar]]:
    """Return a function that returns fit config."""
    return fit_config


def get_on_evaluate_config_fn() -> Callable[[int], Dict[str, fl.common.Scalar]]:
    """Return a function that returns evaluate config."""
    return evaluate_config


if __name__ == "__main__":
    # Example usage (for testing)
    print("Testing server components...")

    # Create dummy data for testing
    input_dim = 70
    X_test_dummy = np.random.randn(100, input_dim).astype(np.float32)
    y_test_dummy = np.random.randint(0, 2, 100).astype(np.int64)

    # Create evaluate function
    evaluate_fn = get_evaluate_fn(
        X_test=X_test_dummy,
        y_test=y_test_dummy,
        input_dim=input_dim
    )

    # Test evaluate function with dummy parameters
    dummy_params = [np.random.randn(*size).astype(np.float32)
                   for size in [(70, 128), (128,), (128, 64), (64,),
                              (64, 32), (32,), (32, 1), (1,)]]

    # Convert to Parameters object
    tensors = [fl.common.ndarrays_to_parameters(dummy_params)]
    parameters = tensors[0] if tensors else fl.common.Parameters(tensors=[], tensor_type="numpy.ndarray")

    # Call evaluate function
    result = evaluate_fn(server_round=1, parameters=parameters, config={})
    if result is not None:
        loss, metrics = result
        print(f"Evaluation - Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    # Test config functions
    fit_conf = fit_config(server_round=1)
    eval_conf = evaluate_config(server_round=1)
    print(f"Fit config: {fit_conf}")
    print(f"Evaluate config: {eval_conf}")

    print("Server components test completed successfully!")