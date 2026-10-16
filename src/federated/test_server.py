"""
Simple test for server components.
"""

import flwr as fl
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from collections import OrderedDict


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
    # Test config functions
    fit_conf = fit_config(server_round=1)
    eval_conf = evaluate_config(server_round=1)
    print(f"Fit config: {fit_conf}")
    print(f"Evaluate config: {eval_conf}")

    # Test function getters
    fit_fn = get_on_fit_config_fn()
    eval_fn = get_on_evaluate_config_fn()

    fit_result = fit_fn(1)
    eval_result = eval_fn(1)
    print(f"Fit function result: {fit_result}")
    print(f"Evaluate function result: {eval_result}")

    print("Server components test completed successfully!")