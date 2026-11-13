"""
Differentially-private Federated Averaging (DP-FedAvg) for the IDS pipeline.

Each client trains locally with DP-SGD (per-example gradient clipping +
Gaussian noise via :class:`src.privacy.dp_sgd.DPSGDTrainer`), then the server
aggregates updates with sample-weighted FedAvg. Privacy spending
(epsilon/delta) is tracked per client and per run.

The non-private baseline is preserved: set ``noise_multiplier=0.0`` to run
FedAvg with gradient clipping but no noise (no privacy spend).
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federated.run_manual_federated import (
    load_and_partition_data,
    get_model_parameters,
    set_model_parameters,
    average_parameters,
    evaluate_model,
    load_centralized_model,
)
from src.models.mlp import MLPIDS
from src.privacy.dp_sgd import DPSGDTrainer, privacy_report

PROJECT_ROOT = "/Users/admin/Desktop/federated-ids"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")


def _client_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)


def _transfer_linear_weights(src_state_dict, dst_network):
    """Copy Linear weight/bias tensors from ``src`` (may include BatchNorm params
    and buffers) into ``dst_network`` (no BatchNorm), matching by the order in
    which Linear layers appear.

    Linear weights are 2-D tensors; a Linear bias is the 1-D tensor that
    immediately follows a Linear weight. BatchNorm weight/bias and running
    statistics are skipped.
    """
    dst_keys = list(dst_network.state_dict().keys())
    linear_items: List[Tuple[str, torch.Tensor]] = []
    prev_was_2d = False
    for _, v in src_state_dict.items():
        t = v if torch.is_tensor(v) else torch.tensor(v)
        if t.dim() == 2:
            # Linear weight
            linear_items.append(("weight", t.clone()))
            prev_was_2d = True
        elif prev_was_2d and t.dim() == 1:
            # Linear bias immediately follows its Linear weight
            linear_items.append(("bias", t.clone()))
            prev_was_2d = False
        else:
            prev_was_2d = False

    if len(linear_items) != len(dst_keys):
        raise ValueError(
            f"Linear tensor count mismatch: {len(linear_items)} source tensors "
            f"vs {len(dst_keys)} destination keys"
        )

    new_state_dict = OrderedDict(
        (key, tensor) for (_, tensor), key in zip(linear_items, dst_keys)
    )
    dst_network.load_state_dict(new_state_dict, strict=True)


def train_client_dp(
    client_model: MLPIDS,
    X_train: np.ndarray,
    y_train: np.ndarray,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
    noise_multiplier: float,
    delta: float,
    device: str,
) -> Dict:
    """Train one client with DP-SGD for ``local_epochs`` epochs.

    Returns metrics including the privacy spent (epsilon, delta).
    """
    dataloader = _client_dataloader(X_train, y_train, batch_size)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
    trainer = DPSGDTrainer(
        model=client_model,
        optimizer=optimizer,
        dataloader=dataloader,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        delta=delta,
        use_opacus=True,
    )

    epoch_losses = []
    for _ in range(local_epochs):
        info = trainer.train_epoch()
        epoch_losses.append(info["loss"])

    epsilon, spent_delta = trainer.get_privacy_spent()
    return {
        "train_losses": epoch_losses,
        "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epsilon": epsilon,
        "delta": spent_delta,
        "dp_enabled": trainer.is_dp(),
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
    }


def run_dp_federated(
    num_clients: int = 5,
    num_rounds: int = 30,
    partition_strategy: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    local_epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5,
    val_ratio: float = 0.2,
    device: str = "cpu",
    warm_start: bool = True,
    checkpoint_interval: int = 10,
    max_samples_per_client: int = 50000,
) -> Dict:
    """Run DP-FedAvg training and return final metrics + privacy report."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    start_time = time.time()

    # 1. Load and partition data
    print(f"[DP-FedAvg] Loading and partitioning data for {num_clients} clients...")
    client_train_data, client_val_data, (X_test, y_test), input_dim = (
        load_and_partition_data(
            num_clients=num_clients,
            val_ratio=val_ratio,
            partition_strategy=partition_strategy,
            dirichlet_alpha=dirichlet_alpha,
            random_state=42,
            max_samples_per_client=max_samples_per_client,
        )
    )

    # 2. Model configuration
    hidden_dims = [128, 64, 32]
    dropout_rate = 0.3
    print(f"[DP-FedAvg] input_dim={input_dim} hidden={hidden_dims}")

    # 3. Initialize global model (warm start from centralized model)
    # In DP mode (noise_multiplier > 0) BatchNorm is disabled on clients because
    # Opacus rejects it under per-example DP-SGD, so the global model must match.
    print("[DP-FedAvg] Initializing global model...")
    global_use_batchnorm = (noise_multiplier == 0.0)
    if warm_start:
        centralized = load_centralized_model(input_dim, hidden_dims, dropout_rate, device)
        if centralized is not None:
            global_model = MLPIDS(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_batchnorm=global_use_batchnorm,
            ).to(device)
            _transfer_linear_weights(
                centralized.network.state_dict(), global_model.network
            )
            print(
                "[DP-FedAvg] Warm-started from centralized checkpoint "
                f"(use_batchnorm={global_use_batchnorm})"
            )
        else:
            print("[DP-FedAvg]  Centralized model not found; random init.")
            global_model = MLPIDS(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_batchnorm=global_use_batchnorm,
            ).to(device)
    else:
        global_model = MLPIDS(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batchnorm=global_use_batchnorm,
        ).to(device)

    # 4. Initial evaluation
    initial_metrics = evaluate_model(global_model, X_test, y_test, device)
    print(
        f"[DP-FedAvg] Initial - acc={initial_metrics['accuracy']:.4f} "
        f"f1={initial_metrics['f1']:.4f} roc_auc={initial_metrics.get('roc_auc', 0.0):.4f}"
    )

    history = {
        "rounds": [0],
        "global_loss": [initial_metrics["loss"]],
        "global_accuracy": [initial_metrics["accuracy"]],
        "global_f1": [initial_metrics["f1"]],
        "global_roc_auc": [initial_metrics.get("roc_auc", 0.0)],
        "client_metrics": [[]],
        "privacy_per_round": [],
    }

    global_params = get_model_parameters(global_model)

    # Save round-0 checkpoint
    if checkpoint_interval > 0:
        _save_checkpoint(
            global_model, 0, input_dim, hidden_dims, dropout_rate, initial_metrics,
            {
                "num_clients": num_clients, "num_rounds": num_rounds,
                "max_grad_norm": max_grad_norm, "noise_multiplier": noise_multiplier,
                "delta": delta, "local_epochs": local_epochs,
                "batch_size": batch_size, "learning_rate": learning_rate,
                "partition_strategy": partition_strategy,
                "dirichlet_alpha": dirichlet_alpha,
            },
        )

    # 5. Federated rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n[DP-FedAvg] Round {round_num}/{num_rounds}")
        client_params_list: List[List[np.ndarray]] = []
        client_weights: List[int] = []
        client_metrics_round: List[Dict] = []
        round_epsilons: List[float] = []

        for client_id in range(num_clients):
            X_train, y_train = client_train_data[client_id]
            X_val, y_val = client_val_data[client_id]

            # BatchNorm is incompatible with per-example DP-SGD (Opacus rejects it),
            # so client models trained with DP use_batchnorm=False. The non-private
            # baseline (noise_multiplier == 0) keeps BatchNorm.
            client_use_batchnorm = (noise_multiplier == 0.0)
            client_model = MLPIDS(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_batchnorm=client_use_batchnorm,
            ).to(device)
            set_model_parameters(client_model, global_params)

            info = train_client_dp(
                client_model=client_model,
                X_train=X_train,
                y_train=y_train,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                delta=delta,
                device=device,
            )

            client_params_list.append(get_model_parameters(client_model))
            client_weights.append(len(X_train))
            round_epsilons.append(info["epsilon"])
            client_metrics_round.append({
                "client_id": client_id,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "final_train_loss": info["final_train_loss"],
                "epsilon": info["epsilon"],
                "delta": info["delta"],
                "dp_enabled": info["dp_enabled"],
            })
            print(
                f"    Client {client_id}: train_loss={info['final_train_loss']:.4f} "
                f"epsilon={info['epsilon']:.4f}"
            )

        # FedAvg aggregation
        global_params = average_parameters(client_params_list, client_weights)
        set_model_parameters(global_model, global_params)

        # Evaluate global model
        global_metrics = evaluate_model(global_model, X_test, y_test, device)
        history["rounds"].append(round_num)
        history["global_loss"].append(global_metrics["loss"])
        history["global_accuracy"].append(global_metrics["accuracy"])
        history["global_f1"].append(global_metrics["f1"])
        history["global_roc_auc"].append(global_metrics.get("roc_auc", 0.0))
        history["client_metrics"].append(client_metrics_round)
        history["privacy_per_round"].append({
            "round": round_num,
            "max_epsilon": max(round_epsilons) if round_epsilons else 0.0,
            "mean_epsilon": float(np.mean(round_epsilons)) if round_epsilons else 0.0,
            "delta": delta,
        })

        print(
            f"    Global - loss={global_metrics['loss']:.4f} acc={global_metrics['accuracy']:.4f} "
            f"f1={global_metrics['f1']:.4f} roc_auc={global_metrics.get('roc_auc',0.0):.4f} "
            f"max_eps={history['privacy_per_round'][-1]['max_epsilon']:.4f}"
        )

        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            _save_checkpoint(
                global_model, round_num, input_dim, hidden_dims, dropout_rate,
                global_metrics,
                {
                    "num_clients": num_clients, "num_rounds": num_rounds,
                    "max_grad_norm": max_grad_norm, "noise_multiplier": noise_multiplier,
                    "delta": delta,
                },
            )

    # 6. Final evaluation + privacy report
    total_time = time.time() - start_time
    final_metrics = evaluate_model(global_model, X_test, y_test, device)

    # Aggregate privacy: total steps across a client = num_rounds * local_epochs * batches_per_epoch
    # Use the max per-round epsilon as the run's privacy spend (composition across rounds is
    # conservative; here we report the per-round max and a naive composition upper bound).
    per_round_max = [p["max_epsilon"] for p in history["privacy_per_round"]]
    naive_total_epsilon = sum(per_round_max) if per_round_max else 0.0

    privacy_report_dict = privacy_report(
        steps=num_rounds * local_epochs,
        num_examples=max(len(X_train) for _, _ in client_train_data),
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        delta=delta,
        target_epsilons=[1.0, 2.0, 4.0, 8.0],
    )
    privacy_report_dict.update({
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "max_per_round_epsilon": max(per_round_max) if per_round_max else 0.0,
        "naive_composition_epsilon": naive_total_epsilon,
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "dp_enabled": noise_multiplier > 0.0,
    })

    # Save final DP model
    dp_model_path = os.path.join(MODELS_DIR, "federated_mlp_dp.pth")
    network = global_model.network if hasattr(global_model, "network") else global_model
    torch.save({
        "model_state_dict": network.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "metrics": final_metrics,
        "privacy_report": privacy_report_dict,
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "partition_strategy": partition_strategy,
            "dirichlet_alpha": dirichlet_alpha,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "noise_multiplier": noise_multiplier,
            "delta": delta,
            "device": device,
            "warm_start": warm_start,
        },
    }, dp_model_path)

    # Save results JSON
    results_json_path = os.path.join(MODELS_DIR, "dp_federated_results.json")
    import json
    with open(results_json_path, "w") as f:
        json.dump({
            "results": final_metrics,
            "privacy_report": privacy_report_dict,
            "history": history,
            "config": {
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "partition_strategy": partition_strategy,
                "dirichlet_alpha": dirichlet_alpha,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_grad_norm": max_grad_norm,
                "noise_multiplier": noise_multiplier,
                "delta": delta,
                "device": device,
                "warm_start": warm_start,
            },
            "simulation_time": total_time,
        }, f, indent=2)

    print(f"\n[DP-FedAvg] Completed in {total_time:.1f}s")
    print(f"  Final - acc={final_metrics['accuracy']:.4f} f1={final_metrics['f1']:.4f} "
          f"roc_auc={final_metrics.get('roc_auc',0.0):.4f}")
    print(f"  Privacy - epsilon={privacy_report_dict['epsilon']:.4f} "
          f"(naive composition upper bound={naive_total_epsilon:.4f}), delta={delta}")
    print(f"  Model saved: {dp_model_path}")
    print(f"  Results saved: {results_json_path}")

    return {
        "final_metrics": final_metrics,
        "privacy_report": privacy_report_dict,
        "history": history,
        "simulation_time": total_time,
    }


def _save_checkpoint(model, round_num, input_dim, hidden_dims, dropout_rate, metrics, config):
    network = model.network if hasattr(model, "network") else model
    path = os.path.join(CHECKPOINT_DIR, f"dp_federated_round_{round_num}.pth")
    torch.save({
        "model_state_dict": network.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "round": round_num,
        "metrics": metrics,
        "config": config,
    }, path)


def main():
    results = run_dp_federated(
        num_clients=5,
        num_rounds=30,
        partition_strategy="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        local_epochs=3,
        batch_size=128,
        learning_rate=0.001,
        max_grad_norm=1.0,
        noise_multiplier=1.0,
        delta=1e-5,
        device="cpu",
        warm_start=True,
        checkpoint_interval=10,
        max_samples_per_client=50000,
    )
    return results


if __name__ == "__main__":
    main()
