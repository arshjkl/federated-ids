#!/usr/bin/env python
"""
Register existing trained models in the dashboard database.
Run after database initialization.
"""

import asyncio
import sys
import os
from pathlib import Path

# Load .env file (without python-dotenv dependency)
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dashboard.db.session import db_manager
from src.dashboard.db.models import GlobalModel, TrainingAlgorithm, GlobalModelStatus
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


MODELS_TO_REGISTER = [
    {
        "version": "centralized_v1",
        "name": "Centralized MLP Baseline",
        "description": "Baseline centralized MLP trained on full CIC-IDS2017 (20 epochs, BCE loss)",
        "algorithm": TrainingAlgorithm.FEDAVG,
        "architecture": "MLPIDS",
        "architecture_config": {"hidden_dims": [128, 64, 32], "dropout": 0.3, "use_batchnorm": True},
        "checkpoint_path": "models/centralized_mlp.pth",
        "status": GlobalModelStatus.ACTIVE,
        "training_rounds": 20,
        "participating_clients": 1,
        "total_samples": 2017889,
        "metrics": {
            "accuracy": 0.9771,
            "precision": 0.9616,
            "recall": 0.9001,
            "f1": 0.9298,
            "roc_auc": 0.9981,
            "fpr": 0.0073,
            "fnr": 0.0999,
        },
        "training_config": {
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 0.001,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
        },
    },
    {
        "version": "centralized_improved_v1",
        "name": "Centralized Improved MLP",
        "description": "Improved MLP with residual connections, GELU, LayerNorm, FocalLoss, threshold optimization",
        "algorithm": TrainingAlgorithm.FEDAVG,
        "architecture": "ImprovedMLPIDS",
        "architecture_config": {"hidden_dims": [256, 128, 64, 32], "dropout": 0.3, "use_batchnorm": True},
        "checkpoint_path": "models/centralized_mlp_improved.pth",
        "status": GlobalModelStatus.ACTIVE,
        "training_rounds": 20,
        "participating_clients": 1,
        "total_samples": 2017889,
        "metrics": {
            "accuracy": 0.997187,
            "precision": 0.995398,
            "recall": 0.987907,
            "f1": 0.991639,
            "roc_auc": 0.999906,
            "fpr": 0.000928,
            "fnr": 0.012093,
        },
        "optimal_threshold": 0.58,
        "training_time": 589.7,
        "training_config": {
            "epochs": 20,
            "batch_size": 256,
            "learning_rate": 0.001,
            "loss": "FocalLoss",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        },
    },
    {
        "version": "federated_manual_v1",
        "name": "Manual FedAvg (50 rounds)",
        "description": "Manual FedAvg with 5 clients, Dirichlet α=0.5, warm start from centralized baseline",
        "algorithm": TrainingAlgorithm.FEDAVG,
        "architecture": "MLPIDS",
        "architecture_config": {"hidden_dims": [128, 64, 32], "dropout": 0.3, "use_batchnorm": True},
        "checkpoint_path": "models/federated_mlp_manual.pth",
        "status": GlobalModelStatus.ACTIVE,
        "training_rounds": 50,
        "participating_clients": 5,
        "total_samples": 2017889,
        "metrics": {
            "accuracy": 0.9587,
            "precision": 0.8818,
            "recall": 0.8721,
            "f1": 0.8770,
            "fpr": 0.0237,
            "fnr": 0.1279,
        },
        "training_config": {
            "rounds": 50,
            "local_epochs": 5,
            "batch_size": 128,
            "learning_rate": 0.001,
            "max_samples_per_client": 100000,
            "partitioning": "dirichlet_0.5",
            "warm_start": True,
        },
    },
    {
        "version": "dp_federated_v1",
        "name": "DP-FedAvg (30 rounds, ε=0.1457)",
        "description": "DP-FedAvg with Opacus DP-SGD, moments accountant, no BatchNorm",
        "algorithm": TrainingAlgorithm.DP_FEDAVG,
        "architecture": "MLPIDS",
        "architecture_config": {"hidden_dims": [128, 64, 32], "dropout": 0.3, "use_batchnorm": False},
        "checkpoint_path": "models/federated_mlp_dp.pth",
        "status": GlobalModelStatus.ACTIVE,
        "training_rounds": 30,
        "participating_clients": 5,
        "total_samples": 2017889,
        "metrics": {
            "accuracy": 0.9392,
            "precision": 0.7371,
            "recall": 0.9948,
            "f1": 0.8468,
            "roc_auc": 0.9950,
            "fpr": 0.0721,
            "fnr": 0.0052,
        },
        "privacy_config": {
            "epsilon": 0.1457,
            "delta": 1e-5,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "accountant": "moments",
            "secure_mode": False,
        },
        "training_config": {
            "rounds": 30,
            "local_epochs": 5,
            "batch_size": 128,
            "learning_rate": 0.001,
            "max_samples_per_client": 100000,
            "partitioning": "dirichlet_0.5",
        },
    },
]


async def register_models():
    await db_manager.create_tables()

    async with db_manager.session() as db:
        for model_data in MODELS_TO_REGISTER:
            # Check if already exists
            existing = await db.execute(
                select(GlobalModel).where(GlobalModel.version == model_data["version"])
            )
            if existing.scalar_one_or_none():
                print(f"✓ Model {model_data['version']} already registered")
                continue

            model = GlobalModel(**model_data)
            db.add(model)
            print(f"✓ Registered: {model_data['name']} ({model_data['version']})")

        await db.commit()
        print("\n✓ All models registered successfully!")

    await db_manager.close()


if __name__ == "__main__":
    from sqlalchemy import select
    asyncio.run(register_models())