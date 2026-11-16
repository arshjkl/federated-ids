"""
Seed the dashboard database with the existing trained models.

Run from the project root:

    PYTHONPATH=/Users/admin/Desktop/federated-ids .venv/bin/python scripts/seed_models.py

This registers the trained checkpoints in `models/` as GlobalModel records so the
Inference page can offer them for CSV upload and prediction.
"""
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.db.session import db_manager  # noqa: E402
from src.dashboard.db.models import (  # noqa: E402
    Base,
    GlobalModel,
    GlobalModelStatus,
    TrainingAlgorithm,
)

PROJECT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT / "models"

# (checkpoint file, version, name, algorithm, architecture, rounds, clients, metrics file)
MODEL_SPECS = [
    dict(
        file="centralized_mlp.pth",
        version="centralized-baseline-v1",
        name="Centralized MLP Baseline",
        algorithm=TrainingAlgorithm.FEDAVG,
        architecture="MLPIDS",
        training_rounds=20,
        participating_clients=1,
        metrics_file="centralized_mlp_results.json",
        architecture_config={"input_dim": 70, "hidden_dims": [256, 128, 64], "use_batchnorm": True},
    ),
    dict(
        file="centralized_mlp_improved.pth",
        version="centralized-improved-v1",
        name="Centralized MLP Improved",
        algorithm=TrainingAlgorithm.FEDAVG,
        architecture="ImprovedMLPIDS",
        training_rounds=20,
        participating_clients=1,
        metrics_file=None,
        architecture_config={"input_dim": 70, "hidden_dims": [256, 128, 64, 32], "use_batchnorm": True},
    ),
    dict(
        file="federated_mlp_manual.pth",
        version="federated-manual-v1",
        name="Manual FedAvg MLP",
        algorithm=TrainingAlgorithm.FEDAVG,
        architecture="MLPIDS",
        training_rounds=50,
        participating_clients=5,
        metrics_file="manual_federated_results.json",
        architecture_config={"input_dim": 70, "hidden_dims": [256, 128, 64], "use_batchnorm": True},
    ),
    dict(
        file="manual_federated_mlp.pth",
        version="federated-manual-improved-v1",
        name="Improved Manual FedAvg MLP",
        algorithm=TrainingAlgorithm.FEDAVG,
        architecture="MLPIDS",
        training_rounds=50,
        participating_clients=5,
        metrics_file="manual_federated_improved_results.json",
        architecture_config={"input_dim": 70, "hidden_dims": [256, 128, 64, 32], "use_batchnorm": True},
    ),
    dict(
        file="federated_mlp_dp.pth",
        version="dp-federated-v1",
        name="DP-FedAvg MLP",
        algorithm=TrainingAlgorithm.DP_FEDAVG,
        architecture="MLPIDS",
        training_rounds=30,
        participating_clients=5,
        metrics_file="dp_federated_results.json",
        architecture_config={"input_dim": 70, "hidden_dims": [256, 128, 64], "use_batchnorm": False},
        privacy_config={"epsilon": 0.1457, "delta": 1e-5, "noise_multiplier": 1.0},
    ),
]


def load_metrics(metrics_file):
    if not metrics_file:
        return None
    path = MODELS_DIR / metrics_file
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    # Some files nest metrics under "results"
    if "results" in data and isinstance(data["results"], dict):
        return data["results"]
    return data


async def seed():
    await db_manager.create_tables()

    async with db_manager.session() as db:
        from sqlalchemy import select

        existing = await db.scalar(select(GlobalModel).where(GlobalModel.version == MODEL_SPECS[0]["version"]))
        if existing:
            print("Models already seeded; skipping.")
            return

        for spec in MODEL_SPECS:
            ckpt = MODELS_DIR / spec["file"]
            metrics = load_metrics(spec.get("metrics_file"))
            model = GlobalModel(
                id=uuid4(),
                version=spec["version"],
                name=spec["name"],
                description=f"Pre-trained {spec['name'].lower()} checkpoint.",
                algorithm=spec["algorithm"],
                architecture=spec["architecture"],
                architecture_config=spec.get("architecture_config"),
                checkpoint_path=str(ckpt),
                status=GlobalModelStatus.ACTIVE,
                training_rounds=spec["training_rounds"],
                participating_clients=spec["participating_clients"],
                total_samples=None,
                metrics=metrics,
                privacy_config=spec.get("privacy_config"),
                training_config=None,
                trained_at=datetime.utcnow(),
                validated_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )
            db.add(model)
            print(f"Registered {spec['name']} -> {ckpt.name}")

        await db.commit()
    print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(seed())
