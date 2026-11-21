"""Phase 3: Centralized MLP validation.

Loads models/centralized_mlp.pth, runs a forward pass on the processed test
set, and compares computed metrics against models/centralized_mlp_results.json.
Writes reports/centralized_validation.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load checkpoint
    ckpt_path = os.path.join(MODELS_DIR, "centralized_mlp.pth")
    assert os.path.exists(ckpt_path), f"missing {ckpt_path}"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    input_dim = checkpoint["input_dim"]
    hidden_dims = checkpoint["hidden_dims"]
    dropout_rate = checkpoint.get("dropout_rate", 0.3)

    # Build model + load weights
    from src.models.mlp import MLPIDS

    model = MLPIDS(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
    )
    state_dict = checkpoint["model_state_dict"]
    network_state_dict = {}
    for k, v in state_dict.items():
        network_state_dict[k[8:] if k.startswith("network.") else k] = v
    model.network.load_state_dict(network_state_dict, strict=True)
    model.eval()

    # Load processed test data (X_test 269 MB — fine to load once)
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy")).astype(np.float32)

    # Forward pass in batches to keep memory bounded
    batch_size = 4096
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.FloatTensor(X_test[i : i + batch_size])
            probs = model.predict_proba(xb)  # (B, 2) — prob of class 1 is col 1
            all_probs.append(probs[:, 1].numpy())
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_test.astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    # Confusion matrix counts
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    report = {
        "model": "centralized_mlp",
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "n_test": int(len(y_test)),
        "metrics": metrics,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    # Compare against saved results
    saved_path = os.path.join(MODELS_DIR, "centralized_mlp_results.json")
    saved = None
    if os.path.exists(saved_path):
        with open(saved_path) as f:
            saved = json.load(f)
        report["saved_metrics"] = saved
        report["metrics_match"] = {}
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            saved_val = saved.get(key)
            computed_val = metrics[key]
            if saved_val is not None:
                report["metrics_match"][key] = {
                    "saved": saved_val,
                    "computed": computed_val,
                    "abs_diff": float(abs(saved_val - computed_val)),
                    "match": abs(saved_val - computed_val) < 1e-6,
                }

    out_path = os.path.join(REPORTS_DIR, "centralized_validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Phase 3: Centralized MLP validation ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test samples: {report['n_test']}")
    print(f"Computed metrics: accuracy={metrics['accuracy']:.6f} "
          f"precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
          f"f1={metrics['f1']:.6f} roc_auc={metrics['roc_auc']:.6f}")
    if saved:
        all_match = all(v["match"] for v in report["metrics_match"].values())
        print(f"Metrics match saved results: {all_match}")
        for key, v in report["metrics_match"].items():
            print(f"  {key}: saved={v['saved']:.6f} computed={v['computed']:.6f} "
                  f"diff={v['abs_diff']:.2e} match={v['match']}")
    print(f"Report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
