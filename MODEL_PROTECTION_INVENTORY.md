# Model Protection Inventory
**Federated Learning IDS Platform - Read-Only Artifacts**

This document catalogs all existing models, checkpoints, preprocessing artifacts, and research implementations that MUST remain untouched during platform development.

---

## 1. Trained Model Checkpoints (READ-ONLY)

| File | Size | Description | Model Type |
|------|------|-------------|------------|
| `models/centralized_mlp.pth` | 89.4K | Baseline centralized MLP | MLPIDS (128→64→32) |
| `models/centralized_mlp_improved.pth` | 358.6K | Improved centralized MLP | ImprovedMLPIDS (256→128→64→32) |
| `models/federated_mlp_manual.pth` | 87.8K | Manual FedAvg (50 rounds) | MLPIDS |
| `models/manual_federated_mlp.pth` | 88.4K | Manual FedAvg baseline | MLPIDS |
| `models/test_manual_federated_mlp.pth` | 88.6K | Test manual FedAvg | MLPIDS |
| `models/federated_mlp_dp.pth` | 80.7K | DP-FedAvg (30 rounds) | MLPIDS (no BatchNorm) |

### Checkpoint Directories (READ-ONLY)
- `models/checkpoints/` - Federated round checkpoints (round_0, 5, 10, 15, 20, 30, 40, 50)
- `models/checkpoints/dp_federated_round_*.pth` - DP-FedAvg round checkpoints (0, 10, 20, 30)
- `models/checkpoints/federated_v2_round_*.pth` - Improved federated checkpoints (0, 10, 20)

---

## 2. Model Results & Metrics (READ-ONLY)

| File | Description |
|------|-------------|
| `models/centralized_mlp_results.json` | Baseline centralized: Acc=97.71%, F1=92.98%, ROC-AUC=99.81% |
| `models/manual_federated_improved_results.json` | Improved FedAvg: Acc=95.87%, F1=87.70% |
| `models/manual_federated_results.json` | Baseline FedAvg results |
| `models/dp_federated_results.json` | DP-FedAvg: Acc=93.92%, F1=84.68%, ε=0.1457 |

---

## 3. Preprocessing Artifacts (READ-ONLY)

| File | Description |
|------|-------------|
| `data/processed/scaler.pkl` | StandardScaler fitted on training data only |
| `data/processed/feature_names.json` | 70 feature names in correct order |
| `data/processed/label_mapping.json` | BENIGN=0, 16 attack types → 1 |
| `data/processed/constant_features.json` | Removed constant features list |
| `data/processed/X_train.npy` | 1,077.7M (2,017,889 × 70) |
| `data/processed/X_test.npy` | 269.4M (504,473 × 70) |
| `data/processed/y_train.npy` | 15.4M |
| `data/processed/y_test.npy` | 3.8M |

---

## 4. Model Implementations (READ-ONLY - Core Research Code)

| File | Key Classes/Functions |
|------|----------------------|
| `src/models/mlp.py` | `MLPIDS`, `create_mlp_model`, `count_parameters` |
| `src/models/improved_mlp.py` | `ImprovedMLPIDS`, `ResidualBlock`, `FocalLoss`, `WeightedBCELoss`, `get_optimal_threshold` |
| `src/models/train_centralized.py` | Baseline centralized training |
| `src/models/train_improved.py` | Improved training with FocalLoss, LR scheduling, threshold optimization |
| `src/models/quick_train.py` | Quick training utility |

---

## 5. Preprocessing Pipeline (READ-ONLY)

| File | Key Class |
|------|-----------|
| `src/data/preprocessing.py` | `CICIDS2017Preprocessor` - Full pipeline: load → clean → binary labels → clean data → split → scale |
| `src/data/preprocessing_validation.py` | Validation script |

---

## 6. Federated Learning Implementations (READ-ONLY)

| File | Algorithm |
|------|-----------|
| `src/federated/run_manual_federated.py` | Manual FedAvg (50 rounds, 5 clients, Dirichlet α=0.5) |
| `src/federated/run_manual_federated_v2.py` | Improved manual FedAvg with warm start |
| `src/federated/run_dp_federated.py` | DP-FedAvg with Opacus DP-SGD |
| `src/federated/run_federated_v2.py` | Improved federated (Ray/Flower) |
| `src/federated/server.py` | Flower server |
| `src/federated/client.py` | Flower client |
| `src/federated/manual_simulation.py` | Manual simulation utilities |
| `src/federated/simulation.py` | Simulation helpers |

---

## 7. Differential Privacy (READ-ONLY)

| File | Key Components |
|------|---------------|
| `src/privacy/dp_sgd.py` | `DPSGDTrainer`, Opacus integration, privacy accounting (moments accountant), manual fallback |

---

## 8. Evaluation & Validation Scripts (READ-ONLY)

| File | Purpose |
|------|---------|
| `validate_centralized.py` | Centralized model validation |
| `tests/test_mlp.py` | Model tests |
| `tests/test_federated.py` | Federated tests |
| `tests/test_dp.py` | DP tests |
| `tests/test_preprocessing.py` | Preprocessing tests |

---

## 9. Backup Directory (READ-ONLY)

| Directory | Purpose |
|-----------|---------|
| `backups/models_backups/models_20261112_232144/` | Full model backup from 2026-11-12 |

---

## PROTECTION RULES

### ❌ NEVER DO:
- Modify any `.pth`, `.pt`, `.pkl`, `.joblib`, `.json` model artifacts
- Retrain existing improved models
- Replace improved models with baseline models
- Change preprocessing behavior (scaler, feature order, label mapping)
- Modify `src/models/mlp.py`, `src/models/improved_mlp.py`, `src/data/preprocessing.py`
- Modify federated training algorithms in `src/federated/`
- Modify DP-SGD implementation in `src/privacy/dp_sgd.py`
- Overwrite existing checkpoint paths
- Delete existing experiments or evaluation results
- Change model inference threshold silently

### ✅ DO:
- Create NEW model files in separate directories: `artifacts/global_models/`, `artifacts/simulation_checkpoints/`, `artifacts/uploaded_datasets/`
- Use NEW namespaces: `src/dashboard/`, `src/simulation/`, `src/organisation/`, `src/inference/`, `src/model_registry/`
- Load existing models for inference only
- Reference existing results for display/comparison
- Extend federated implementations via new files without changing core logic

---

## Model Loading Interface (for new code)

```python
# Example: Loading existing models for inference
from src.models.mlp import MLPIDS, create_mlp_model
from src.models.improved_mlp import ImprovedMLPIDS, create_improved_mlp_model
import torch

# Baseline centralized
model = create_mlp_model(input_dim=70)
checkpoint = torch.load('models/centralized_mlp.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Improved centralized
model = create_improved_mlp_model(input_dim=70, hidden_dims=[256,128,64,32])
checkpoint = torch.load('models/centralized_mlp_improved.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Federated models (MLPIDS architecture)
model = create_mlp_model(input_dim=70)
checkpoint = torch.load('models/federated_mlp_manual.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# DP-FedAvg model (no BatchNorm)
model = create_mlp_model(input_dim=70, use_batchnorm=False)
checkpoint = torch.load('models/federated_mlp_dp.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Preprocessing Interface (for new code)

```python
# Loading preprocessing artifacts
import pickle
import json
import numpy as np
from pathlib import Path

artifacts_dir = Path("data/processed")

# Load scaler
with open(artifacts_dir / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature names (MUST match model input order)
with open(artifacts_dir / "feature_names.json", "r") as f:
    feature_names = json.load(f)

# Load label mapping
with open(artifacts_dir / "label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Preprocess new data
X_scaled = scaler.transform(X_raw[feature_names])
```

---

## Inference Thresholds (from existing results)

| Model | Optimal Threshold | Source |
|-------|------------------|--------|
| Centralized Baseline | 0.5 (default) | `centralized_mlp_results.json` |
| Centralized Improved | ~0.3-0.4 | `train_improved.py` finds optimal |
| Manual FedAvg | 0.5 (default) | `manual_federated_improved_results.json` |
| DP-FedAvg | 0.5 (default) | `dp_federated_results.json` |

**Note:** Improved models use threshold optimization. Use `get_optimal_threshold()` from `improved_mlp.py` for new evaluations.

---

## Dataset Schema (for synthetic data generation)

**70 Features (in exact order from `feature_names.json`):**
1. Destination Port
2. Flow Duration
3. Total Fwd Packets
4. Total Backward Packets
5. Total Length of Fwd Packets
6. Total Length of Bwd Packets
7. Fwd Packet Length Max
8. Fwd Packet Length Min
9. Fwd Packet Length Mean
10. Fwd Packet Length Std
11. Bwd Packet Length Max
12. Bwd Packet Length Min
13. Bwd Packet Length Mean
14. Bwd Packet Length Std
15. Flow Bytes/s
16. Flow Packets/s
17. Flow IAT Mean
18. Flow IAT Std
19. Flow IAT Max
20. Flow IAT Min
21. Fwd IAT Total
22. Fwd IAT Mean
23. Fwd IAT Std
24. Fwd IAT Max
25. Fwd IAT Min
26. Bwd IAT Total
27. Bwd IAT Mean
28. Bwd IAT Std
29. Bwd IAT Max
30. Bwd IAT Min
31. Fwd PSH Flags
32. Fwd URG Flags
33. Fwd Header Length
34. Bwd Header Length
35. Fwd Packets/s
36. Bwd Packets/s
37. Min Packet Length
38. Max Packet Length
39. Packet Length Mean
40. Packet Length Std
41. Packet Length Variance
42. FIN Flag Count
43. SYN Flag Count
44. RST Flag Count
45. PSH Flag Count
46. ACK Flag Count
47. URG Flag Count
48. CWE Flag Count
49. ECE Flag Count
50. Down/Up Ratio
51. Average Packet Size
52. Avg Fwd Segment Size
53. Avg Bwd Segment Size
54. Fwd Header Length.1
55. Subflow Fwd Packets
56. Subflow Fwd Bytes
57. Subflow Bwd Packets
58. Subflow Bwd Bytes
59. Init_Win_bytes_forward
60. Init_Win_bytes_backward
61. act_data_pkt_fwd
62. min_seg_size_forward
63. Active Mean
64. Active Std
65. Active Max
66. Active Min
67. Idle Mean
68. Idle Std
69. Idle Max
70. Idle Min

**Labels:** Binary (0=BENIGN, 1=ATTACK)

---

*Generated: 2026-11-15 | This inventory must be consulted before any platform development.*