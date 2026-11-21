# Model Compatibility & Adaptive Detection Mode Routing

## 1. Routing Philosophy

The Model Compatibility Router (`src/inference/model_router.py`) evaluates each dataset across five dimensions before selecting a detection mode:
1. **Effective Feature Coverage** ($C_{\text{eff}}$): Fraction of the 70 CIC-IDS2017 features satisfied via exact match, semantic alias match, or safe mathematical derivation.
2. **Core Feature Coverage** ($C_{\text{core}}$): Fraction of the 17 core flow, directional, packet-length, rate, and TCP-flag features available.
3. **Mapping Confidence** ($M_{\text{conf}}$): Mean confidence across all active feature mappings.
4. **Data Quality Score** ($Q_{\text{data}}$): Weighted completeness, validity, uniqueness, variance, and network relevance score (`0–100`).
5. **Training Distribution Alignment** ($D_{\text{align}}$): Percentage of mapped features whose median lies within plausible z-score bounds of the CIC-IDS2017 training `StandardScaler`.

The overall compatibility score is:
$$\text{Compatibility Score} = 0.35 C_{\text{eff}} + 0.25 C_{\text{core}} + 0.15 M_{\text{conf}} + 0.15 Q_{\text{data}} + 0.10 D_{\text{align}}$$

---

## 2. The Four Detection Modes

### Mode 1 — `supervised` (`CICIDS2017_SUPERVISED_MODEL`)
- **Selection Criteria**: $C_{\text{eff}} \ge 0.90$, $C_{\text{core}} \ge 0.85$, $M_{\text{conf}} \ge 0.80$, and $\le 10$ approximate matches.
- **Execution**: Uses the trained PyTorch neural checkpoint (`centralized_mlp_improved.pth` by default, or any selected FedAvg / DP-FedAvg checkpoint) with `data/processed/scaler.pkl`.
- **Neutral Baseline Handling**: If a small number of non-core features (such as `Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `act_data_pkt_fwd`, `min_seg_size_forward`) are unavailable, they are set to `scaler.mean_[i]` in raw space so that `scaler.transform` maps them to `0.0` (the neutral mean of the training distribution) rather than raw `0.0` (which would distort to $-\mu_i / \sigma_i$).
- **Signal Weights**: `supervised_weight = 0.85`, `anomaly_weight = 0.05`, `rule_weight = 0.10`.

### Mode 2 — `hybrid` (`HYBRID_ENSEMBLE_DETECTOR`)
- **Selection Criteria**: $0.55 \le C_{\text{eff}} < 0.90$ and $C_{\text{core}} \ge 0.60$ (or 70-feature positional matrices where column names are anonymized).
- **Execution**: Combines `CICIDS2017_SUPERVISED_MODEL` probabilities with `UnsupervisedAnomalyDetector` (`IsolationForest` + Robust Modified Z-Score) and `NetworkRuleEngine`.
- **Signal Weights**: Dynamically scaled by feature coverage:
  $$w_{\text{sup}} \propto 0.55 \cdot C_{\text{eff}} \cdot M_{\text{conf}}, \quad w_{\text{anom}} = 0.25, \quad w_{\text{rule}} = \max(0.15, 1 - w_{\text{sup}} - w_{\text{anom}})$$
  normalized to sum to `1.0`.

### Mode 3 — `anomaly` (`UNSUPERVISED_ANOMALY_DETECTOR`)
- **Selection Criteria**: $C_{\text{eff}} < 0.55$ (e.g., UNSW-NB15, NSL-KDD, TON_IoT, or partial flow tables) with $\ge 2$ numeric columns and $\ge 3$ rows.
- **Execution**: Runs `RobustScaler` (median/IQR) + `IsolationForest(n_estimators=100, random_state=42)` combined with a multivariate Modified Z-Score detector (`0.6745 * |x - median| / MAD`) and `NetworkRuleEngine`.
- **Technical Honesty**: Never labels records as `"Confirmed Attack"`. Flagged flows are categorized as `"Statistical Anomaly"` or `"Suspicious Anomaly + Rule Alert"`, with top deviating features listed in the explanation.

### Mode 4 — `rule_based` (`HEURISTIC_RULE_ENGINE`)
- **Selection Criteria**: Fallback when fewer than 2 numeric features or fewer than 3 rows are available, or when explicitly selected.
- **Execution**: Evaluates 8 configurable evidence-based network rules (`RULE_SYN_FLOOD`, `RULE_HIGH_SYN_ERROR_RATE`, `RULE_PORT_SCAN_PROBE`, `RULE_HIGH_PORT_DIVERSITY`, `RULE_EXCESSIVE_PACKET_RATE`, `RULE_HIGH_BYTE_THROUGHPUT`, `RULE_UNIDIRECTIONAL_FLOOD`, `RULE_SHORT_DURATION_BURST`, `RULE_TCP_RST_SPIKE`, `RULE_HIGH_CONNECTION_FREQUENCY`).
