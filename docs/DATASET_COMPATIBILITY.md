# Dataset Compatibility & Federated Schema Guide

## 1. Supported File Formats

The `UniversalIngestionEngine` (`src/inference/ingestion.py`) ingests network traffic datasets in:

| Format | Extensions | Capabilities |
|---|---|---|
| **CSV** | `.csv`, `.txt` | Auto-detects delimiter (`,`, `;`, `|`, `\t`), encoding (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`), and recovers from malformed rows |
| **TSV** | `.tsv` | Tab-delimited flow logs (including Zeek/Bro and custom TSV exports) |
| **JSON** | `.json` | Array of record objects or `{ "data": [...] }` / `{ "records": [...] }` envelopes |
| **JSON Lines / NDJSON** | `.jsonl`, `.ndjson` | Streaming newline-delimited JSON objects (e.g. Suricata Eve / flow logs) |
| **Parquet** | `.parquet` | Columnar binary format (when `pyarrow` or `fastparquet` is available) |

---

## 2. Supported Network Dataset Families

The `DatasetProfiler` (`src/inference/profiler.py`) and `SemanticFeatureMapper` (`src/inference/feature_mapper.py`) automatically identify and map the following dataset families based on feature structure:

| Dataset Family | Signature Columns | Typical Detection Mode | Notes |
|---|---|---|---|
| **CIC-IDS2017** | `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Flow Bytes/s`, `Fwd Packet Length Mean`, `Init_Win_bytes_forward`, `Subflow Fwd Packets` | **Mode 1: Supervised** (or **Mode 2: Hybrid** if partial) | Native 70-feature schema for `CICIDS2017_SUPERVISED_MODEL`. Even if up to 21 non-core columns are removed or derived, remains eligible for supervised/hybrid detection. |
| **CIC-IDS2018 / CIC-DDoS2019** | `Dst Port`, `Tot Fwd Pkts`, `Tot Bwd Pkts`, `TotLen Fwd Pkts`, `Fwd Pkt Len Max`, `Flow Byts/s`, `Flow Pkts/s`, `Init Fwd Win Byts` | **Mode 1: Supervised** or **Mode 2: Hybrid** | Abbreviated CICFlowMeter v4 column names map directly to the 70 CIC-IDS2017 features with 100% alias confidence. |
| **UNSW-NB15** | `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sinpkt`, `dinpkt` | **Mode 2: Hybrid** or **Mode 3: Anomaly + Rule** | Duration in seconds (`dur`) is automatically converted to microseconds (`seconds_to_microseconds`), and directional packets/bytes/rates/means are mapped and derived. |
| **NSL-KDD / KDD99** | `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `count`, `srv_count`, `serror_rate`, `same_srv_rate` | **Mode 3: Anomaly + Rule** | Maps `duration` (with seconds-to-microseconds scaling), `src_bytes` -> `total_fwd_bytes`, `dst_bytes` -> `total_bwd_bytes`, `count` -> `connection_frequency`, and `serror_rate` -> `failed_connection_rate`. |
| **TON_IoT** | `src_ip`, `src_port`, `dst_ip`, `dst_port`, `proto`, `duration`, `src_bytes`, `dst_bytes`, `src_pkts`, `dst_pkts`, `conn_state` | **Mode 2: Hybrid** or **Mode 3: Anomaly + Rule** | Maps directional packets, bytes, ports, protocol, and duration; derives total packets, bytes, average packet sizes, and packet/byte rates. |
| **Edge-IIoTset** | `tcp.srcport`, `tcp.dstport`, `tcp.len`, `tcp.flags.syn`, `tcp.flags.ack`, `tcp.flags.reset`, `tcp.window_size_value` | **Mode 3: Anomaly + Rule** or **Mode 4: Rule-Based** | Maps packet-level TCP flags, ports, lengths, and window sizes into canonical protocol/TCP properties. |
| **Custom / Generic Flow Exports** | Any CSV/JSON containing subset of duration, packets, bytes, ports, or flags | **Mode 2 / 3 / 4** depending on coverage | Never rejected; evaluated transparently according to available canonical features. |

---

## 3. Handling the User's Missing Feature Examples

When a user uploads a dataset missing specific CIC-IDS2017 columns such as:
- `Min Packet Length`
- `Max Packet Length`
- `Init_Win_bytes_forward`
- `Init_Win_bytes_backward`
- `act_data_pkt_fwd`
- `min_seg_size_forward`

The pipeline handles them responsibly:
1. **`Min Packet Length` & `Max Packet Length`**: Mathematically derived by `FeatureDerivationEngine` from `min(Fwd Packet Length Min, Bwd Packet Length Min)` and `max(Fwd Packet Length Max, Bwd Packet Length Max)` when directional bounds exist, or set to `Packet Length Mean` when packet variance is zero. Marked with provenance `derived`.
2. **`Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `act_data_pkt_fwd`, `min_seg_size_forward`**: Recognized as non-derivable TCP header/segment measurements. They are **never** fabricated or filled with raw `0.0`. Instead:
   - With 66/70 features available (94.3% effective coverage, 100% core coverage), the dataset still qualifies for `CICIDS2017_SUPERVISED_MODEL`.
   - When preparing the 70-feature input vector for `StandardScaler`, those 4 unavailable non-core features are set to their training mean (`scaler.mean_[idx]`) so that after `(x - mean) / scale` transformation they become `0.0` (neutral z-score) rather than an artificial $-\mu / \sigma$ spike.

---

## 4. Federated Learning Schema & Model Compatibility (`src/federated/schema_validator.py`)

In federated learning across heterogeneous clients (`FedAvg` / `FedProx`):
- **`FederatedCompatibilityValidator.validate_client_dataset(columns, client_id)`**:
  - Checks each federated client's local dataset via `SemanticFeatureMapper`.
  - Computes effective coverage after safe derivations (`FeatureDerivationEngine.DERIVATION_SPECS`).
  - Determines whether the client can participate in training the 70-feature global model (`v1.0-cicids70`, requiring $\ge 70\%$ effective coverage and $\ge 65\%$ core coverage) or the compact 22-core-feature canonical federated model (`v1.0-canonical-core22`, requiring $\ge 65\%$ core coverage).
- **`FederatedCompatibilityValidator.validate_client_parameters(global_params, client_params, client_id)`**:
  - Verifies layer count, exact tensor shapes, and parameter finiteness (`np.all(np.isfinite(arr))`) before allowing client weights into `FedAvg` or `FedProx` aggregation, preventing shape mismatches or `NaN` poisoning.
