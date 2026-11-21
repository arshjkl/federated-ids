# Semantic Feature Mapping & Canonical Feature Reference

## 1. Canonical Feature Groups

The canonical network-traffic feature schema (`src/inference/canonical_schema.py`) organizes 77 network flow features into 8 dataset-independent groups:

1. **`flow_properties`**: `flow_duration`, `flow_packet_count`, `flow_byte_count`
2. **`directional_properties`**: `total_fwd_packets`, `total_bwd_packets`, `total_fwd_bytes`, `total_bwd_bytes`, `subflow_fwd_packets`, `subflow_fwd_bytes`, `subflow_bwd_packets`, `subflow_bwd_bytes`, `act_data_pkt_fwd`
3. **`packet_statistics`**: `fwd_packet_length_max`, `fwd_packet_length_min`, `fwd_packet_length_mean`, `fwd_packet_length_std`, `bwd_packet_length_max`, `bwd_packet_length_min`, `bwd_packet_length_mean`, `bwd_packet_length_std`, `min_packet_length`, `max_packet_length`, `packet_length_mean`, `packet_length_std`, `packet_length_variance`, `average_packet_size`
4. **`timing`**: `flow_iat_mean`, `flow_iat_std`, `flow_iat_max`, `flow_iat_min`, `fwd_iat_total`, `fwd_iat_mean`, `fwd_iat_std`, `fwd_iat_max`, `fwd_iat_min`, `bwd_iat_total`, `bwd_iat_mean`, `bwd_iat_std`, `bwd_iat_max`, `bwd_iat_min`, `active_mean`, `active_std`, `active_max`, `active_min`, `idle_mean`, `idle_std`, `idle_max`, `idle_min`
5. **`protocol_properties`**: `destination_port`, `source_port`, `protocol`, `fwd_header_length`, `bwd_header_length`, `fwd_header_length_dup`
6. **`tcp_properties`**: `fwd_psh_flags`, `fwd_urg_flags`, `fin_flag_count`, `syn_flag_count`, `rst_flag_count`, `psh_flag_count`, `ack_flag_count`, `urg_flag_count`, `cwe_flag_count`, `ece_flag_count`, `avg_fwd_segment_size`, `avg_bwd_segment_size`, `init_win_bytes_forward`, `init_win_bytes_backward`, `min_seg_size_forward`
7. **`rate_properties`**: `flow_bytes_per_second`, `flow_packets_per_second`, `fwd_packets_per_second`, `bwd_packets_per_second`, `down_up_ratio`
8. **`behavioral_features`**: `connection_frequency`, `failed_connections`, `port_diversity`

---

## 2. Representative Alias Mappings Across Dataset Families

| Canonical ID | CIC-IDS2017 Target | CIC-IDS2018 / CICFlowMeter | UNSW-NB15 / NSL-KDD / TON_IoT | Derivable? |
|---|---|---|---|---|
| `destination_port` | `Destination Port` | `Dst Port` | `dport`, `dst_port`, `id.resp_p` | No |
| `flow_duration` | `Flow Duration` | `Flow Duration` | `dur`, `duration` (auto-scaled if in seconds) | No |
| `total_fwd_packets` | `Total Fwd Packets` | `Tot Fwd Pkts` | `spkts`, `src_pkts`, `orig_pkts` | No |
| `total_bwd_packets` | `Total Backward Packets` | `Tot Bwd Pkts` | `dpkts`, `dst_pkts`, `resp_pkts` | No |
| `total_fwd_bytes` | `Total Length of Fwd Packets` | `TotLen Fwd Pkts` | `sbytes`, `src_bytes`, `orig_bytes` | No |
| `total_bwd_bytes` | `Total Length of Bwd Packets` | `TotLen Bwd Pkts` | `dbytes`, `dst_bytes`, `resp_bytes` | No |
| `flow_bytes_per_second` | `Flow Bytes/s` | `Flow Byts/s` | `bps`, `byte_rate` | Yes (`total_bytes / duration_s`) |
| `flow_packets_per_second` | `Flow Packets/s` | `Flow Pkts/s` | `rate`, `pps`, `pkt_rate` | Yes (`total_pkts / duration_s`) |
| `min_packet_length` | `Min Packet Length` | `Pkt Len Min` | `min_pkt_len` | Yes (`min(fwd_min, bwd_min)`) |
| `max_packet_length` | `Max Packet Length` | `Pkt Len Max` | `max_pkt_len` | Yes (`max(fwd_max, bwd_max)`) |
| `average_packet_size` | `Average Packet Size` | `Pkt Size Avg` | `avg_pkt_size` | Yes (`total_bytes / total_pkts`) |
| `init_win_bytes_forward` | `Init_Win_bytes_forward` | `Init Fwd Win Byts` | `swin` | **No** (Never fabricated) |
| `init_win_bytes_backward` | `Init_Win_bytes_backward` | `Init Bwd Win Byts` | `dwin` | **No** (Never fabricated) |
| `act_data_pkt_fwd` | `act_data_pkt_fwd` | `Fwd Act Data Pkts` | — | **No** (Never fabricated) |
| `min_seg_size_forward` | `min_seg_size_forward` | `Fwd Seg Size Min` | — | **No** (Never fabricated) |

---

## 3. Mapping Classifications

Every feature in `FeatureMappingReport` (`src/inference/feature_mapper.py`) is classified into one of seven categories:
1. **`exact_match`** (Confidence `0.96–1.00`): Exact match or normalized identifier match to the canonical / CIC-IDS2017 feature name.
2. **`strong_semantic_match`** (Confidence `0.80–0.95`): Verified schema alias or high-confidence token-expanded match passing directional, statistical, and unit checks.
3. **`derived_feature`** (Confidence `0.72–0.90`): Computed via exact mathematical formula from mapped parent features (`src/inference/feature_derivation.py`).
4. **`approximate_match`** (Confidence `0.60–0.75`): Positional mapping on verified 70-column standardized matrices (`feature_1..feature_70`) or single-subflow approximations.
5. **`ambiguous_match`** (Confidence `0.55–0.70`): Partial token similarity with multiple candidate columns; flagged for optional manual review.
6. **`unavailable_feature`** (Confidence `0.00`): Feature is absent and cannot be mathematically derived without fabrication.
7. **`incompatible_feature`** (Confidence `0.00`): Candidate column failed data type or statistical plausibility validation (e.g., non-numeric string column or port values outside `[0, 65535]`).
