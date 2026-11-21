"""
Phase 13, 14 & 15: Comprehensive Unit, Integration, Dataset-Matrix, and
Large-Dataset Performance Tests for Universal Dataset Intelligence & Adaptive IDS.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from httpx import ASGITransport, AsyncClient

from src.dashboard.main import app
from src.federated.schema_validator import FederatedCompatibilityValidator
from src.inference import (
    CICIDS70_NAMES,
    get_adaptive_pipeline,
    get_dataset_profiler,
    get_derivation_engine,
    get_feature_mapper,
    get_ingestion_engine,
    get_model_router,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase2UniversalIngestion(unittest.TestCase):
    """Phase 14 — Ingestion tests across valid, empty, malformed, duplicate, mixed, missing, inf, and formats."""

    def setUp(self):
        self.engine = get_ingestion_engine()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_temp(self, name: str, content: str) -> Path:
        p = Path(self.temp_dir.name) / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_csv_ingestion(self):
        sample_path = PROJECT_ROOT / "data" / "mock" / "inference" / "mock_inference_batch_01.csv"
        df, meta = self.engine.load_dataframe(sample_path)
        self.assertEqual(meta.detected_format, "csv")
        self.assertEqual(meta.total_rows, 833)
        self.assertEqual(meta.total_columns, 70)
        self.assertFalse(meta.is_labeled)
        self.assertEqual(len(df), 833)

    def test_empty_csv_ingestion(self):
        p = self._write_temp("empty.csv", "")
        df, meta = self.engine.load_dataframe(p)
        self.assertEqual(meta.detected_format, "empty")
        self.assertEqual(meta.total_rows, 0)
        self.assertEqual(len(df), 0)
        self.assertTrue(len(meta.errors) > 0)

    def test_malformed_csv_recovery(self):
        content = (
            "Destination Port,Flow Duration,Total Fwd Packets,Total Backward Packets\n"
            "80,1500,10,8\n"
            "443,2200,12,10,EXTRA_COL_1,EXTRA_COL_2\n"
            "22,900,5,4\n"
        )
        p = self._write_temp("malformed.csv", content)
        df, meta = self.engine.load_dataframe(p)
        self.assertEqual(len(df), 2)
        self.assertEqual(meta.malformed_rows_skipped, 1)

    def test_duplicate_columns_disambiguation(self):
        content = "Dst Port,Fwd Header Length,Fwd Header Length,Flow Duration\n80,20,20,1000\n443,32,32,2000\n"
        p = self._write_temp("dup_cols.csv", content)
        df, meta = self.engine.load_dataframe(p)
        self.assertIn("Fwd Header Length", df.columns)
        self.assertIn("Fwd Header Length.1", df.columns)
        self.assertIn("Fwd Header Length.1", meta.duplicate_columns)

    def test_mixed_types_missing_and_infinite_values(self):
        content = (
            "Flow ID,Dst Port,Flow Duration,Total Fwd Packets,Label\n"
            "flow-1,80,1000,10,BENIGN\n"
            "flow-2,443,Infinity,20,DDoS\n"
            "flow-3,22,,15,BENIGN\n"
            "flow-4,8080,bad_num,12,PortScan\n"
        )
        p = self._write_temp("mixed.csv", content)
        df, meta = self.engine.load_dataframe(p)
        self.assertTrue(meta.is_labeled)
        self.assertEqual(meta.primary_label_column, "Label")
        self.assertIn("Flow ID", meta.identifier_columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["Flow Duration"]))

    def test_tsv_and_jsonl_ingestion(self):
        tsv_p = self._write_temp("sample.tsv", "Dst Port\tFlow Duration\tTotal Fwd Packets\n80\t1200\t6\n53\t400\t2\n")
        df_tsv, meta_tsv = self.engine.load_dataframe(tsv_p)
        self.assertEqual(meta_tsv.detected_format, "tsv")
        self.assertEqual(len(df_tsv), 2)

        jsonl_p = self._write_temp(
            "sample.jsonl",
            '{"Dst Port": 80, "Flow Duration": 500, "Total Fwd Packets": 4}\n'
            '{"Dst Port": 443, "Flow Duration": 1500, "Total Fwd Packets": 9}\n',
        )
        df_j, meta_j = self.engine.load_dataframe(jsonl_p)
        self.assertEqual(meta_j.detected_format, "jsonl")
        self.assertEqual(len(df_j), 2)


class TestPhase4And6FeatureMappingAndDerivation(unittest.TestCase):
    """Phase 14 — Semantic feature mapping, aliases, unit validation, and safe derivations."""

    def setUp(self):
        self.mapper = get_feature_mapper()
        self.derivation = get_derivation_engine()
        self.profiler = get_dataset_profiler()

    def test_exact_cicids2017_mapping(self):
        sample_path = PROJECT_ROOT / "data" / "mock" / "inference" / "mock_inference_batch_01.csv"
        df, meta, profile = self.profiler.profile_file(str(sample_path))
        report = self.mapper.map_features(df, meta.column_types, profile.detected_schema_family)
        self.assertEqual(profile.detected_schema_family, "CICIDS2017")
        self.assertGreaterEqual(report.mapped_cicids_count, 68)
        self.assertGreaterEqual(report.effective_coverage_ratio, 0.97)

    def test_alias_and_case_insensitive_mapping(self):
        df = pd.DataFrame({
            "Dst Port": [80, 443, 22],
            "dur": [0.5, 1.2, 0.05],  # duration in seconds!
            "Tot Fwd Pkts": [10, 20, 4],
            "Tot Bwd Pkts": [8, 18, 0],
            "sbytes": [1200, 3400, 240],
            "dbytes": [4000, 12000, 0],
            "Fwd Pkt Len Max": [200, 300, 60],
            "Fwd Pkt Len Min": [60, 60, 60],
            "Bwd Pkt Len Max": [1460, 1460, 0],
            "Bwd Pkt Len Min": [60, 60, 0],
            "SYN Flag Cnt": [1, 1, 3],
            "ACK Flag Cnt": [1, 1, 0],
        })
        col_types = {c: "numeric" for c in df.columns}
        report = self.mapper.map_features(df, col_types)
        by_id = {m.canonical_id: m for m in report.mappings}

        self.assertEqual(by_id["destination_port"].source_column, "Dst Port")
        self.assertEqual(by_id["flow_duration"].source_column, "dur")
        self.assertIn("seconds_to_microseconds", by_id["flow_duration"].unit_conversion or "")
        self.assertEqual(by_id["total_fwd_packets"].source_column, "Tot Fwd Pkts")
        self.assertEqual(by_id["total_fwd_bytes"].source_column, "sbytes")

        # Verify mathematical derivations
        deriv = self.derivation.derive_features(df, report)
        self.assertIn("min_packet_length", deriv.derived_features_list)
        self.assertIn("max_packet_length", deriv.derived_features_list)
        self.assertIn("average_packet_size", deriv.derived_features_list)
        self.assertIn("flow_bytes_per_second", deriv.derived_features_list)
        self.assertIn("flow_packets_per_second", deriv.derived_features_list)
        # Verify non-derivable TCP window features are NOT fabricated
        self.assertNotIn("init_win_bytes_forward", deriv.available_canonical_features)
        self.assertEqual(deriv.provenance["init_win_bytes_forward"].provenance_type, "missing")
        # Verify seconds -> microseconds conversion was applied (0.5s -> 500,000 us)
        self.assertAlmostEqual(float(deriv.canonical_df["flow_duration"].iloc[0]), 500000.0)

    def test_incompatible_port_and_text_rejected(self):
        df = pd.DataFrame({
            "Destination Port": [999999, 888888, 777777],  # Invalid port > 65535
            "Flow Duration": ["not_a_duration", "invalid", "text"],
            "Total Fwd Packets": [10, 20, 30],
        })
        col_types = {
            "Destination Port": "numeric",
            "Flow Duration": "text",
            "Total Fwd Packets": "numeric",
        }
        report = self.mapper.map_features(df, col_types)
        by_id = {m.canonical_id: m for m in report.mappings}
        self.assertEqual(by_id["destination_port"].mapping_type, "incompatible_feature")
        self.assertEqual(by_id["flow_duration"].mapping_type, "incompatible_feature")
        self.assertEqual(by_id["total_fwd_packets"].mapping_type, "exact_match")


class TestPhase7And8ModelRoutingAndPipeline(unittest.TestCase):
    """Phase 14 & 15 — 10-Dataset Scenario Matrix & 18-Stage Pipeline Execution Tests."""

    def setUp(self):
        self.pipeline = get_adaptive_pipeline()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_df(self, name: str, df: pd.DataFrame) -> Path:
        p = Path(self.temp_dir.name) / name
        df.to_csv(p, index=False)
        return p

    def test_10_dataset_scenarios_end_to_end(self):
        """Phase 15: Execute and verify all 10 required dataset scenarios."""
        base_csv = PROJECT_ROOT / "data" / "mock" / "inference" / "mock_inference_batch_01.csv"
        base_df = pd.read_csv(base_csv).head(120)

        scenarios = []

        # 1. Existing CICIDS-compatible labeled dataset
        s1_path = PROJECT_ROOT / "data" / "mock" / "inference" / "mock_cic_ids_5000_labelled.csv"
        scenarios.append(("1_cicids_labeled", s1_path, "supervised"))

        # 2. Unlabeled dataset with matching features
        scenarios.append(("2_cicids_unlabeled", base_csv, "supervised"))

        # 3. Dataset with renamed CICFlowMeter columns
        renamed_df = base_df.rename(columns={
            "Destination Port": "Dst Port",
            "Total Fwd Packets": "Tot Fwd Pkts",
            "Total Backward Packets": "Tot Bwd Pkts",
            "Total Length of Fwd Packets": "TotLen Fwd Pkts",
            "Total Length of Bwd Packets": "TotLen Bwd Pkts",
            "Flow Bytes/s": "Flow Byts/s",
            "Flow Packets/s": "Flow Pkts/s",
        })
        s3_path = self._save_df("3_renamed_cols.csv", renamed_df)
        scenarios.append(("3_renamed_columns", s3_path, "supervised"))

        # 4. Dataset with missing TCP & packet-length features (the user's exact examples!)
        missing_tcp_df = base_df.drop(columns=[
            "Min Packet Length",
            "Max Packet Length",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
        ], errors="ignore")
        s4_path = self._save_df("4_missing_tcp.csv", missing_tcp_df)
        scenarios.append(("4_missing_tcp_features", s4_path, "supervised"))

        # 5. Dataset with a different network-flow schema (UNSW-NB15 style)
        rng = np.random.default_rng(42)
        unsw_df = pd.DataFrame({
            "dur": rng.exponential(0.5, 100),
            "proto": ["tcp"] * 80 + ["udp"] * 20,
            "spkts": rng.integers(1, 100, 100),
            "dpkts": rng.integers(0, 80, 100),
            "sbytes": rng.integers(60, 50000, 100),
            "dbytes": rng.integers(0, 80000, 100),
            "rate": rng.uniform(10, 8000, 100),
            "sttl": rng.choice([64, 128, 254], 100),
            "dttl": rng.choice([64, 128, 254], 100),
            "sinpkt": rng.uniform(0.1, 50, 100),
            "dinpkt": rng.uniform(0.1, 50, 100),
            "sjit": rng.uniform(0, 20, 100),
            "djit": rng.uniform(0, 20, 100),
            "swin": rng.choice([255, 8192, 65535], 100),
            "dwin": rng.choice([0, 8192, 65535], 100),
            "smean": rng.uniform(40, 1200, 100),
            "dmean": rng.uniform(0, 1400, 100),
            "ct_srv_src": rng.integers(1, 30, 100),
            "ct_dst_ltm": rng.integers(1, 25, 100),
            "ct_src_dport_ltm": rng.integers(1, 20, 100),
        })
        s5_path = self._save_df("5_unsw_nb15.csv", unsw_df)
        scenarios.append(("5_unsw_nb15_schema", s5_path, "anomaly"))

        # 6. Dataset with only partial network features (8 core columns)
        partial_df = pd.DataFrame({
            "Dst Port": rng.choice([22, 53, 80, 443, 8080], 80),
            "Flow Duration": rng.integers(50, 500000, 80),
            "Total Fwd Packets": rng.integers(1, 200, 80),
            "Total Backward Packets": rng.integers(0, 100, 80),
            "Total Length of Fwd Packets": rng.integers(60, 100000, 80),
            "Total Length of Bwd Packets": rng.integers(0, 100000, 80),
            "SYN Flag Count": rng.choice([0, 1, 5], 80),
            "ACK Flag Count": rng.choice([0, 1, 10], 80),
        })
        s6_path = self._save_df("6_partial_flow.csv", partial_df)
        scenarios.append(("6_partial_network_features", s6_path, "anomaly"))

        # 7. Dataset containing extra unrelated metadata columns
        extra_df = base_df.copy()
        extra_df["Sensor_Hostname"] = "edge-sensor-01"
        extra_df["Geo_Country"] = "US"
        extra_df["Customer_Tier"] = "Enterprise"
        s7_path = self._save_df("7_extra_cols.csv", extra_df)
        scenarios.append(("7_extra_unrelated_columns", s7_path, "supervised"))

        # 8. Dataset with mixed types, NaN, and Infinity
        mixed_df = base_df.copy()
        mixed_df.loc[0, "Flow Bytes/s"] = np.inf
        mixed_df.loc[1, "Flow Packets/s"] = np.nan
        mixed_df["Flow_ID"] = [f"flow_{i}" for i in range(len(mixed_df))]
        s8_path = self._save_df("8_mixed_nan_inf.csv", mixed_df)
        scenarios.append(("8_mixed_types_nan_inf", s8_path, "supervised"))

        # 9. Dataset with invalid/corrupted rows mixed with valid rows
        s9_path = Path(self.temp_dir.name) / "9_invalid_rows.csv"
        valid_lines = base_df.head(40).to_csv(index=False).splitlines()
        valid_lines.insert(5, "corrupted,row,with,too,many," + ",".join(["1"] * 90))
        s9_path.write_text("\n".join(valid_lines), encoding="utf-8")
        scenarios.append(("9_invalid_rows", s9_path, "supervised"))

        # 10. Anonymized 70-feature dataset (test_inference.csv)
        s10_path = PROJECT_ROOT / "test_inference.csv"
        scenarios.append(("10_anonymized_70_features", s10_path, "hybrid"))

        for name, path, expected_mode in scenarios:
            report = self.pipeline.execute(str(path), dataset_id=name)
            self.assertIn(report.status, ("completed", "warning"), f"Scenario {name} failed: {report.error_message}")
            self.assertEqual(len(report.stages), 18)
            self.assertEqual(report.summary_metrics["detection_mode"], expected_mode, f"Scenario {name} mode mismatch")
            self.assertGreater(len(report.outcomes), 0)
            # In anomaly mode, never claim Confirmed Attack
            if expected_mode == "anomaly":
                for o in report.outcomes:
                    self.assertNotIn("Confirmed Attack", o.predicted_category)

    def test_performance_scaling_benchmarks(self):
        """Phase 13: Test pipeline across 100, 1,000, 10,000, and 25,000 records."""
        base_csv = PROJECT_ROOT / "data" / "mock" / "inference" / "mock_cic_ids_5000.csv"
        base_df = pd.read_csv(base_csv)

        for target_rows in (100, 1000, 10000, 25000):
            if target_rows <= len(base_df):
                df_test = base_df.iloc[:target_rows].copy()
            else:
                repeats = (target_rows // len(base_df)) + 1
                df_test = pd.concat([base_df] * repeats, ignore_index=True).iloc[:target_rows]
            p = self._save_df(f"scale_{target_rows}.csv", df_test)
            t0 = time.perf_counter()
            report = self.pipeline.execute(str(p), dataset_id=f"bench_{target_rows}")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertIn(report.status, ("completed", "warning"))
            self.assertEqual(report.summary_metrics["total_records"], target_rows)
            self.assertLess(elapsed_ms, 30000.0)


class TestPhase11FederatedCompatibility(unittest.TestCase):
    """Phase 11 — Federated schema & parameter aggregation compatibility validator tests."""

    def test_federated_schema_validator(self):
        validator = FederatedCompatibilityValidator(target_version="v1.0-cicids70")
        cic_df = pd.read_csv(PROJECT_ROOT / "data" / "mock" / "inference" / "mock_inference_batch_01.csv").head(20)
        rep_ok = validator.validate_client_dataset("client_1", cic_df)
        self.assertTrue(rep_ok.is_compatible)

        incompat_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        rep_bad = validator.validate_client_dataset("client_2", incompat_df)
        self.assertFalse(rep_bad.is_compatible)
        self.assertGreater(len(rep_bad.errors), 0)

    def test_federated_parameter_aggregation_validator(self):
        ref = [np.zeros((128, 70), dtype=np.float32), np.zeros((128,), dtype=np.float32)]
        c1 = ("client_1", [np.ones((128, 70), dtype=np.float32), np.ones((128,), dtype=np.float32)])
        c2_bad = ("client_2", [np.ones((128, 35), dtype=np.float32), np.ones((128,), dtype=np.float32)])

        ok, errs = FederatedCompatibilityValidator.validate_parameter_aggregation(ref, [c1])
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

        ok2, errs2 = FederatedCompatibilityValidator.validate_parameter_aggregation(ref, [c1, c2_bad])
        self.assertFalse(ok2)
        self.assertGreater(len(errs2), 0)


class TestPhase12BackendAPI(unittest.TestCase):
    """Phase 12 & 14 — FastAPI endpoint integration tests."""

    def test_api_end_to_end_partial_dataset_upload(self):
        async def _run():
            async with app.router.lifespan_context(app):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    # Upload a dataset missing TCP window features
                    base_df = pd.read_csv(
                        PROJECT_ROOT / "data" / "mock" / "inference" / "mock_inference_batch_01.csv"
                    ).head(50)
                    partial_df = base_df.drop(columns=[
                        "Min Packet Length", "Max Packet Length",
                        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                        "act_data_pkt_fwd", "min_seg_size_forward",
                    ])
                    csv_bytes = partial_df.to_csv(index=False).encode("utf-8")

                    r_an = await client.post(
                        "/api/inference/analyze-csv",
                        files={"file": ("partial_no_tcp.csv", io.BytesIO(csv_bytes), "text/csv")},
                    )
                    self.assertEqual(r_an.status_code, 200, r_an.text)
                    ds_id = r_an.json()["dataset_id"]

                    r_val = await client.get(f"/api/inference/validate/{ds_id}")
                    self.assertEqual(r_val.status_code, 200)
                    self.assertTrue(r_val.json()["is_valid"])

                    r_map = await client.get(f"/api/inference/feature-mapping/{ds_id}")
                    self.assertEqual(r_map.status_code, 200)
                    # Min Packet Length and Max Packet Length should be derived automatically!
                    derived_list = r_map.json()["derivation_summary"]["derived_features_list"]
                    self.assertIn("min_packet_length", derived_list)
                    self.assertIn("max_packet_length", derived_list)

                    r_run = await client.post(
                        "/api/inference/run-on-dataset",
                        data={"dataset_id": ds_id, "name": "Partial TCP Test"},
                    )
                    self.assertEqual(r_run.status_code, 200, r_run.text)
                    job_id = r_run.json()["id"]

                    r_stages = await client.get(f"/api/inference/{job_id}/stages")
                    self.assertEqual(r_stages.status_code, 200)
                    self.assertEqual(len(r_stages.json()["stages"]), 18)

                    r_rec = await client.get(f"/api/inference/{job_id}/records/0")
                    self.assertEqual(r_rec.status_code, 200)
                    self.assertIn("predicted_category", r_rec.json())

        asyncio.run(_run())

    def test_cybersecurity_session_intrusion_dataset(self):
        """Verify session & behavioral cybersecurity datasets detect brute-force, IP reputation, and client agent attacks."""
        pipe = get_adaptive_pipeline()
        df = pd.DataFrame({
            "session_id": [f"SID_{i:04d}" for i in range(1, 21)],
            "network_packet_size": [500, 620, 480, 510, 390, 550, 720, 460, 530, 610] * 2,
            "protocol_type": ["TCP", "UDP", "TCP", "ICMP", "TCP"] * 4,
            "login_attempts": [2, 3, 4, 8, 9, 2, 3, 10, 4, 1, 3, 2, 7, 4, 2, 3, 4, 2, 1, 5],
            "session_duration": [300.5, 450.2, 120.0, 980.1, 600.4] * 4,
            "encryption_used": ["AES", "DES", "AES", "None", "DES"] * 4,
            "ip_reputation_score": [0.15, 0.22, 0.18, 0.72, 0.65, 0.12, 0.28, 0.81, 0.64, 0.10] * 2,
            "failed_logins": [0, 1, 1, 4, 3, 0, 1, 5, 3, 0] * 2,
            "browser_type": ["Chrome", "Firefox", "Edge", "Unknown", "Safari"] * 4,
            "unusual_time_access": [0, 0, 0, 1, 1, 0, 0, 1, 0, 0] * 2,
            "attack_detected": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0] * 2,
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            tmp_path = f.name
        try:
            rep = pipe.execute(tmp_path)
            sm = rep.summary_metrics
            self.assertEqual(sm["schema_family"], "CYBERSECURITY_INTRUSION_SESSION")
            self.assertGreater(sm["suspicious_or_malicious_count"], 0)
            self.assertGreaterEqual(sm["ground_truth_evaluation"]["f1"], 0.85)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
