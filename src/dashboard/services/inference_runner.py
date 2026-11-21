"""
Inference runner service for universal adaptive batch inference on datasets.
Integrates UniversalIngestionEngine, DatasetProfiler, SemanticFeatureMapper,
FeatureDerivationEngine, ModelCompatibilityRouter, Multi-Mode Detectors,
and 18-Stage Traceable AdaptiveInferencePipeline with database storage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.dashboard.db.models import (
    Dataset,
    DetectionResult,
    GlobalModel,
    InferenceJob,
    InferenceStatus,
)
from src.dashboard.services.model_loader import get_model_loader
from src.dashboard.services.preprocessing import get_preprocessing_loader
from src.inference.pipeline import AdaptiveInferencePipeline, get_adaptive_pipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "inference_results"


class InferenceRunner:
    """Runs adaptive multi-mode batch inference jobs with 18-stage telemetry."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_loader = get_model_loader(device)
        self.preprocessing = get_preprocessing_loader()
        self.pipeline: AdaptiveInferencePipeline = get_adaptive_pipeline()

    async def run_inference_job(
        self,
        job_id: UUID,
        db: AsyncSession,
        requested_mode: Optional[str] = None,
        manual_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run a complete 18-stage adaptive inference job."""
        result = await db.execute(select(InferenceJob).where(InferenceJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            raise ValueError(f"Inference job not found: {job_id}")

        job.status = InferenceStatus.RUNNING
        job.started_at = datetime.utcnow()
        await db.commit()

        try:
            dataset_result = await db.execute(select(Dataset).where(Dataset.id == job.dataset_id))
            dataset = dataset_result.scalar_one_or_none()
            if not dataset:
                raise ValueError(f"Dataset not found: {job.dataset_id}")

            model_result = await db.execute(select(GlobalModel).where(GlobalModel.id == job.model_id))
            model = model_result.scalar_one_or_none()
            if not model:
                raise ValueError(f"Model not found: {job.model_id}")

            model_key = self._get_model_key(model.checkpoint_path) or "federated_v2_round_20"
            job_cfg = dict(job.config) if isinstance(job.config, dict) else {}
            threshold = float(job_cfg.get("threshold", 0.5))
            batch_size = int(job_cfg.get("batch_size", 512))
            mode_req = requested_mode or job_cfg.get("detection_mode")
            overrides = manual_overrides or job_cfg.get("manual_overrides")

            # Execute 18-stage adaptive pipeline
            report = self.pipeline.execute(
                file_path=dataset.file_path,
                dataset_id=str(dataset.id),
                model_key=model_key,
                model_name=model.name,
                requested_mode=mode_req,
                manual_overrides=overrides,
                threshold=threshold,
                batch_size=batch_size,
            )

            if report.status == "failed":
                job.status = InferenceStatus.FAILED
                job.error_message = report.error_message or "Pipeline execution failed"
                job.completed_at = datetime.utcnow()
                await db.commit()
                raise ValueError(job.error_message)

            # Load source dataframe for storing compact per-record feature snapshots
            df_raw, _ = self.pipeline.ingestion_engine.load_dataframe(dataset.file_path)

            # Persist full pipeline report JSON to disk
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = RESULTS_DIR / f"job_{job.id}.json"
            with open(report_path, "w") as f:
                json.dump(report.to_dict(include_outcomes=False), f, indent=2)

            # Store detection results in database
            await self._store_adaptive_results(
                job=job,
                df_raw=df_raw,
                outcomes=report.outcomes,
                compatibility=report.compatibility or {},
                db=db,
            )

            # Update job status and summary metadata
            job.status = InferenceStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            if job.started_at:
                job.processing_time = (job.completed_at - job.started_at).total_seconds()
            job.total_records = len(report.outcomes)
            job.malicious_count = int(sum(1 for o in report.outcomes if o.prediction == 1))
            job.benign_count = int(job.total_records - job.malicious_count)
            job.results_path = str(report_path)

            # Save pipeline summary & stage telemetry in job.config alongside threshold/batch_size
            job.config = {
                **job_cfg,
                "threshold": threshold,
                "batch_size": batch_size,
                "detection_mode": report.summary_metrics.get("detection_mode"),
                "model_family": report.summary_metrics.get("model_family"),
                "schema_family": report.summary_metrics.get("schema_family"),
                "compatibility_score": report.summary_metrics.get("compatibility_score"),
                "feature_coverage_score": report.summary_metrics.get("feature_coverage_score"),
                "data_quality_score": report.summary_metrics.get("data_quality_score"),
                "severity_breakdown": report.summary_metrics.get("severity_breakdown"),
                "category_breakdown": report.summary_metrics.get("category_breakdown"),
                "stages_summary": [s.to_dict() for s in report.stages],
                "summary_metrics": report.summary_metrics,
            }
            await db.commit()

            logger.info(
                f"Adaptive inference job {job_id} completed in mode='{report.summary_metrics.get('detection_mode')}': "
                f"{job.benign_count} normal, {job.malicious_count} flagged"
            )

            return {
                "job_id": str(job_id),
                "status": "completed",
                "detection_mode": report.summary_metrics.get("detection_mode"),
                "total_records": job.total_records,
                "benign_count": job.benign_count,
                "malicious_count": job.malicious_count,
                "summary_metrics": report.summary_metrics,
                "stages": [s.to_dict() for s in report.stages],
            }

        except Exception as e:
            logger.error(f"Inference job {job_id} failed: {e}")
            job.status = InferenceStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()
            raise

    def _get_model_key(self, checkpoint_path: str) -> Optional[str]:
        """Map checkpoint path to model key."""
        path = Path(checkpoint_path).name
        mapping = {
            "centralized_mlp.pth": "centralized_baseline",
            "centralized_mlp_improved.pth": "centralized_improved",
            "federated_mlp_manual.pth": "federated_manual",
            "manual_federated_mlp.pth": "federated_manual_v2",
            "federated_mlp_dp.pth": "dp_federated",
            "federated_v2_round_0.pth": "federated_v2_round_0",
            "federated_v2_round_10.pth": "federated_v2_round_10",
            "federated_v2_round_20.pth": "federated_v2_round_20",
        }
        return mapping.get(path)

    async def _store_adaptive_results(
        self,
        job: InferenceJob,
        df_raw: pd.DataFrame,
        outcomes: List[Any],
        compatibility: Dict[str, Any],
        db: AsyncSession,
    ) -> None:
        """
        Store detection results in database with rich Phase 9 metadata and feature values.
        For large datasets (> 5,000 rows), stores full feature snapshots for flagged
        records and the first 500 records to keep SQLite database size bounded.
        """
        n_rows = len(outcomes)
        cols = list(df_raw.columns[:35]) if len(df_raw) == n_rows else []
        comp_score = compatibility.get("compatibility_score", 100.0)
        model_used = compatibility.get("selected_model_name", "CICIDS2017_SUPERVISED_MODEL")

        results: List[DetectionResult] = []
        for idx, outcome in enumerate(outcomes):
            include_raw_row = (n_rows <= 5000) or (outcome.prediction == 1) or (idx < 500)
            row_features: Dict[str, Any] = {}
            if include_raw_row and cols and idx < len(df_raw):
                row_s = df_raw.iloc[idx]
                for c in cols:
                    val = row_s[c]
                    if pd.isna(val):
                        continue
                    elif isinstance(val, (int, np.integer)):
                        row_features[c] = int(val)
                    elif isinstance(val, (float, np.floating)):
                        if np.isfinite(val):
                            row_features[c] = round(float(val), 4)
                    else:
                        row_features[c] = str(val)

            payload_features: Dict[str, Any] = {
                **row_features,
                "__meta__": {
                    "predicted_category": outcome.predicted_category,
                    "supervised_probability": outcome.supervised_probability,
                    "anomaly_score": outcome.anomaly_score,
                    "rule_score": outcome.rule_score,
                    "risk_score": outcome.risk_score,
                    "severity": outcome.severity,
                    "detection_mode": outcome.detection_mode,
                    "triggered_rules": outcome.triggered_rules,
                    "explanation": outcome.explanation,
                    "top_anomaly_features": outcome.top_anomaly_features,
                    "model_used": model_used,
                    "compatibility_score": comp_score,
                },
            }

            primary_prob = (
                outcome.supervised_probability
                if (outcome.detection_mode == "supervised" and outcome.supervised_probability is not None)
                else outcome.risk_score
            )

            results.append(
                DetectionResult(
                    inference_job_id=job.id,
                    record_index=int(outcome.record_index),
                    prediction=int(outcome.prediction),
                    probability=float(primary_prob),
                    confidence=float(outcome.confidence),
                    true_label=outcome.true_label,
                    features=payload_features,
                )
            )

        # Insert in batches of 2000 for memory efficiency on large datasets
        batch_step = 2000
        for start in range(0, len(results), batch_step):
            db.add_all(results[start : start + batch_step])
            await db.commit()


async def run_inference_background(job_id: UUID, db_session_factory):
    """Run inference in background."""
    async with db_session_factory() as db:
        runner = InferenceRunner()
        return await runner.run_inference_job(job_id, db)
