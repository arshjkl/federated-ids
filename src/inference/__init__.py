"""
Universal Dataset Intelligence & Adaptive Intrusion Detection Package.
"""

from src.inference.canonical_schema import (
    CANONICAL_BY_CICIDS_NAME,
    CANONICAL_BY_ID,
    CANONICAL_FEATURES,
    CANONICAL_GROUPS,
    CICIDS70_FEATURES,
    CICIDS70_NAMES,
    CORE_CICIDS_FEATURES,
    DATASET_FAMILY_SIGNATURES,
    CanonicalFeatureSpec,
)
from src.inference.detectors import (
    AdaptiveHybridScorer,
    NetworkRuleEngine,
    RecordDetectionOutcome,
    SupervisedCICIDSDetector,
    TriggeredRule,
    UnsupervisedAnomalyDetector,
)
from src.inference.feature_derivation import (
    DerivationResult,
    FeatureDerivationEngine,
    FeatureProvenance,
    get_derivation_engine,
)
from src.inference.feature_mapper import (
    FeatureMappingEntry,
    FeatureMappingReport,
    SemanticFeatureMapper,
    get_feature_mapper,
)
from src.inference.ingestion import (
    IngestionMetadata,
    UniversalIngestionEngine,
    get_ingestion_engine,
)
from src.inference.model_router import (
    CompatibilityEvaluation,
    ModelCompatibilityRouter,
    get_model_router,
)
from src.inference.pipeline import (
    AdaptiveInferencePipeline,
    PipelineExecutionReport,
    StageTelemetry,
    get_adaptive_pipeline,
)
from src.inference.profiler import (
    DatasetProfile,
    DatasetProfiler,
    get_dataset_profiler,
)

__all__ = [
    "CANONICAL_BY_CICIDS_NAME",
    "CANONICAL_BY_ID",
    "CANONICAL_FEATURES",
    "CANONICAL_GROUPS",
    "CICIDS70_FEATURES",
    "CICIDS70_NAMES",
    "CORE_CICIDS_FEATURES",
    "DATASET_FAMILY_SIGNATURES",
    "CanonicalFeatureSpec",
    "AdaptiveHybridScorer",
    "NetworkRuleEngine",
    "RecordDetectionOutcome",
    "SupervisedCICIDSDetector",
    "TriggeredRule",
    "UnsupervisedAnomalyDetector",
    "DerivationResult",
    "FeatureDerivationEngine",
    "FeatureProvenance",
    "get_derivation_engine",
    "FeatureMappingEntry",
    "FeatureMappingReport",
    "SemanticFeatureMapper",
    "get_feature_mapper",
    "IngestionMetadata",
    "UniversalIngestionEngine",
    "get_ingestion_engine",
    "CompatibilityEvaluation",
    "ModelCompatibilityRouter",
    "get_model_router",
    "AdaptiveInferencePipeline",
    "PipelineExecutionReport",
    "StageTelemetry",
    "get_adaptive_pipeline",
    "DatasetProfile",
    "DatasetProfiler",
    "get_dataset_profiler",
]
