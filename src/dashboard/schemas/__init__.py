"""
Pydantic schemas for the Federated Learning IDS Platform API.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


# Enums
class OrganisationStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    PAUSED = "paused"


class DatasetStatus(str, Enum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    PROCESSING = "processing"
    READY = "ready"


class ExperimentStatus(str, Enum):
    CREATED = "created"
    CONFIGURING = "configuring"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingAlgorithm(str, Enum):
    CENTRALIZED = "centralized"
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    DP_FEDAVG = "dp_fedavg"
    DP_FEDPROX = "dp_fedprox"


class ClientStatus(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class GlobalModelStatus(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    ARCHIVED = "archived"
    FAILED = "failed"


class InferenceStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Base schemas with ORM mode
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# Organisation schemas
class OrganisationBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    org_id: str = Field(..., min_length=1, max_length=100)


class OrganisationCreate(OrganisationBase):
    dataset_id: Optional[UUID] = None


class OrganisationUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[OrganisationStatus] = None
    dataset_id: Optional[UUID] = None
    local_model_path: Optional[str] = None


class OrganisationResponse(OrganisationBase):
    id: UUID
    status: OrganisationStatus
    dataset_id: Optional[UUID] = None
    local_model_path: Optional[str] = None
    training_rounds_participated: int
    last_training_round: Optional[int] = None
    total_samples: int
    created_at: datetime
    updated_at: datetime


class OrganisationDetail(OrganisationResponse):
    dataset: Optional["DatasetResponse"] = None


# Dataset schemas
class DatasetBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class DatasetCreate(DatasetBase):
    filename: str
    file_path: str
    file_size: int
    uploaded_by: Optional[str] = None


class DatasetUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[DatasetStatus] = None
    assigned_org_id: Optional[UUID] = None


class DatasetResponse(DatasetBase):
    id: UUID
    filename: str
    file_path: str
    file_size: int
    status: DatasetStatus
    rows: Optional[int] = None
    columns: Optional[int] = None
    feature_names: Optional[List[str]] = None
    label_distribution: Optional[Dict[str, int]] = None
    missing_values: Optional[Dict[str, int]] = None
    duplicate_rows: Optional[int] = None
    assigned_org_id: Optional[UUID] = None
    uploaded_by: Optional[str] = None
    created_at: datetime
    validated_at: Optional[datetime] = None


class DatasetValidationResponse(BaseSchema):
    is_valid: bool
    rows: int
    columns: int
    feature_names: List[str]
    label_distribution: Dict[str, int]
    missing_values: Dict[str, int]
    duplicate_rows: int
    constant_features: List[str]
    warnings: List[str] = []


# Experiment schemas
class ExperimentConfig(BaseSchema):
    model_config = ConfigDict(from_attributes=True, extra="allow")
    local_epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=128, ge=16, le=2048)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    max_samples_per_client: int = Field(default=100000, ge=1000)
    fedprox_mu: float = Field(default=0.01, ge=0, le=1)
    dp_noise_multiplier: float = Field(default=1.0, ge=0)
    dp_max_grad_norm: float = Field(default=1.0, gt=0)
    dp_delta: float = Field(default=1e-5, gt=0, le=1)
    secure_aggregation: bool = False
    client_sampling_fraction: float = Field(default=1.0, gt=0, le=1)
    checkpoint_interval: int = Field(default=10, ge=1)
    dataset_id: Optional[UUID] = None
    sample_file: Optional[str] = None
    target_column: Optional[str] = None
    trained_model_id: Optional[str] = None
    trained_model_name: Optional[str] = None


class ExperimentBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    algorithm: TrainingAlgorithm
    total_rounds: int = Field(..., ge=1, le=1000)
    config: ExperimentConfig


class ExperimentCreate(ExperimentBase):
    participating_org_ids: List[UUID] = Field(..., min_length=1)
    dataset_id: Optional[UUID] = None
    sample_file: Optional[str] = None
    target_column: Optional[str] = None


class ExperimentUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[ExperimentStatus] = None
    current_round: Optional[int] = None


class ExperimentResponse(ExperimentBase):
    id: UUID
    status: ExperimentStatus
    global_model_id: Optional[UUID] = None
    current_round: int
    participating_orgs: Optional[List[UUID]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    updated_at: datetime


class ExperimentDetail(ExperimentResponse):
    global_model: Optional["GlobalModelResponse"] = None
    rounds: List["TrainingRoundResponse"] = []


# Training Round schemas
class TrainingRoundBase(BaseSchema):
    round_number: int
    status: str = "pending"


class TrainingRoundResponse(TrainingRoundBase):
    id: UUID
    experiment_id: UUID
    global_loss: Optional[float] = None
    global_accuracy: Optional[float] = None
    global_f1: Optional[float] = None
    global_precision: Optional[float] = None
    global_recall: Optional[float] = None
    aggregation_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    client_metrics: List["ClientRoundMetricsResponse"] = []


# Client Participation schemas
class ClientParticipationResponse(BaseSchema):
    id: UUID
    experiment_id: UUID
    organisation_id: UUID
    is_active: bool
    joined_at: datetime
    last_round: int


# Client Round Metrics schemas
class ClientRoundMetricsBase(BaseSchema):
    local_loss: Optional[float] = None
    local_accuracy: Optional[float] = None
    local_f1: Optional[float] = None
    local_precision: Optional[float] = None
    local_recall: Optional[float] = None
    training_time: Optional[float] = None
    samples_used: Optional[int] = None


class ClientRoundMetricsResponse(ClientRoundMetricsBase):
    id: UUID
    round_id: UUID
    participation_id: UUID
    organisation_id: UUID
    completed_at: Optional[datetime] = None


# Global Model schemas
class GlobalModelBase(BaseSchema):
    version: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    algorithm: TrainingAlgorithm
    architecture: str = Field(..., min_length=1, max_length=100)
    architecture_config: Optional[Dict[str, Any]] = None
    checkpoint_path: str
    training_rounds: int = Field(..., ge=0)
    participating_clients: int = Field(..., ge=0)


class GlobalModelCreate(GlobalModelBase):
    experiment_id: Optional[UUID] = None
    parent_model_id: Optional[UUID] = None
    total_samples: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    privacy_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None


class GlobalModelUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[GlobalModelStatus] = None
    metrics: Optional[Dict[str, Any]] = None


class GlobalModelResponse(GlobalModelBase):
    id: UUID
    experiment_id: Optional[UUID] = None
    parent_model_id: Optional[UUID] = None
    status: GlobalModelStatus
    total_samples: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    privacy_config: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    trained_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    created_at: datetime


class GlobalModelDetail(GlobalModelResponse):
    experiment: Optional[ExperimentResponse] = None
    inference_jobs: List["InferenceJobResponse"] = []


# Inference Job schemas
class InferenceConfig(BaseSchema):
    model_config = ConfigDict(from_attributes=True, extra="allow")
    threshold: float = Field(default=0.5, ge=0, le=1)
    batch_size: int = Field(default=256, ge=1, le=4096)
    return_probabilities: bool = True
    detection_mode: Optional[str] = None
    model_family: Optional[str] = None
    schema_family: Optional[str] = None
    compatibility_score: Optional[float] = None
    feature_coverage_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    severity_breakdown: Optional[Dict[str, int]] = None
    category_breakdown: Optional[Dict[str, int]] = None
    stages_summary: Optional[List[Dict[str, Any]]] = None
    summary_metrics: Optional[Dict[str, Any]] = None
    manual_overrides: Optional[Dict[str, str]] = None


class InferenceJobBase(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)


class InferenceJobCreate(InferenceJobBase):
    model_id: UUID
    dataset_id: UUID
    organisation_id: Optional[UUID] = None
    config: Optional[InferenceConfig] = None


class InferenceJobUpdate(BaseSchema):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[InferenceStatus] = None


class InferenceJobResponse(InferenceJobBase):
    id: UUID
    model_id: UUID
    dataset_id: UUID
    organisation_id: Optional[UUID] = None
    status: InferenceStatus
    config: Optional[Dict[str, Any]] = None
    total_records: Optional[int] = None
    benign_count: Optional[int] = None
    malicious_count: Optional[int] = None
    processing_time: Optional[float] = None
    results_path: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class InferenceJobDetail(InferenceJobResponse):
    model: Optional[GlobalModelResponse] = None
    dataset: Optional[DatasetResponse] = None


# Detection Result schemas
class DetectionResultResponse(BaseSchema):
    id: UUID
    inference_job_id: UUID
    record_index: int
    prediction: int
    probability: float
    confidence: Optional[float] = None
    true_label: Optional[int] = None
    features: Optional[Dict[str, Any]] = None
    timestamp: datetime
    predicted_category: Optional[str] = None
    supervised_probability: Optional[float] = None
    anomaly_score: Optional[float] = None
    rule_score: Optional[float] = None
    risk_score: Optional[float] = None
    severity: Optional[str] = None
    detection_mode: Optional[str] = None
    triggered_rules: Optional[List[Dict[str, Any]]] = None
    explanation: Optional[str] = None
    top_anomaly_features: Optional[List[str]] = None
    model_used: Optional[str] = None
    compatibility_score: Optional[float] = None


# Detection History schemas
class DetectionHistoryResponse(BaseSchema):
    id: UUID
    organisation_id: UUID
    inference_job_id: Optional[UUID] = None
    timestamp: datetime
    total_analyzed: int
    benign_count: int
    malicious_count: int
    attack_types: Optional[Dict[str, int]] = None
    model_version: Optional[str] = None


# Training History schemas
class TrainingHistoryResponse(BaseSchema):
    id: UUID
    organisation_id: UUID
    experiment_id: Optional[UUID] = None
    round_number: int
    local_loss: Optional[float] = None
    local_accuracy: Optional[float] = None
    local_f1: Optional[float] = None
    local_precision: Optional[float] = None
    local_recall: Optional[float] = None
    training_time: Optional[float] = None
    samples_used: Optional[int] = None
    completed_at: datetime


# Privacy Record schemas
class PrivacyRecordResponse(BaseSchema):
    id: UUID
    experiment_id: UUID
    global_model_id: Optional[UUID] = None
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    noise_multiplier: Optional[float] = None
    max_grad_norm: Optional[float] = None
    privacy_accountant: Optional[str] = None
    composition_method: Optional[str] = None
    total_epsilon: Optional[float] = None
    per_round_epsilon: Optional[float] = None
    secure_aggregation: bool
    mechanism: Optional[str] = None
    created_at: datetime


# System Event schemas
class SystemEventResponse(BaseSchema):
    id: UUID
    event_type: str
    severity: str
    message: str
    details: Optional[Dict[str, Any]] = None
    organisation_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    created_at: datetime


# Incident schemas
class IncidentBase(BaseSchema):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    attack_type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None


class IncidentCreate(IncidentBase):
    organisation_id: Optional[UUID] = None
    inference_job_id: Optional[UUID] = None
    detection_result_id: Optional[UUID] = None


class IncidentUpdate(BaseSchema):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    severity: Optional[str] = Field(None, pattern="^(low|medium|high|critical)$")
    status: Optional[str] = Field(None, pattern="^(open|investigating|resolved|closed)$")
    assigned_to: Optional[str] = None


class IncidentResponse(IncidentBase):
    id: UUID
    status: str
    organisation_id: Optional[UUID] = None
    inference_job_id: Optional[UUID] = None
    detection_result_id: Optional[UUID] = None
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# Health and stats schemas
class HealthResponse(BaseSchema):
    status: str
    timestamp: datetime
    version: str
    database: str
    uptime_seconds: float


class DashboardStats(BaseSchema):
    total_organisations: int
    active_experiments: int
    completed_experiments: int
    total_global_models: int
    active_models: int
    total_inference_jobs: int
    completed_inference_jobs: int
    total_detections: int
    malicious_detections: int
    incidents_open: int
    incidents_critical: int


class WebSocketMessage(BaseSchema):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Dataset Analysis schemas
class DatasetAnalysisResponse(BaseSchema):
    dataset_id: UUID
    filename: str
    rows: int
    columns: int
    feature_names: List[str]
    sample_rows: List[Dict[str, Any]]
    column_stats: Dict[str, Dict[str, Any]]  # min, max, mean, std, null_count per column
    label_distribution: Optional[Dict[str, int]] = None
    has_label_column: bool
    primary_label_column: Optional[str] = None
    candidate_label_columns: Optional[List[str]] = None
    file_size_bytes: Optional[int] = None
    memory_footprint_mb: Optional[float] = None
    detected_format: Optional[str] = None
    detected_encoding: Optional[str] = None
    detected_schema_family: Optional[str] = None
    schema_family_confidence: Optional[float] = None
    quality_score: Optional[float] = None
    quality_breakdown: Optional[Dict[str, float]] = None
    valid_row_count: Optional[int] = None
    invalid_row_count: Optional[int] = None
    duplicate_rows: Optional[int] = None
    total_missing_cells: Optional[int] = None
    total_infinite_cells: Optional[int] = None
    column_types: Optional[Dict[str, str]] = None
    type_counts: Optional[Dict[str, int]] = None
    identifier_columns: Optional[List[str]] = None
    empty_columns: Optional[List[str]] = None
    constant_columns: Optional[List[str]] = None
    network_related_columns: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class DatasetPreviewResponse(BaseSchema):
    dataset_id: UUID
    total_rows: int
    filtered_rows: Optional[int] = None
    total_columns: int
    page: int
    page_size: int
    total_pages: int
    columns: List[str]
    column_types: Optional[Dict[str, str]] = None
    missing_counts: Optional[Dict[str, int]] = None
    rows: List[Dict[str, Any]]


class ValidationReportResponse(BaseSchema):
    dataset_id: UUID
    is_valid: bool
    status: str  # "valid", "warning", "error"
    checks: List[Dict[str, Any]]  # Each check: {name, status, message, details}
    feature_compatibility: Dict[str, Any]  # expected vs actual features
    preprocessing_compatibility: bool
    valid_rows: Optional[int] = None
    invalid_rows: Optional[int] = None
    duplicate_rows: Optional[int] = None
    missing_values_count: Optional[int] = None
    infinite_values_count: Optional[int] = None
    quality_score: Optional[float] = None
    detected_schema_family: Optional[str] = None
    selected_detection_mode: Optional[str] = None
    compatibility_score: Optional[float] = None


class CompatibleModelResponse(BaseSchema):
    id: UUID
    version: str
    name: str
    algorithm: TrainingAlgorithm
    architecture: str
    metrics: Optional[Dict[str, Any]] = None
    threshold: float
    input_dim: int


# Pagination
class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


# Update forward references
OrganisationDetail.model_rebuild()
ExperimentDetail.model_rebuild()
GlobalModelDetail.model_rebuild()
InferenceJobDetail.model_rebuild()
TrainingRoundResponse.model_rebuild()
ClientRoundMetricsResponse.model_rebuild()