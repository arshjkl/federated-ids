"""
Database models for the Federated Learning IDS Platform.
Uses SQLAlchemy with PostgreSQL.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean,
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import TypeDecorator, VARCHAR
import uuid


class ValueEnum(TypeDecorator):
    """Enum column that binds by value and reads back case-insensitively."""
    impl = VARCHAR
    cache_ok = True

    def __init__(self, enum_class, **kwargs):
        self.enum_class = enum_class
        super().__init__(**kwargs)

    def process_bind_param(self, value, dialect):
        if value is not None:
            return value.value if hasattr(value, "value") else str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            for member in self.enum_class:
                if member.value.lower() == str(value).lower():
                    return member
            raise ValueError(f"Invalid {self.enum_class.__name__} value: {value}")
        return value


Base = declarative_base()


class OrganisationStatus(PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    PAUSED = "paused"


class DatasetStatus(PyEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    VALID = "valid"
    INVALID = "invalid"
    PROCESSING = "processing"
    READY = "ready"


class ExperimentStatus(PyEnum):
    CREATED = "created"
    CONFIGURING = "configuring"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingAlgorithm(PyEnum):
    CENTRALIZED = "centralized"
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    DP_FEDAVG = "dp_fedavg"
    DP_FEDPROX = "dp_fedprox"


class ClientStatus(PyEnum):
    IDLE = "idle"
    TRAINING = "training"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class GlobalModelStatus(PyEnum):
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    ARCHIVED = "archived"
    FAILED = "failed"


class InferenceStatus(PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Organisation(Base):
    __tablename__ = "organisations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    org_id = Column(String(100), nullable=False, unique=True)  # Human-readable ID
    status = Column(ValueEnum(OrganisationStatus), default=OrganisationStatus.ACTIVE, nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=True)
    local_model_path = Column(String(500), nullable=True)
    training_rounds_participated = Column(Integer, default=0)
    last_training_round = Column(Integer, nullable=True)
    total_samples = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    dataset = relationship("Dataset", foreign_keys="Organisation.dataset_id")
    assigned_dataset = relationship("Dataset", foreign_keys="Dataset.assigned_org_id", back_populates="assigned_organisation")
    training_history = relationship("TrainingHistory", back_populates="organisation")
    detection_history = relationship("DetectionHistory", back_populates="organisation")

    __table_args__ = (
        Index("ix_organisations_org_id", "org_id"),
        Index("ix_organisations_status", "status"),
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    status = Column(ValueEnum(DatasetStatus), default=DatasetStatus.UPLOADED, nullable=False)
    rows = Column(Integer, nullable=True)
    columns = Column(Integer, nullable=True)
    feature_names = Column(JSON, nullable=True)  # List of feature names
    label_distribution = Column(JSON, nullable=True)  # Class counts
    missing_values = Column(JSON, nullable=True)
    duplicate_rows = Column(Integer, nullable=True)
    assigned_org_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=True)
    uploaded_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    validated_at = Column(DateTime, nullable=True)

    # Relationships
    assigned_organisation = relationship("Organisation", foreign_keys="Dataset.assigned_org_id", back_populates="assigned_dataset")
    inference_jobs = relationship("InferenceJob", back_populates="dataset")

    __table_args__ = (
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_assigned_org", "assigned_org_id"),
    )


class FederatedExperiment(Base):
    __tablename__ = "federated_experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    algorithm = Column(ValueEnum(TrainingAlgorithm), nullable=False)
    status = Column(ValueEnum(ExperimentStatus), default=ExperimentStatus.CREATED, nullable=False)
    config = Column(JSON, nullable=False)  # Training configuration
    current_round = Column(Integer, default=0)
    total_rounds = Column(Integer, nullable=False)
    participating_orgs = Column(JSON, nullable=True)  # List of org IDs
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    global_models = relationship("GlobalModel", back_populates="experiment")
    rounds = relationship("TrainingRound", back_populates="experiment")
    client_participation = relationship("ClientParticipation", back_populates="experiment")

    __table_args__ = (
        Index("ix_experiments_status", "status"),
        Index("ix_experiments_algorithm", "algorithm"),
    )


class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("federated_experiments.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    status = Column(String(50), default="pending", nullable=False)  # pending, training, aggregating, completed
    global_loss = Column(Float, nullable=True)
    global_accuracy = Column(Float, nullable=True)
    global_f1 = Column(Float, nullable=True)
    global_precision = Column(Float, nullable=True)
    global_recall = Column(Float, nullable=True)
    aggregation_time = Column(Float, nullable=True)  # seconds
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    experiment = relationship("FederatedExperiment", back_populates="rounds")
    client_metrics = relationship("ClientRoundMetrics", back_populates="round")

    __table_args__ = (
        Index("ix_rounds_experiment", "experiment_id"),
        Index("ix_rounds_number", "round_number"),
        UniqueConstraint("experiment_id", "round_number", name="uq_experiment_round"),
    )


class ClientParticipation(Base):
    __tablename__ = "client_participation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("federated_experiments.id"), nullable=False)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_round = Column(Integer, default=0)

    # Relationships
    experiment = relationship("FederatedExperiment", back_populates="client_participation")
    organisation = relationship("Organisation")
    round_metrics = relationship("ClientRoundMetrics", back_populates="participation")

    __table_args__ = (
        Index("ix_client_participation_experiment", "experiment_id"),
        Index("ix_client_participation_org", "organisation_id"),
        UniqueConstraint("experiment_id", "organisation_id", name="uq_experiment_org"),
    )


class ClientRoundMetrics(Base):
    __tablename__ = "client_round_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id = Column(UUID(as_uuid=True), ForeignKey("training_rounds.id"), nullable=False)
    participation_id = Column(UUID(as_uuid=True), ForeignKey("client_participation.id"), nullable=False)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    local_loss = Column(Float, nullable=True)
    local_accuracy = Column(Float, nullable=True)
    local_f1 = Column(Float, nullable=True)
    local_precision = Column(Float, nullable=True)
    local_recall = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)  # seconds
    samples_used = Column(Integer, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    round = relationship("TrainingRound", back_populates="client_metrics")
    participation = relationship("ClientParticipation", back_populates="round_metrics")
    organisation = relationship("Organisation")

    __table_args__ = (
        Index("ix_client_metrics_round", "round_id"),
        Index("ix_client_metrics_org", "organisation_id"),
        UniqueConstraint("round_id", "organisation_id", name="uq_round_org_metrics"),
    )


class GlobalModel(Base):
    __tablename__ = "global_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String(50), nullable=False, unique=True)  # e.g., "v1.0.0", "exp_001_round_50"
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("federated_experiments.id"), nullable=True)
    parent_model_id = Column(UUID(as_uuid=True), ForeignKey("global_models.id"), nullable=True)
    algorithm = Column(ValueEnum(TrainingAlgorithm), nullable=False)
    architecture = Column(String(100), nullable=False)  # e.g., "MLPIDS", "ImprovedMLPIDS"
    architecture_config = Column(JSON, nullable=True)  # hidden_dims, dropout, etc.
    checkpoint_path = Column(String(500), nullable=False, unique=True)
    status = Column(ValueEnum(GlobalModelStatus), default=GlobalModelStatus.TRAINING, nullable=False)
    training_rounds = Column(Integer, nullable=False)
    participating_clients = Column(Integer, nullable=False)
    total_samples = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)  # accuracy, precision, recall, f1, roc_auc, etc.
    privacy_config = Column(JSON, nullable=True)  # epsilon, delta, noise_multiplier for DP
    training_config = Column(JSON, nullable=True)  # lr, batch_size, local_epochs, etc.
    trained_at = Column(DateTime, nullable=True)
    validated_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    experiment = relationship("FederatedExperiment", foreign_keys="GlobalModel.experiment_id", back_populates="global_models")
    parent_model = relationship("GlobalModel", remote_side=[id], backref="child_models")
    inference_jobs = relationship("InferenceJob", back_populates="model")

    __table_args__ = (
        Index("ix_global_models_version", "version"),
        Index("ix_global_models_status", "status"),
        Index("ix_global_models_algorithm", "algorithm"),
    )


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey("global_models.id"), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=True)
    status = Column(ValueEnum(InferenceStatus), default=InferenceStatus.PENDING, nullable=False)
    config = Column(JSON, nullable=True)  # threshold, batch_size, etc.
    total_records = Column(Integer, nullable=True)
    benign_count = Column(Integer, nullable=True)
    malicious_count = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds
    results_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    model = relationship("GlobalModel", back_populates="inference_jobs")
    dataset = relationship("Dataset", back_populates="inference_jobs")
    organisation = relationship("Organisation")
    detection_results = relationship("DetectionResult", back_populates="inference_job")

    __table_args__ = (
        Index("ix_inference_jobs_model", "model_id"),
        Index("ix_inference_jobs_dataset", "dataset_id"),
        Index("ix_inference_jobs_status", "status"),
    )


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    inference_job_id = Column(UUID(as_uuid=True), ForeignKey("inference_jobs.id"), nullable=False)
    record_index = Column(Integer, nullable=False)
    prediction = Column(Integer, nullable=False)  # 0 or 1
    probability = Column(Float, nullable=False)  # P(ATTACK)
    confidence = Column(Float, nullable=True)
    true_label = Column(Integer, nullable=True)  # If ground truth available
    features = Column(JSON, nullable=True)  # Original features for this record
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    inference_job = relationship("InferenceJob", back_populates="detection_results")

    __table_args__ = (
        Index("ix_detection_results_job", "inference_job_id"),
        Index("ix_detection_results_prediction", "prediction"),
    )


class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    inference_job_id = Column(UUID(as_uuid=True), ForeignKey("inference_jobs.id"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_analyzed = Column(Integer, nullable=False)
    benign_count = Column(Integer, nullable=False)
    malicious_count = Column(Integer, nullable=False)
    attack_types = Column(JSON, nullable=True)  # Distribution if multi-class
    model_version = Column(String(50), nullable=True)

    # Relationships
    organisation = relationship("Organisation", back_populates="detection_history")
    inference_job = relationship("InferenceJob")

    __table_args__ = (
        Index("ix_detection_history_org", "organisation_id"),
        Index("ix_detection_history_timestamp", "timestamp"),
    )


class TrainingHistory(Base):
    __tablename__ = "training_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=False)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("federated_experiments.id"), nullable=True)
    round_number = Column(Integer, nullable=False)
    local_loss = Column(Float, nullable=True)
    local_accuracy = Column(Float, nullable=True)
    local_f1 = Column(Float, nullable=True)
    local_precision = Column(Float, nullable=True)
    local_recall = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)
    samples_used = Column(Integer, nullable=True)
    completed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    organisation = relationship("Organisation", back_populates="training_history")
    experiment = relationship("FederatedExperiment")

    __table_args__ = (
        Index("ix_training_history_org", "organisation_id"),
        Index("ix_training_history_experiment", "experiment_id"),
    )


class PrivacyRecord(Base):
    __tablename__ = "privacy_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("federated_experiments.id"), nullable=False)
    global_model_id = Column(UUID(as_uuid=True), ForeignKey("global_models.id"), nullable=True)
    epsilon = Column(Float, nullable=True)
    delta = Column(Float, nullable=True)
    noise_multiplier = Column(Float, nullable=True)
    max_grad_norm = Column(Float, nullable=True)
    privacy_accountant = Column(String(50), nullable=True)  # "moments", "rdp", "gdp"
    composition_method = Column(String(50), nullable=True)  # "moments", "rdp", "naive"
    total_epsilon = Column(Float, nullable=True)
    per_round_epsilon = Column(Float, nullable=True)
    secure_aggregation = Column(Boolean, default=False)
    mechanism = Column(String(100), nullable=True)  # "gaussian", "laplace"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_privacy_experiment", "experiment_id"),
        Index("ix_privacy_global_model", "global_model_id"),
    )


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(100), nullable=False)  # "training_started", "round_completed", "model_deployed", "inference_completed", "error"
    severity = Column(String(20), default="info", nullable=False)  # "info", "warning", "error", "critical"
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=True)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("federated_experiments.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_system_events_type", "event_type"),
        Index("ix_system_events_severity", "severity"),
        Index("ix_system_events_created", "created_at"),
    )


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(20), nullable=False)  # "low", "medium", "high", "critical"
    status = Column(String(20), default="open", nullable=False)  # "open", "investigating", "resolved", "closed"
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id"), nullable=True)
    inference_job_id = Column(UUID(as_uuid=True), ForeignKey("inference_jobs.id"), nullable=True)
    detection_result_id = Column(UUID(as_uuid=True), ForeignKey("detection_results.id"), nullable=True)
    attack_type = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    source_ip = Column(String(50), nullable=True)
    destination_ip = Column(String(50), nullable=True)
    assigned_to = Column(String(255), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_incidents_org", "organisation_id"),
        Index("ix_incidents_status", "status"),
        Index("ix_incidents_severity", "severity"),
        Index("ix_incidents_created", "created_at"),
    )