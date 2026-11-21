"""
FastAPI application for the Federated Learning IDS Platform.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, or_, delete, text
from sqlalchemy.orm import selectinload
import uuid
import json
import asyncio
import os

from src.dashboard.db.session import db_manager, get_db, init_database, close_database
from src.dashboard.db.models import (
    Organisation, Dataset, FederatedExperiment, TrainingRound,
    ClientParticipation, ClientRoundMetrics, GlobalModel, InferenceJob,
    DetectionResult, DetectionHistory, TrainingHistory, PrivacyRecord,
    SystemEvent, Incident, OrganisationStatus, DatasetStatus,
    ExperimentStatus, TrainingAlgorithm, GlobalModelStatus, InferenceStatus
)
from src.dashboard.schemas import (
    OrganisationCreate, OrganisationUpdate, OrganisationResponse, OrganisationDetail,
    DatasetCreate, DatasetUpdate, DatasetResponse, DatasetValidationResponse,
    ExperimentCreate, ExperimentUpdate, ExperimentResponse, ExperimentDetail,
    ExperimentConfig, TrainingRoundResponse, ClientParticipationResponse,
    ClientRoundMetricsResponse, GlobalModelCreate, GlobalModelUpdate,
    GlobalModelResponse, GlobalModelDetail, InferenceJobCreate, InferenceJobUpdate,
    InferenceJobResponse, InferenceJobDetail, DetectionResultResponse,
    DetectionHistoryResponse, TrainingHistoryResponse, PrivacyRecordResponse,
    SystemEventResponse, IncidentCreate, IncidentUpdate, IncidentResponse,
    HealthResponse, DashboardStats, WebSocketMessage, PaginatedResponse,
    InferenceConfig,
    DatasetAnalysisResponse, DatasetPreviewResponse, ValidationReportResponse,
    CompatibleModelResponse
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MOCK_DIR = PROJECT_ROOT / "data" / "mock" / "inference"


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.org_connections: dict = {}  # org_id -> list of websockets

    async def connect(self, websocket: WebSocket, org_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if org_id:
            if org_id not in self.org_connections:
                self.org_connections[org_id] = []
            self.org_connections[org_id].append(websocket)

    def disconnect(self, websocket: WebSocket, org_id: Optional[str] = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if org_id and org_id in self.org_connections:
            if websocket in self.org_connections[org_id]:
                self.org_connections[org_id].remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

    async def broadcast_to_org(self, org_id: str, message: dict):
        if org_id in self.org_connections:
            for connection in self.org_connections[org_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    async with db_manager.session_factory() as db:
        # Normalize enum string columns to lowercase so ValueEnum queries match consistently
        await db.execute(text("UPDATE global_models SET status = lower(status), algorithm = lower(algorithm)"))
        await db.execute(text("UPDATE datasets SET status = lower(status)"))
        await db.execute(text("UPDATE organisations SET status = lower(status)"))
        await db.execute(text("UPDATE federated_experiments SET status = lower(status), algorithm = lower(algorithm)"))
        await db.execute(text("UPDATE inference_jobs SET status = lower(status)"))

        # Remove duplicate legacy entries with stale paths that are not referenced by inference_jobs
        await db.execute(
            text(
                "DELETE FROM global_models WHERE version IN ("
                "'centralized-baseline-v1', 'centralized-improved-v1', "
                "'federated-manual-v1', 'federated-manual-improved-v1', 'dp-federated-v1'"
                ") AND id NOT IN (SELECT DISTINCT model_id FROM inference_jobs WHERE model_id IS NOT NULL)"
            )
        )

        # Ensure centralized models use algorithm='centralized'
        await db.execute(
            text(
                "UPDATE global_models SET algorithm = 'centralized' "
                "WHERE version IN ('centralized_v1', 'centralized_improved_v1')"
            )
        )

        # Register Federated v2 (Improved FedAvg Round 20 & Round 10) checkpoints if not already registered
        fed_v2_models = [
            {
                "id": uuid.UUID("a1f2e3d4-5678-49ab-9cde-fed200000020"),
                "version": "federated_v2_r20",
                "name": "Federated v2 (Improved FedAvg, Round 20)",
                "description": "ImprovedMLPIDS trained via 5-client Federated Averaging (Round 20 checkpoint, 99.44% accuracy)",
                "algorithm": TrainingAlgorithm.FEDAVG,
                "architecture": "ImprovedMLPIDS",
                "checkpoint_path": "models/checkpoints/federated_v2_round_20.pth",
                "status": GlobalModelStatus.ACTIVE,
                "training_rounds": 20,
                "participating_clients": 5,
                "metrics": {
                    "accuracy": 0.9944,
                    "precision": 0.9743,
                    "recall": 0.9930,
                    "f1": 0.9836,
                    "roc_auc": 0.9998,
                    "fpr": 0.0053,
                    "fnr": 0.0070,
                },
            },
            {
                "id": uuid.UUID("a1f2e3d4-5678-49ab-9cde-fed200000010"),
                "version": "federated_v2_r10",
                "name": "Federated v2 (Improved FedAvg, Round 10)",
                "description": "ImprovedMLPIDS trained via 5-client Federated Averaging (Round 10 checkpoint, 99.33% accuracy)",
                "algorithm": TrainingAlgorithm.FEDAVG,
                "architecture": "ImprovedMLPIDS",
                "checkpoint_path": "models/checkpoints/federated_v2_round_10.pth",
                "status": GlobalModelStatus.ACTIVE,
                "training_rounds": 10,
                "participating_clients": 5,
                "metrics": {
                    "accuracy": 0.9933,
                    "precision": 0.9662,
                    "recall": 0.9949,
                    "f1": 0.9803,
                    "roc_auc": 0.9998,
                    "fpr": 0.0071,
                    "fnr": 0.0051,
                },
            },
        ]
        for m_spec in fed_v2_models:
            existing = await db.scalar(select(GlobalModel).where(GlobalModel.version == m_spec["version"]))
            if not existing:
                db.add(GlobalModel(**m_spec))

        # Seed default consortium organisations if table is empty
        try:
            org_rows = (await db.execute(select(Organisation))).scalars().all()
            if not org_rows:
                default_orgs = [
                    Organisation(
                        id=uuid.UUID("11111111-1111-4111-8111-111111111111"),
                        name="FinTech Security Operations",
                        description="Financial services threat intelligence & fraud network defense",
                        org_id="org_fintech_soc",
                        status=OrganisationStatus.ACTIVE,
                        total_samples=25000,
                        training_rounds_participated=20,
                    ),
                    Organisation(
                        id=uuid.UUID("22222222-2222-4222-8222-222222222222"),
                        name="Healthcare Defense Center",
                        description="Hospital networks and biomedical telemetry intrusion detection",
                        org_id="org_health_shield",
                        status=OrganisationStatus.ACTIVE,
                        total_samples=20000,
                        training_rounds_participated=20,
                    ),
                    Organisation(
                        id=uuid.UUID("33333333-3333-4333-8333-333333333333"),
                        name="Critical Infrastructure Protection",
                        description="Smart grid, SCADA, and industrial IoT monitoring consortium",
                        org_id="org_grid_guard",
                        status=OrganisationStatus.ACTIVE,
                        total_samples=18000,
                        training_rounds_participated=18,
                    ),
                    Organisation(
                        id=uuid.UUID("44444444-4444-4444-8444-444444444444"),
                        name="Telecom Cyber Defense",
                        description="Carrier-grade backbone IP traffic & DDoS mitigation center",
                        org_id="org_telco_soc",
                        status=OrganisationStatus.ACTIVE,
                        total_samples=32000,
                        training_rounds_participated=20,
                    ),
                ]
                for o in default_orgs:
                    db.add(o)
                await db.commit()
                org_rows = default_orgs

            if org_rows:
                first_org_id_str = str(org_rows[0].id).replace("-", "")
                await db.execute(
                    text("UPDATE inference_jobs SET organisation_id = :oid WHERE organisation_id IS NULL OR organisation_id = ''"),
                    {"oid": first_org_id_str},
                )
                await db.commit()
        except Exception as e:
            logger.warning(f"Organisation seeding: {e}")

        await db.commit()
    yield
    # Shutdown
    await close_database()


app = FastAPI(
    title="Federated Learning IDS Platform",
    description="A platform for federated learning-based intrusion detection",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins_env = os.getenv("CORS_ORIGINS", "*").strip()
cors_origins = ["*"] if cors_origins_env == "*" else [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# Health check
@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(select(1))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        database=db_status,
        uptime_seconds=0.0,  # Could track actual uptime
    )


# Dashboard stats
@app.get("/api/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    total_orgs = await db.scalar(select(func.count(Organisation.id)))
    active_experiments = await db.scalar(
        select(func.count(FederatedExperiment.id)).where(
            FederatedExperiment.status.in_([ExperimentStatus.RUNNING, ExperimentStatus.CONFIGURING])
        )
    )
    completed_experiments = await db.scalar(
        select(func.count(FederatedExperiment.id)).where(
            FederatedExperiment.status == ExperimentStatus.COMPLETED
        )
    )
    total_models = await db.scalar(select(func.count(GlobalModel.id)))
    active_models = await db.scalar(
        select(func.count(GlobalModel.id)).where(GlobalModel.status == GlobalModelStatus.ACTIVE)
    )
    total_inference = await db.scalar(select(func.count(InferenceJob.id)))
    completed_inference = await db.scalar(
        select(func.count(InferenceJob.id)).where(InferenceJob.status == InferenceStatus.COMPLETED)
    )
    total_detections = await db.scalar(select(func.count(DetectionResult.id)))
    malicious_detections = await db.scalar(
        select(func.count(DetectionResult.id)).where(DetectionResult.prediction == 1)
    )
    incidents_open = await db.scalar(
        select(func.count(Incident.id)).where(Incident.status == "open")
    )
    incidents_critical = await db.scalar(
        select(func.count(Incident.id)).where(and_(Incident.status == "open", Incident.severity == "critical"))
    )

    return DashboardStats(
        total_organisations=total_orgs or 0,
        active_experiments=active_experiments or 0,
        completed_experiments=completed_experiments or 0,
        total_global_models=total_models or 0,
        active_models=active_models or 0,
        total_inference_jobs=total_inference or 0,
        completed_inference_jobs=completed_inference or 0,
        total_detections=total_detections or 0,
        malicious_detections=malicious_detections or 0,
        incidents_open=incidents_open or 0,
        incidents_critical=incidents_critical or 0,
    )


# Organisation endpoints
@app.post("/api/organisations", response_model=OrganisationResponse, status_code=201)
async def create_organisation(org: OrganisationCreate, db: AsyncSession = Depends(get_db)):
    # Check if org_id already exists
    existing = await db.scalar(select(Organisation).where(Organisation.org_id == org.org_id))
    if existing:
        raise HTTPException(status_code=400, detail="Organisation ID already exists")

    db_org = Organisation(
        name=org.name,
        description=org.description,
        org_id=org.org_id,
        dataset_id=org.dataset_id,
        status=OrganisationStatus.ACTIVE,
    )
    db.add(db_org)
    await db.commit()
    await db.refresh(db_org)
    return db_org


@app.get("/api/organisations", response_model=PaginatedResponse)
async def list_organisations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[OrganisationStatus] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Organisation)
    if status:
        query = query.where(Organisation.status == status)
    query = query.order_by(desc(Organisation.created_at))

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    items = result.scalars().all()

    # Convert ORM models to Pydantic schemas
    item_schemas = [OrganisationResponse.model_validate(item) for item in items]

    return PaginatedResponse(
        items=item_schemas,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/organisations/{org_id}", response_model=OrganisationDetail)
async def get_organisation(org_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Organisation)
        .options(selectinload(Organisation.dataset))
        .where(Organisation.id == org_id)
    )
    org = result.scalar_one_or_none()
    if not org:
        raise HTTPException(status_code=404, detail="Organisation not found")
    return org


@app.patch("/api/organisations/{org_id}", response_model=OrganisationResponse)
async def update_organisation(
    org_id: uuid.UUID, org_update: OrganisationUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Organisation).where(Organisation.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise HTTPException(status_code=404, detail="Organisation not found")

    update_data = org_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(org, field, value)

    await db.commit()
    await db.refresh(org)
    return org


@app.delete("/api/organisations/{org_id}", status_code=204)
async def delete_organisation(org_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Organisation).where(Organisation.id == org_id))
    org = result.scalar_one_or_none()
    if not org:
        raise HTTPException(status_code=404, detail="Organisation not found")
    await db.delete(org)
    await db.commit()


# Dataset endpoints
@app.post("/api/datasets", response_model=DatasetResponse, status_code=201)
async def create_dataset(dataset: DatasetCreate, db: AsyncSession = Depends(get_db)):
    db_dataset = Dataset(**dataset.model_dump())
    db.add(db_dataset)
    await db.commit()
    await db.refresh(db_dataset)
    return db_dataset


@app.get("/api/datasets", response_model=PaginatedResponse)
async def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[DatasetStatus] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Dataset)
    if status:
        query = query.where(Dataset.status == status)
    query = query.order_by(desc(Dataset.created_at))

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    items = result.scalars().all()

    # Convert ORM models to Pydantic schemas
    item_schemas = [DatasetResponse.model_validate(item) for item in items]

    return PaginatedResponse(
        items=item_schemas,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@app.post("/api/datasets/{dataset_id}/validate", response_model=DatasetValidationResponse)
async def validate_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Validate a dataset using the Universal Dataset Intelligence & Profiling Engine."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        rel_name = os.path.basename(dataset.file_path)
        candidates = [
            f"data/mock/inference/{rel_name}",
            f"/app/data/mock/inference/{rel_name}",
            f"artifacts/uploaded_datasets/{rel_name}",
            f"/app/artifacts/uploaded_datasets/{rel_name}",
        ]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                dataset.file_path = cand
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Dataset file not found")

    dataset.status = DatasetStatus.VALIDATING
    await db.commit()

    from src.inference.pipeline import get_adaptive_pipeline
    pipeline = get_adaptive_pipeline()
    try:
        df, meta, profile, mapping, derivation, compatibility, val_report = pipeline.inspect_dataset(
            dataset.file_path, dataset_id=str(dataset.id)
        )
        missing_by_col = {
            k: int(v.get("null_count", 0))
            for k, v in profile.column_stats.items()
            if v.get("null_count", 0) > 0
        }
        validation = DatasetValidationResponse(
            is_valid=bool(val_report["is_valid"]),
            rows=profile.total_rows,
            columns=profile.total_columns,
            feature_names=list(df.columns),
            label_distribution=profile.label_distribution or {},
            missing_values=missing_by_col,
            duplicate_rows=profile.duplicate_rows,
            constant_features=profile.constant_columns,
            warnings=profile.warnings,
        )
        dataset.status = DatasetStatus.VALID if validation.is_valid else DatasetStatus.INVALID
        dataset.rows = validation.rows
        dataset.columns = validation.columns
        dataset.feature_names = validation.feature_names
        dataset.label_distribution = validation.label_distribution
        dataset.missing_values = validation.missing_values
        dataset.duplicate_rows = validation.duplicate_rows
        dataset.validated_at = datetime.utcnow()
        await db.commit()
        return validation
    except Exception as exc:
        dataset.status = DatasetStatus.INVALID
        await db.commit()
        raise HTTPException(status_code=400, detail=f"Dataset validation failed: {exc}")


@app.patch("/api/datasets/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: uuid.UUID, dataset_update: DatasetUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = dataset_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(dataset, field, value)

    await db.commit()
    await db.refresh(dataset)
    return dataset


@app.delete("/api/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await db.delete(dataset)
    await db.commit()


# Experiment endpoints
@app.post("/api/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(exp: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    # Verify organisations exist
    for org_id in exp.participating_org_ids:
        org = await db.scalar(select(Organisation).where(Organisation.id == org_id))
        if not org:
            raise HTTPException(status_code=400, detail=f"Organisation {org_id} not found")

    cfg = exp.config.model_dump()
    if exp.dataset_id:
        cfg["dataset_id"] = str(exp.dataset_id)
    if exp.sample_file:
        cfg["sample_file"] = exp.sample_file
    if exp.target_column:
        cfg["target_column"] = exp.target_column

    db_exp = FederatedExperiment(
        name=exp.name,
        description=exp.description,
        algorithm=exp.algorithm,
        config=cfg,
        total_rounds=exp.total_rounds,
        participating_orgs=[str(oid) for oid in exp.participating_org_ids],
        status=ExperimentStatus.CREATED,
    )
    db.add(db_exp)
    await db.commit()
    await db.refresh(db_exp)

    # Create client participation records
    for org_id in exp.participating_org_ids:
        participation = ClientParticipation(
            experiment_id=db_exp.id,
            organisation_id=org_id,
            is_active=True,
        )
        db.add(participation)

    await db.commit()
    return db_exp


@app.get("/api/experiments", response_model=PaginatedResponse)
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[ExperimentStatus] = None,
    algorithm: Optional[TrainingAlgorithm] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(FederatedExperiment)
    if status:
        query = query.where(FederatedExperiment.status == status)
    if algorithm:
        query = query.where(FederatedExperiment.algorithm == algorithm)
    query = query.order_by(desc(FederatedExperiment.created_at))

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    items = result.scalars().all()

    # Convert ORM models to Pydantic schemas
    item_schemas = [ExperimentResponse.model_validate(item) for item in items]

    return PaginatedResponse(
        items=item_schemas,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/experiments/{exp_id}", response_model=ExperimentDetail)
async def get_experiment(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(FederatedExperiment)
        .options(
            selectinload(FederatedExperiment.global_models),
            selectinload(FederatedExperiment.rounds).selectinload(TrainingRound.client_metrics),
        )
        .where(FederatedExperiment.id == exp_id)
    )
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@app.delete("/api/experiments/{exp_id}", status_code=204)
async def delete_experiment(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FederatedExperiment).where(FederatedExperiment.id == exp_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    rounds_result = await db.execute(select(TrainingRound.id).where(TrainingRound.experiment_id == exp_id))
    round_ids = [r[0] for r in rounds_result.all()]
    if round_ids:
        await db.execute(delete(ClientRoundMetrics).where(ClientRoundMetrics.round_id.in_(round_ids)))
    await db.execute(delete(TrainingRound).where(TrainingRound.experiment_id == exp_id))
    await db.execute(delete(ClientParticipation).where(ClientParticipation.experiment_id == exp_id))
    await db.delete(exp)
    await db.commit()


@app.patch("/api/experiments/{exp_id}", response_model=ExperimentResponse)
async def update_experiment(
    exp_id: uuid.UUID, exp_update: ExperimentUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(FederatedExperiment).where(FederatedExperiment.id == exp_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    update_data = exp_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(exp, field, value)

    await db.commit()
    await db.refresh(exp)
    return exp


@app.post("/api/experiments/{exp_id}/start", response_model=ExperimentResponse)
async def start_experiment(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FederatedExperiment).where(FederatedExperiment.id == exp_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    if exp.status not in [ExperimentStatus.CREATED, ExperimentStatus.CONFIGURING, ExperimentStatus.PAUSED]:
        raise HTTPException(status_code=400, detail="Experiment cannot be started in current state")

    cfg = dict(exp.config) if isinstance(exp.config, dict) else {}
    dataset_id = cfg.get("dataset_id")
    sample_file = cfg.get("sample_file")
    target_col = cfg.get("target_column")

    # Broadcast started via WebSocket
    await manager.broadcast({
        "type": "experiment_started",
        "data": {"experiment_id": str(exp.id), "status": "running"},
        "timestamp": datetime.utcnow().isoformat(),
    })

    try:
        from src.dashboard.services.experiment_trainer import get_experiment_trainer
        trainer = get_experiment_trainer()
        exp, trained_model = await trainer.train_experiment(
            experiment_id=exp.id,
            db=db,
            dataset_id=dataset_id,
            sample_file=sample_file,
            target_column=target_col,
        )

        await manager.broadcast({
            "type": "experiment_completed",
            "data": {
                "experiment_id": str(exp.id),
                "model_id": str(trained_model.id),
                "model_name": trained_model.name,
                "metrics": trained_model.metrics,
            },
            "timestamp": datetime.utcnow().isoformat(),
        })
        return exp
    except Exception as exc:
        exp.status = ExperimentStatus.FAILED
        await db.commit()
        await db.refresh(exp)
        await manager.broadcast({
            "type": "experiment_failed",
            "data": {"experiment_id": str(exp.id), "error": str(exc)},
            "timestamp": datetime.utcnow().isoformat(),
        })
        raise HTTPException(status_code=500, detail=f"Experiment training failed: {exc}")


@app.post("/api/experiments/{exp_id}/pause", response_model=ExperimentResponse)
async def pause_experiment(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FederatedExperiment).where(FederatedExperiment.id == exp_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp.status = ExperimentStatus.PAUSED
    await db.commit()
    await db.refresh(exp)

    await manager.broadcast({
        "type": "experiment_paused",
        "data": {"experiment_id": str(exp.id)},
        "timestamp": datetime.utcnow().isoformat(),
    })

    return exp


@app.post("/api/experiments/{exp_id}/resume", response_model=ExperimentResponse)
async def resume_experiment(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FederatedExperiment).where(FederatedExperiment.id == exp_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp.status = ExperimentStatus.RUNNING
    await db.commit()
    await db.refresh(exp)

    await manager.broadcast({
        "type": "experiment_resumed",
        "data": {"experiment_id": str(exp.id)},
        "timestamp": datetime.utcnow().isoformat(),
    })

    return exp


@app.post("/api/experiments/{exp_id}/stop", response_model=ExperimentResponse)
async def stop_experiment(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(FederatedExperiment).where(FederatedExperiment.id == exp_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp.status = ExperimentStatus.STOPPED
    exp.completed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(exp)

    await manager.broadcast({
        "type": "experiment_stopped",
        "data": {"experiment_id": str(exp.id)},
        "timestamp": datetime.utcnow().isoformat(),
    })

    return exp


# Global Model endpoints
@app.post("/api/models", response_model=GlobalModelResponse, status_code=201)
async def create_global_model(model: GlobalModelCreate, db: AsyncSession = Depends(get_db)):
    db_model = GlobalModel(**model.model_dump())
    db.add(db_model)
    await db.commit()
    await db.refresh(db_model)
    return db_model


@app.get("/api/models", response_model=PaginatedResponse)
async def list_global_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[GlobalModelStatus] = None,
    algorithm: Optional[TrainingAlgorithm] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(GlobalModel)
    if status:
        query = query.where(GlobalModel.status == status)
    if algorithm:
        query = query.where(GlobalModel.algorithm == algorithm)
    query = query.order_by(desc(GlobalModel.created_at))

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    models = result.scalars().all()

    # Convert SQLAlchemy models to Pydantic schemas
    items = [GlobalModelResponse.model_validate(m) for m in models]

    return PaginatedResponse(
        items=items,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/models/compatible", response_model=List[CompatibleModelResponse])
async def list_compatible_models(
    dataset_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List models compatible with the given dataset (or all active models if no dataset)."""
    # Get all active models from database
    query = select(GlobalModel).where(GlobalModel.status == GlobalModelStatus.ACTIVE)
    result = await db.execute(query)
    models = result.scalars().all()

    # Also include file-based models from ModelLoader
    from src.dashboard.services.model_loader import get_model_loader
    loader = get_model_loader()
    file_models = loader.list_available_models()

    # For now, return database models + indicate file-based models
    # The frontend can use this to show all available models
    compatible = []

    for model in models:
        # Get threshold from model loader if available
        threshold = 0.5
        try:
            model_key = None
            for key, path in loader.MODEL_PATHS.items():
                if path == model.checkpoint_path or model.version in path:
                    model_key = key
                    break
            if model_key:
                threshold = loader.get_threshold(model_key)
        except Exception:
            pass

        in_dim = 70
        if isinstance(model.architecture_config, dict):
            in_dim = model.architecture_config.get("input_dim", 70)

        compatible.append(CompatibleModelResponse(
            id=model.id,
            version=model.version,
            name=model.name,
            algorithm=model.algorithm,
            architecture=model.architecture,
            metrics=model.metrics,
            threshold=threshold,
            input_dim=in_dim,
        ))

    return compatible


@app.get("/api/models/{model_id}", response_model=GlobalModelDetail)
async def get_global_model(model_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(GlobalModel)
        .options(
            selectinload(GlobalModel.experiment),
            selectinload(GlobalModel.inference_jobs),
        )
        .where(GlobalModel.id == model_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return GlobalModelDetail.model_validate(model)


@app.patch("/api/models/{model_id}", response_model=GlobalModelResponse)
async def update_global_model(
    model_id: uuid.UUID, model_update: GlobalModelUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(GlobalModel).where(GlobalModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    update_data = model_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)

    await db.commit()
    await db.refresh(model)
    return model


@app.delete("/api/models/{model_id}", status_code=204)
async def delete_global_model(model_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(GlobalModel).where(GlobalModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    await db.delete(model)
    await db.commit()


# Inference Job endpoints

def _unpack_detection_record(det: DetectionResult) -> Dict[str, Any]:
    """Extract Phase 9 detection metadata and clean feature dictionary from a DetectionResult."""
    raw_features = dict(det.features) if isinstance(det.features, dict) else {}
    meta = raw_features.pop("__meta__", {}) if isinstance(raw_features.get("__meta__"), dict) else {}

    pred_cat = meta.get("predicted_category") or ("Confirmed Attack (Supervised)" if det.prediction == 1 else "Normal Traffic")
    risk_score = meta.get("risk_score", det.probability)
    sev = meta.get("severity")
    if not sev:
        if risk_score >= 0.85:
            sev = "critical"
        elif risk_score >= 0.70:
            sev = "high"
        elif risk_score >= 0.50 or det.prediction == 1:
            sev = "medium"
        elif risk_score >= 0.30:
            sev = "low"
        else:
            sev = "informational"

    return {
        "id": det.id,
        "inference_job_id": det.inference_job_id,
        "record_index": det.record_index,
        "prediction": det.prediction,
        "probability": det.probability,
        "confidence": det.confidence,
        "true_label": det.true_label,
        "features": raw_features,
        "timestamp": det.timestamp,
        "predicted_category": pred_cat,
        "supervised_probability": meta.get("supervised_probability", det.probability),
        "anomaly_score": meta.get("anomaly_score"),
        "rule_score": meta.get("rule_score", 0.0),
        "risk_score": risk_score,
        "severity": sev,
        "detection_mode": meta.get("detection_mode", "supervised"),
        "triggered_rules": meta.get("triggered_rules", []),
        "explanation": meta.get("explanation", ""),
        "top_anomaly_features": meta.get("top_anomaly_features", []),
        "model_used": meta.get("model_used"),
        "compatibility_score": meta.get("compatibility_score"),
    }


def _get_dataset_overrides(dataset: Dataset) -> Dict[str, str]:
    if isinstance(dataset.missing_values, dict) and isinstance(dataset.missing_values.get("__manual_overrides__"), dict):
        return dict(dataset.missing_values["__manual_overrides__"])
    return {}


# Specific routes MUST come before parameterized routes
# File Upload for direct inference
@app.post("/api/inference/upload-csv", response_model=InferenceJobDetail)
async def upload_csv_and_run_inference(
    file: UploadFile = File(...),
    model_id: Optional[uuid.UUID] = Form(None),
    organisation_id: Optional[uuid.UUID] = Form(None),
    name: str = Form(...),
    threshold: float = Form(0.5),
    batch_size: int = Form(256),
    detection_mode: Optional[str] = Form(None),
    manual_overrides: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a dataset file and run adaptive inference directly."""
    overrides_dict = json.loads(manual_overrides) if manual_overrides else None
    return await _run_inference_from_path(
        model_id=model_id,
        name=name,
        threshold=threshold,
        batch_size=batch_size,
        file=file,
        is_sample=False,
        sample_filename=None,
        detection_mode=detection_mode,
        manual_overrides=overrides_dict,
        organisation_id=organisation_id,
        db=db,
    )


# Sample data inference endpoint
@app.post("/api/inference/upload-sample", response_model=InferenceJobDetail)
async def upload_sample_and_run_inference(
    model_id: Optional[uuid.UUID] = Form(None),
    organisation_id: Optional[uuid.UUID] = Form(None),
    name: str = Form(...),
    threshold: float = Form(0.5),
    batch_size: int = Form(256),
    sample_file: str = Form(...),
    detection_mode: Optional[str] = Form(None),
    manual_overrides: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Use pre-generated sample data and run adaptive inference directly."""
    mock_dir = MOCK_DIR
    temp_path = mock_dir / sample_file
    if not temp_path.exists():
        # Also allow root-level sample CSVs (mock_cic_ids.csv, test_cic_ids.csv, test_inference.csv)
        alt_path = PROJECT_ROOT / sample_file
        if not alt_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_file}")

    overrides_dict = json.loads(manual_overrides) if manual_overrides else None
    return await _run_inference_from_path(
        model_id=model_id,
        name=name,
        threshold=threshold,
        batch_size=batch_size,
        file=None,
        is_sample=True,
        sample_filename=sample_file,
        detection_mode=detection_mode,
        manual_overrides=overrides_dict,
        organisation_id=organisation_id,
        db=db,
    )


@app.post("/api/inference/run-on-dataset", response_model=InferenceJobDetail)
async def run_inference_on_dataset(
    dataset_id: uuid.UUID = Form(...),
    model_id: Optional[uuid.UUID] = Form(None),
    organisation_id: Optional[uuid.UUID] = Form(None),
    name: str = Form(...),
    threshold: float = Form(0.5),
    batch_size: int = Form(256),
    detection_mode: Optional[str] = Form(None),
    manual_overrides: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Run adaptive 18-stage inference on an analyzed dataset."""
    if model_id is None:
        model_id = uuid.UUID("a1f2e3d4-5678-49ab-9cde-fed200000020")
    model = await db.scalar(select(GlobalModel).where(GlobalModel.id == model_id))
    if not model:
        model = await db.scalar(select(GlobalModel))
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        model_id = model.id

    dataset = await db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if organisation_id is None:
        if dataset.assigned_org_id:
            organisation_id = dataset.assigned_org_id
        else:
            first_org = await db.scalar(select(Organisation))
            if first_org:
                organisation_id = first_org.id

    overrides_dict = json.loads(manual_overrides) if manual_overrides else _get_dataset_overrides(dataset)

    from src.dashboard.schemas import InferenceConfig
    db_job = InferenceJob(
        name=name,
        model_id=model_id,
        dataset_id=dataset_id,
        organisation_id=organisation_id,
        config=InferenceConfig(
            threshold=threshold,
            batch_size=batch_size,
            detection_mode=detection_mode,
            manual_overrides=overrides_dict,
        ).model_dump(),
        status=InferenceStatus.RUNNING,
        started_at=datetime.utcnow(),
    )
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)

    from src.dashboard.services.inference_runner import InferenceRunner
    runner = InferenceRunner()

    try:
        await runner.run_inference_job(
            db_job.id, db, requested_mode=detection_mode, manual_overrides=overrides_dict
        )
        job_result = await db.execute(select(InferenceJob).where(InferenceJob.id == db_job.id))
        db_job = job_result.scalar_one()
        model_result = await db.execute(select(GlobalModel).where(GlobalModel.id == db_job.model_id))
        db_model = model_result.scalar_one()
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == db_job.dataset_id))
        db_dataset = dataset_result.scalar_one()
        return InferenceJobDetail(
            id=db_job.id,
            name=db_job.name,
            model_id=db_job.model_id,
            dataset_id=db_job.dataset_id,
            organisation_id=db_job.organisation_id,
            status=db_job.status,
            config=db_job.config,
            total_records=db_job.total_records,
            benign_count=db_job.benign_count,
            malicious_count=db_job.malicious_count,
            processing_time=db_job.processing_time,
            results_path=db_job.results_path,
            error_message=db_job.error_message,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
            created_at=db_job.created_at,
            model=GlobalModelResponse.model_validate(db_model, from_attributes=True),
            dataset=DatasetResponse.model_validate(db_dataset, from_attributes=True),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


@app.post("/api/inference", response_model=InferenceJobResponse, status_code=201)
async def create_inference_job(job: InferenceJobCreate, db: AsyncSession = Depends(get_db)):
    # Verify model exists
    model = await db.scalar(select(GlobalModel).where(GlobalModel.id == job.model_id))
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Verify dataset exists
    dataset = await db.scalar(select(Dataset).where(Dataset.id == job.dataset_id))
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    db_job = InferenceJob(
        name=job.name,
        model_id=job.model_id,
        dataset_id=job.dataset_id,
        organisation_id=job.organisation_id,
        config=job.config.model_dump() if job.config else None,
        status=InferenceStatus.PENDING,
    )
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)
    return db_job


@app.get("/api/inference", response_model=PaginatedResponse)
async def list_inference_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[InferenceStatus] = None,
    model_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(InferenceJob).options(selectinload(InferenceJob.model))
    if status:
        query = query.where(InferenceJob.status == status)
    if model_id:
        query = query.where(InferenceJob.model_id == model_id)
    query = query.order_by(desc(InferenceJob.created_at))

    count_query = select(InferenceJob)
    if status:
        count_query = count_query.where(InferenceJob.status == status)
    if model_id:
        count_query = count_query.where(InferenceJob.model_id == model_id)

    total = await db.scalar(select(func.count()).select_from(count_query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    items = result.scalars().all()

    # Convert SQLAlchemy models to Pydantic response models with model_name
    serialized_items = []
    for item in items:
        resp = InferenceJobResponse.model_validate(item)
        if item.model:
            resp.model_name = item.model.name
            resp.model_version = item.model.version
        serialized_items.append(resp)

    return PaginatedResponse(
        items=serialized_items,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/inference/{job_id}", response_model=InferenceJobDetail)
async def get_inference_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(InferenceJob)
        .options(
            selectinload(InferenceJob.model),
            selectinload(InferenceJob.dataset),
        )
        .where(InferenceJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")
    return job


@app.delete("/api/inference/{job_id}", status_code=204)
async def delete_inference_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InferenceJob).where(InferenceJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    # Delete associated detection results
    await db.execute(
        delete(DetectionResult).where(DetectionResult.inference_job_id == job_id)
    )

    # Delete the inference job
    await db.delete(job)
    await db.commit()


@app.post("/api/inference/{job_id}/start", response_model=InferenceJobResponse)
async def start_inference_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(InferenceJob).where(InferenceJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    job.status = InferenceStatus.RUNNING
    job.started_at = datetime.utcnow()
    await db.commit()
    await db.refresh(job)

    # Run inference in background
    async def _run_inference_bg(jid: uuid.UUID):
        async with db_manager.session_factory() as bg_db:
            runner = InferenceRunner()
            try:
                await runner.run_inference_job(jid, bg_db)
            except Exception as e:
                logger.error(f"Background inference job {jid} failed: {e}")

    asyncio.create_task(_run_inference_bg(job_id))

    await manager.broadcast({
        "type": "inference_started",
        "data": {"job_id": str(job.id)},
        "timestamp": datetime.utcnow().isoformat(),
    })

    return job


@app.get("/api/inference/{job_id}/results", response_model=List[DetectionResultResponse])
async def get_inference_results(
    job_id: uuid.UUID,
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    prediction: Optional[int] = None,
    severity: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(DetectionResult).where(DetectionResult.inference_job_id == job_id)
    if prediction is not None:
        query = query.where(DetectionResult.prediction == prediction)
    query = query.order_by(DetectionResult.record_index).offset(offset).limit(limit)

    result = await db.execute(query)
    rows = result.scalars().all()
    unpacked = [DetectionResultResponse(**_unpack_detection_record(r)) for r in rows]
    if severity:
        unpacked = [u for u in unpacked if u.severity == severity.lower()]
    return unpacked


@app.get("/api/inference/{job_id}/stages")
async def get_inference_pipeline_stages(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get 18-stage pipeline execution telemetry and summary analytics for an inference job."""
    result = await db.execute(select(InferenceJob).where(InferenceJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    if job.results_path and os.path.exists(job.results_path):
        try:
            with open(job.results_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    cfg = job.config if isinstance(job.config, dict) else {}
    return {
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "stages": cfg.get("stages_summary", []),
        "summary_metrics": cfg.get("summary_metrics", {}),
    }


@app.get("/api/inference/{job_id}/records/{record_index}")
async def get_inference_record_investigation(
    job_id: uuid.UUID,
    record_index: int,
    db: AsyncSession = Depends(get_db),
):
    """Investigate a single record: original row, canonical derived features, model result, rules, explanation."""
    job_res = await db.execute(
        select(InferenceJob)
        .options(selectinload(InferenceJob.model), selectinload(InferenceJob.dataset))
        .where(InferenceJob.id == job_id)
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    det_res = await db.execute(
        select(DetectionResult).where(
            and_(
                DetectionResult.inference_job_id == job_id,
                DetectionResult.record_index == record_index,
            )
        )
    )
    det = det_res.scalar_one_or_none()
    if not det:
        raise HTTPException(status_code=404, detail="Detection record not found")

    unpacked = _unpack_detection_record(det)
    original_row = dict(unpacked.get("features") or {})

    # If the original dataset file exists on disk and features snapshot was truncated, load exact row
    if job.dataset and job.dataset.file_path and os.path.exists(job.dataset.file_path) and not original_row:
        try:
            from src.inference.ingestion import get_ingestion_engine
            df_raw, _ = get_ingestion_engine().load_dataframe(job.dataset.file_path)
            if 0 <= record_index < len(df_raw):
                row_s = df_raw.iloc[record_index]
                original_row = {
                    str(k): (None if pd.isna(v) else (float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (int(v) if isinstance(v, (int, np.integer)) else str(v))))
                    for k, v in row_s.to_dict().items()
                }
        except Exception:
            pass

    cfg = job.config if isinstance(job.config, dict) else {}
    return {
        **unpacked,
        "id": str(unpacked["id"]),
        "inference_job_id": str(unpacked["inference_job_id"]),
        "timestamp": unpacked["timestamp"].isoformat() if unpacked.get("timestamp") else None,
        "original_record": original_row,
        "job_name": job.name,
        "model_name": job.model.name if job.model else unpacked.get("model_used"),
        "dataset_name": job.dataset.name if job.dataset else None,
        "compatibility_info": {
            "detection_mode": unpacked.get("detection_mode"),
            "schema_family": cfg.get("schema_family"),
            "compatibility_score": unpacked.get("compatibility_score") or cfg.get("compatibility_score"),
            "feature_coverage_score": cfg.get("feature_coverage_score"),
            "data_quality_score": cfg.get("data_quality_score"),
        },
    }


async def _run_inference_from_path(
    model_id: Optional[uuid.UUID],
    name: str,
    threshold: float,
    batch_size: int,
    file: Optional[UploadFile],
    is_sample: bool,
    sample_filename: Optional[str],
    db: AsyncSession,
    detection_mode: Optional[str] = None,
    manual_overrides: Optional[Dict[str, str]] = None,
    organisation_id: Optional[uuid.UUID] = None,
) -> InferenceJobDetail:
    """Common adaptive inference logic for both file upload and sample data."""
    if model_id is None:
        model_id = uuid.UUID("a1f2e3d4-5678-49ab-9cde-fed200000020")
    model = await db.scalar(select(GlobalModel).where(GlobalModel.id == model_id))
    if not model:
        model = await db.scalar(select(GlobalModel))
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        model_id = model.id

    if organisation_id is None:
        first_org = await db.scalar(select(Organisation))
        if first_org:
            organisation_id = first_org.id

    from src.inference.ingestion import UniversalIngestionEngine

    if is_sample:
        mock_dir = MOCK_DIR
        temp_path = mock_dir / sample_filename
        if not temp_path.exists():
            temp_path = PROJECT_ROOT / sample_filename
        content = temp_path.read_bytes()
        original_filename = sample_filename
    else:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in UniversalIngestionEngine.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{suffix}'. Supported: {sorted(UniversalIngestionEngine.SUPPORTED_EXTENSIONS)}",
            )
        upload_dir = PROJECT_ROOT / "artifacts" / "uploaded_datasets"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / f"inference_{uuid.uuid4()}_{file.filename}"
        try:
            content = await file.read()
            with open(temp_path, 'wb') as f:
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
        original_filename = file.filename

    temp_dataset = Dataset(
        name=f"{'Sample' if is_sample else 'Uploaded'}: {original_filename}",
        filename=original_filename,
        file_path=str(temp_path),
        file_size=len(content),
        status=DatasetStatus.READY,
        uploaded_by="sample_data" if is_sample else "dataset_upload",
        assigned_org_id=organisation_id,
    )
    db.add(temp_dataset)
    await db.commit()
    await db.refresh(temp_dataset)

    from src.dashboard.schemas import InferenceConfig
    db_job = InferenceJob(
        name=name,
        model_id=model_id,
        dataset_id=temp_dataset.id,
        organisation_id=organisation_id,
        config=InferenceConfig(
            threshold=threshold,
            batch_size=batch_size,
            detection_mode=detection_mode,
            manual_overrides=manual_overrides,
        ).model_dump(),
        status=InferenceStatus.RUNNING,
        started_at=datetime.utcnow(),
    )
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)

    from src.dashboard.services.inference_runner import InferenceRunner
    runner = InferenceRunner()

    try:
        await runner.run_inference_job(
            db_job.id, db, requested_mode=detection_mode, manual_overrides=manual_overrides
        )
        job_result = await db.execute(select(InferenceJob).where(InferenceJob.id == db_job.id))
        db_job = job_result.scalar_one()
        model_result = await db.execute(select(GlobalModel).where(GlobalModel.id == db_job.model_id))
        db_model = model_result.scalar_one()
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == db_job.dataset_id))
        db_dataset = dataset_result.scalar_one()
        return InferenceJobDetail(
            id=db_job.id,
            name=db_job.name,
            model_id=db_job.model_id,
            dataset_id=db_job.dataset_id,
            organisation_id=db_job.organisation_id,
            status=db_job.status,
            config=db_job.config,
            total_records=db_job.total_records,
            benign_count=db_job.benign_count,
            malicious_count=db_job.malicious_count,
            processing_time=db_job.processing_time,
            results_path=db_job.results_path,
            error_message=db_job.error_message,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
            created_at=db_job.created_at,
            model=GlobalModelResponse.model_validate(db_model, from_attributes=True),
            dataset=DatasetResponse.model_validate(db_dataset, from_attributes=True),
        )
    except Exception as e:
        if not is_sample and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


# Dataset Analysis & Universal Intelligence endpoints
async def _analyze_dataset_file(
    file_path: Path,
    original_filename: str,
    uploaded_by: str,
    db: AsyncSession,
) -> DatasetAnalysisResponse:
    """Common universal dataset ingestion, profiling, and database registration."""
    from src.inference.pipeline import get_adaptive_pipeline
    pipeline = get_adaptive_pipeline()

    df, meta, profile = pipeline.profiler.profile_file(str(file_path))
    if len(df) == 0:
        raise HTTPException(status_code=400, detail=f"Dataset '{original_filename}' is empty (0 valid rows).")

    # JSON-safe sample rows (replace NaN/Inf with None)
    sample_df = df.head(10).replace([float("inf"), float("-inf")], None)
    sample_rows = json.loads(sample_df.to_json(orient="records"))

    dataset = Dataset(
        name=f"{'Sample' if uploaded_by == 'sample_data' else 'Analyzed'}: {original_filename}",
        filename=original_filename,
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        status=DatasetStatus.READY,
        rows=profile.total_rows,
        columns=profile.total_columns,
        feature_names=list(df.columns),
        label_distribution=profile.label_distribution,
        missing_values={
            k: int(v.get("null_count", 0))
            for k, v in profile.column_stats.items()
            if v.get("null_count", 0) > 0
        },
        duplicate_rows=profile.duplicate_rows,
        uploaded_by=uploaded_by,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return DatasetAnalysisResponse(
        dataset_id=dataset.id,
        filename=original_filename,
        rows=profile.total_rows,
        columns=profile.total_columns,
        feature_names=list(df.columns),
        sample_rows=sample_rows,
        column_stats=profile.column_stats,
        label_distribution=profile.label_distribution,
        has_label_column=profile.is_labeled,
        primary_label_column=profile.primary_label_column,
        candidate_label_columns=profile.candidate_label_columns,
        file_size_bytes=profile.file_size_bytes,
        memory_footprint_mb=profile.memory_footprint_mb,
        detected_format=profile.detected_format,
        detected_encoding=profile.detected_encoding,
        detected_schema_family=profile.detected_schema_family,
        schema_family_confidence=profile.schema_family_confidence,
        quality_score=profile.quality_score,
        quality_breakdown=profile.quality_breakdown,
        valid_row_count=profile.valid_row_count,
        invalid_row_count=profile.invalid_row_count,
        duplicate_rows=profile.duplicate_rows,
        total_missing_cells=profile.total_missing_cells,
        total_infinite_cells=profile.total_infinite_cells,
        column_types=profile.column_types,
        type_counts=profile.type_counts,
        identifier_columns=profile.identifier_columns,
        empty_columns=profile.empty_columns,
        constant_columns=profile.constant_columns,
        network_related_columns=profile.network_related_columns,
        warnings=profile.warnings,
    )


@app.post("/api/inference/analyze-csv", response_model=DatasetAnalysisResponse)
async def analyze_csv_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a network dataset file (CSV, TSV, JSON, JSONL, Parquet) and profile its schema."""
    from src.inference.ingestion import UniversalIngestionEngine
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in UniversalIngestionEngine.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported: {sorted(UniversalIngestionEngine.SUPPORTED_EXTENSIONS)}",
        )

    upload_dir = PROJECT_ROOT / "artifacts" / "uploaded_datasets"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"analyze_{uuid.uuid4()}_{file.filename}"

    try:
        content = await file.read()
        with open(temp_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        return await _analyze_dataset_file(temp_path, file.filename or temp_path.name, "csv_analysis", db)
    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Analysis failed: {e}")


@app.post("/api/inference/analyze-sample", response_model=DatasetAnalysisResponse)
async def analyze_sample_dataset(
    sample_file: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Analyze a pre-generated sample dataset for inference."""
    sample_path = MOCK_DIR / sample_file
    if not sample_path.exists():
        alt_path = PROJECT_ROOT / sample_file
        if alt_path.exists():
            sample_path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"Sample file not found: {sample_file}")

    try:
        return await _analyze_dataset_file(sample_path, sample_file, "sample_data", db)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {e}")


@app.get("/api/inference/profile/{dataset_id}")
async def get_dataset_profile(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get comprehensive Phase 3 DatasetProfile for an uploaded/analyzed dataset."""
    dataset = await db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if not dataset or not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    from src.inference.pipeline import get_adaptive_pipeline
    pipeline = get_adaptive_pipeline()
    _, _, profile = pipeline.profiler.profile_file(dataset.file_path)
    return {"dataset_id": str(dataset.id), **profile.to_dict()}


@app.get("/api/inference/preview/{dataset_id}", response_model=DatasetPreviewResponse)
async def preview_dataset(
    dataset_id: uuid.UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None),
    sort_by: Optional[str] = Query(None),
    sort_order: str = Query("asc"),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated, searchable, sortable preview of a dataset with column types & missing counts."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        rel_name = os.path.basename(dataset.file_path)
        candidates = [
            f"data/mock/inference/{rel_name}",
            f"/app/data/mock/inference/{rel_name}",
            f"artifacts/uploaded_datasets/{rel_name}",
            f"/app/artifacts/uploaded_datasets/{rel_name}",
        ]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                dataset.file_path = cand
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        import numpy as np
        from src.inference.ingestion import get_ingestion_engine
        df, meta = get_ingestion_engine().load_dataframe(dataset.file_path)

        total_rows = len(df)
        total_columns = len(df.columns)
        missing_counts = {c: int(df[c].isna().sum()) for c in df.columns}

        work_df = df
        if search and search.strip():
            q = search.strip().lower()
            mask = work_df.astype(str).apply(lambda col: col.str.lower().str.contains(q, na=False)).any(axis=1)
            work_df = work_df[mask]

        if sort_by and sort_by in work_df.columns:
            ascending = sort_order.lower() != "desc"
            work_df = work_df.sort_values(by=sort_by, ascending=ascending, na_position="last")

        filtered_rows = len(work_df)
        total_pages = max(1, (filtered_rows + page_size - 1) // page_size)

        offset = (page - 1) * page_size
        page_df = work_df.iloc[offset : offset + page_size].replace([np.inf, -np.inf], None)
        rows = json.loads(page_df.to_json(orient="records"))

        return DatasetPreviewResponse(
            dataset_id=dataset.id,
            total_rows=total_rows,
            filtered_rows=filtered_rows,
            total_columns=total_columns,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            columns=list(df.columns),
            column_types=meta.column_types,
            missing_counts=missing_counts,
            rows=rows,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {e}")


@app.get("/api/inference/validate/{dataset_id}", response_model=ValidationReportResponse)
async def validate_dataset_for_inference(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Validate a dataset using the Universal Dataset Intelligence & Adaptive Routing Engine."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        rel_name = os.path.basename(dataset.file_path)
        candidates = [
            f"data/mock/inference/{rel_name}",
            f"/app/data/mock/inference/{rel_name}",
            f"artifacts/uploaded_datasets/{rel_name}",
            f"/app/artifacts/uploaded_datasets/{rel_name}",
        ]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                dataset.file_path = cand
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        from src.inference.pipeline import get_adaptive_pipeline
        pipeline = get_adaptive_pipeline()
        overrides = _get_dataset_overrides(dataset)
        _, _, _, _, _, _, val_report = pipeline.inspect_dataset(
            dataset.file_path,
            dataset_id=str(dataset.id),
            manual_overrides=overrides,
        )
        return ValidationReportResponse(**val_report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")


@app.get("/api/inference/feature-mapping/{dataset_id}")
async def get_dataset_feature_mapping(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get Phase 4 Semantic Feature Mapping & Phase 6 Feature Derivation report for a dataset."""
    dataset = await db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if not dataset or not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    from src.inference.pipeline import get_adaptive_pipeline
    pipeline = get_adaptive_pipeline()
    overrides = _get_dataset_overrides(dataset)
    df, _, profile, mapping, derivation, compatibility, _ = pipeline.inspect_dataset(
        dataset.file_path,
        dataset_id=str(dataset.id),
        manual_overrides=overrides,
    )
    return {
        "dataset_id": str(dataset.id),
        "uploaded_columns": list(df.columns),
        "manual_overrides": overrides,
        "mapping_report": mapping.to_dict(),
        "derivation_summary": derivation.to_dict(),
        "compatibility": compatibility.to_dict(),
    }


@app.post("/api/inference/feature-mapping/{dataset_id}")
async def update_dataset_feature_mapping(
    dataset_id: uuid.UUID,
    payload: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
):
    """Save manual feature mapping overrides (canonical_id -> source_column) and recompute compatibility."""
    dataset = await db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if not dataset or not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    new_overrides = payload.get("manual_overrides", {})
    if not isinstance(new_overrides, dict):
        raise HTTPException(status_code=400, detail="manual_overrides must be a dictionary")

    cleaned_overrides = {str(k): str(v) for k, v in new_overrides.items() if v}
    existing_mv = dict(dataset.missing_values) if isinstance(dataset.missing_values, dict) else {}
    existing_mv["__manual_overrides__"] = cleaned_overrides
    dataset.missing_values = existing_mv
    await db.commit()
    await db.refresh(dataset)

    from src.inference.pipeline import get_adaptive_pipeline
    pipeline = get_adaptive_pipeline()
    df, _, _, mapping, derivation, compatibility, val_report = pipeline.inspect_dataset(
        dataset.file_path,
        dataset_id=str(dataset.id),
        manual_overrides=cleaned_overrides,
    )
    return {
        "dataset_id": str(dataset.id),
        "uploaded_columns": list(df.columns),
        "manual_overrides": cleaned_overrides,
        "mapping_report": mapping.to_dict(),
        "derivation_summary": derivation.to_dict(),
        "compatibility": compatibility.to_dict(),
        "validation": val_report,
    }


@app.get("/api/inference/compatibility/{dataset_id}")
async def get_dataset_compatibility(
    dataset_id: uuid.UUID,
    requested_mode: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Get Phase 7 Model Compatibility & Detection Mode Routing evaluation for a dataset."""
    dataset = await db.scalar(select(Dataset).where(Dataset.id == dataset_id))
    if not dataset or not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")

    from src.inference.pipeline import get_adaptive_pipeline
    pipeline = get_adaptive_pipeline()
    overrides = _get_dataset_overrides(dataset)
    _, _, _, _, _, compatibility, _ = pipeline.inspect_dataset(
        dataset.file_path,
        dataset_id=str(dataset.id),
        manual_overrides=overrides,
        requested_mode=requested_mode,
    )
    return {"dataset_id": str(dataset.id), **compatibility.to_dict()}


@app.post("/api/inference/preprocess/{dataset_id}")
async def preprocess_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Preprocess a dataset using the Adaptive Feature Derivation & Scaling Engine."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        rel_name = os.path.basename(dataset.file_path)
        candidates = [
            f"data/mock/inference/{rel_name}",
            f"/app/data/mock/inference/{rel_name}",
            f"artifacts/uploaded_datasets/{rel_name}",
            f"/app/artifacts/uploaded_datasets/{rel_name}",
        ]
        found = False
        for cand in candidates:
            if os.path.exists(cand):
                dataset.file_path = cand
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        from src.inference.pipeline import get_adaptive_pipeline
        from src.dashboard.services.preprocessing import get_preprocessing_loader

        pipeline = get_adaptive_pipeline()
        preprocessing = get_preprocessing_loader()
        overrides = _get_dataset_overrides(dataset)

        df, meta, profile, mapping, derivation, compatibility, _ = pipeline.inspect_dataset(
            dataset.file_path,
            dataset_id=str(dataset.id),
            manual_overrides=overrides,
        )

        if compatibility.selected_mode in ("supervised", "hybrid"):
            X_scaled, imputation_notes = pipeline.derivation_engine.build_cicids70_scaled_matrix(
                derivation, preprocessing.scaler
            )
            scaler_type = "StandardScaler (CIC-IDS2017 70-feature pipeline)"
            out_shape = list(X_scaled.shape)
            features_processed = 70
        else:
            n_feat = max(len(derivation.available_canonical_features), profile.type_counts.get("numeric", 0))
            scaler_type = "RobustScaler (Median/IQR Adaptive Pipeline)"
            out_shape = [profile.total_rows, n_feat]
            features_processed = n_feat
            imputation_notes = {}

        return {
            "features_processed": features_processed,
            "scaler_type": scaler_type,
            "constant_features_removed": len(profile.constant_columns) + len(preprocessing.constant_features),
            "shape": out_shape,
            "constant_features": profile.constant_columns or preprocessing.constant_features,
            "selected_mode": compatibility.selected_mode,
            "selected_model_name": compatibility.selected_model_name,
            "compatibility_score": compatibility.compatibility_score,
            "detected_schema_family": profile.detected_schema_family,
            "exact_matches": mapping.exact_match_count,
            "semantic_matches": mapping.semantic_match_count,
            "derived_features_count": len(derivation.derived_features_list),
            "derived_features": derivation.derived_features_list,
            "neutralized_features_count": len(imputation_notes),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")


# Detection History endpoints
@app.get("/api/organisations/{org_id}/detection-history", response_model=List[DetectionHistoryResponse])
async def get_detection_history(
    org_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(DetectionHistory)
        .where(DetectionHistory.organisation_id == org_id)
        .order_by(desc(DetectionHistory.timestamp))
        .limit(limit)
    )
    result = await db.execute(query)
    return result.scalars().all()


# Training History endpoints
@app.get("/api/organisations/{org_id}/training-history", response_model=List[TrainingHistoryResponse])
async def get_training_history(
    org_id: uuid.UUID,
    experiment_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(TrainingHistory).where(TrainingHistory.organisation_id == org_id)
    if experiment_id:
        query = query.where(TrainingHistory.experiment_id == experiment_id)
    query = query.order_by(TrainingHistory.round_number)
    result = await db.execute(query)
    return result.scalars().all()


# Privacy Records endpoints
@app.get("/api/experiments/{exp_id}/privacy", response_model=List[PrivacyRecordResponse])
async def get_privacy_records(exp_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    query = select(PrivacyRecord).where(PrivacyRecord.experiment_id == exp_id)
    result = await db.execute(query)
    return result.scalars().all()


# System Events endpoints
@app.get("/api/events", response_model=List[SystemEventResponse])
async def get_system_events(
    limit: int = Query(100, ge=1, le=500),
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(SystemEvent).order_by(desc(SystemEvent.created_at)).limit(limit)
    if severity:
        query = query.where(SystemEvent.severity == severity)
    if event_type:
        query = query.where(SystemEvent.event_type == event_type)
    result = await db.execute(query)
    return result.scalars().all()


# Incident endpoints
@app.post("/api/incidents", response_model=IncidentResponse, status_code=201)
async def create_incident(incident: IncidentCreate, db: AsyncSession = Depends(get_db)):
    db_incident = Incident(**incident.model_dump())
    db.add(db_incident)
    await db.commit()
    await db.refresh(db_incident)
    return db_incident


@app.get("/api/incidents", response_model=PaginatedResponse)
async def list_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    severity: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(Incident)
    if status:
        query = query.where(Incident.status == status)
    if severity:
        query = query.where(Incident.severity == severity)
    query = query.order_by(desc(Incident.created_at))

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    items = result.scalars().all()

    # Convert ORM models to Pydantic schemas
    item_schemas = [IncidentResponse.model_validate(item) for item in items]

    return PaginatedResponse(
        items=item_schemas,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/incidents/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident


@app.patch("/api/incidents/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: uuid.UUID, incident_update: IncidentUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    update_data = incident_update.model_dump(exclude_unset=True)
    if "status" in update_data and update_data["status"] == "resolved":
        incident.resolved_at = datetime.utcnow()
    for field, value in update_data.items():
        setattr(incident, field, value)

    await db.commit()
    await db.refresh(incident)
    return incident


# Detections endpoints
@app.get("/api/detections", response_model=PaginatedResponse)
async def list_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: Optional[str] = None,
    model_id: Optional[uuid.UUID] = None,
    dataset_id: Optional[uuid.UUID] = None,
    organisation_id: Optional[uuid.UUID] = None,
    prediction: Optional[int] = None,
    min_probability: Optional[float] = None,
    max_probability: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all detection results with advanced filtering."""
    query = select(DetectionResult).join(InferenceJob).join(GlobalModel).join(Dataset)

    if search:
        pattern = f"%{search}%"
        query = query.where(
            or_(
                InferenceJob.name.ilike(pattern),
                GlobalModel.name.ilike(pattern),
                Dataset.name.ilike(pattern),
            )
        )
    if model_id:
        query = query.where(InferenceJob.model_id == model_id)
    if dataset_id:
        query = query.where(InferenceJob.dataset_id == dataset_id)
    if organisation_id:
        query = query.where(InferenceJob.organisation_id == organisation_id)
    if prediction is not None:
        query = query.where(DetectionResult.prediction == prediction)
    if min_probability is not None:
        query = query.where(DetectionResult.probability >= min_probability)
    if max_probability is not None:
        query = query.where(DetectionResult.probability <= max_probability)
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            query = query.where(DetectionResult.timestamp >= start_dt)
        except ValueError:
            pass
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
            query = query.where(DetectionResult.timestamp <= end_dt)
        except ValueError:
            pass

    query = query.order_by(desc(DetectionResult.timestamp), DetectionResult.record_index)

    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    items = result.scalars().all()

    # Convert to response format with related data
    serialized_items = []
    for item in items:
        job_result = await db.execute(
            select(InferenceJob).options(
                selectinload(InferenceJob.model),
                selectinload(InferenceJob.dataset),
                selectinload(InferenceJob.organisation),
            ).where(InferenceJob.id == item.inference_job_id)
        )
        job = job_result.scalar_one()
        unpacked = _unpack_detection_record(item)
        org_name = job.organisation.name if job.organisation else None
        if not org_name and job.dataset and getattr(job.dataset, "assigned_organisation", None):
            org_name = job.dataset.assigned_organisation.name
        if not org_name:
            org_name = "Cyber Defense Consortium"
        serialized_items.append({
            **unpacked,
            "id": str(item.id),
            "inference_job_id": str(item.inference_job_id),
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
            "job_name": job.name,
            "model_name": job.model.name if job.model else unpacked.get("model_used"),
            "model_version": job.model.version if job.model else None,
            "dataset_name": job.dataset.name if job.dataset else None,
            "organisation_id": str(job.organisation_id) if job.organisation_id else None,
            "organisation_name": org_name,
        })

    return PaginatedResponse(
        items=serialized_items,
        total=total or 0,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total else 0,
    )


@app.get("/api/inference/{job_id}/export")
async def export_inference_results_csv(
    job_id: uuid.UUID,
    prediction: Optional[int] = None,
    include_features: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """Export inference results as CSV with rich Phase 9 detection fields appended."""
    import io
    import csv
    from fastapi.responses import StreamingResponse

    job_result = await db.execute(
        select(InferenceJob).options(
            selectinload(InferenceJob.model),
            selectinload(InferenceJob.dataset),
        ).where(InferenceJob.id == job_id)
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Inference job not found")

    query = select(DetectionResult).where(DetectionResult.inference_job_id == job_id)
    if prediction is not None:
        query = query.where(DetectionResult.prediction == prediction)
    query = query.order_by(DetectionResult.record_index)

    result = await db.execute(query)
    detections = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        'Record Index',
        'Predicted Category',
        'Prediction',
        'Severity',
        'Detection Mode',
        'Risk Score',
        'Supervised Probability',
        'Anomaly Score',
        'Rule Score',
        'Confidence',
        'True Label',
        'Triggered Rules',
        'Explanation',
    ]
    if include_features:
        feature_names = job.dataset.feature_names if job.dataset and job.dataset.feature_names else []
        headers.extend(feature_names)
    writer.writerow(headers)

    for det in detections:
        unpacked = _unpack_detection_record(det)
        rules_str = "; ".join(
            f"{r.get('name', '')} ({r.get('evidence', '')})"
            for r in (unpacked.get("triggered_rules") or [])
        )
        sup_prob = unpacked.get("supervised_probability")
        anom_sc = unpacked.get("anomaly_score")
        row = [
            det.record_index,
            unpacked.get("predicted_category", ""),
            'Flagged' if det.prediction == 1 else 'Normal',
            unpacked.get("severity", ""),
            unpacked.get("detection_mode", ""),
            f"{unpacked.get('risk_score', det.probability):.6f}",
            f"{sup_prob:.6f}" if sup_prob is not None else '',
            f"{anom_sc:.6f}" if anom_sc is not None else '',
            f"{unpacked.get('rule_score', 0.0):.6f}",
            f"{det.confidence:.6f}" if det.confidence is not None else '',
            'Attack' if det.true_label == 1 else ('Benign' if det.true_label == 0 else ''),
            rules_str,
            unpacked.get("explanation", ""),
        ]
        if include_features:
            raw_f = unpacked.get("features") or {}
            feature_names = job.dataset.feature_names if job.dataset and job.dataset.feature_names else []
            for fname in feature_names:
                row.append(raw_f.get(fname, ''))
        writer.writerow(row)

    output.seek(0)
    filename = f"inference_results_{job.name}_{job_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# WebSocket endpoint
@app.websocket("/ws/{org_id}")
async def websocket_endpoint(websocket: WebSocket, org_id: str):
    await manager.connect(websocket, org_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or handle messages
            await websocket.send_json({
                "type": "ack",
                "data": {"received": data},
                "timestamp": datetime.utcnow().isoformat(),
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, org_id)


# WebSocket for general updates
@app.websocket("/ws")
async def websocket_general(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "ack",
                "data": {"received": data},
                "timestamp": datetime.utcnow().isoformat(),
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)