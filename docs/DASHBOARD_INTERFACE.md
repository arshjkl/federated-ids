# Dashboard Interface Guide (`FedIDS` Web Platform)

The **Federated Learning Intrusion Detection System (FedIDS) Dashboard** is a full-stack React + TypeScript + FastAPI web application for managing federated network intrusion detection, inspecting global and differentially-private models, running the **18-Stage Universal Dataset Intelligence & Adaptive Detection Pipeline**, and investigating individual flow detections.

---

## 1. Tech Stack & Architecture

| Layer | Technology | Role |
|---|---|---|
| **Frontend UI** | React 18, TypeScript, Vite, TailwindCSS, Lucide Icons | Interactive 6-step adaptive inference wizard, threat explorer, federated experiment manager |
| **State & Data Fetching** | TanStack React Query (`@tanstack/react-query`), Axios | Automatic caching, live polling during 18-stage pipeline execution, optimistic mutations |
| **Backend API** | FastAPI (Python 3.12), Uvicorn, Pydantic v2 | REST API (`http://localhost:18003/api/*`), WebSocket (`ws://localhost:18003/ws`), CSV/JSON streaming exports |
| **Database & Persistence** | SQLAlchemy 2.0 (Async), `aiosqlite` (`federated_ids.db`), JSON stage artifacts | Stores organisations, datasets, global models, federated experiments, inference jobs, and detection records |
| **Proxy & CORS** | Vite Dev Server Proxy (`vite.config.ts`) + FastAPI `CORSMiddleware` | Proxies `/api`, `/health`, and `/ws` from the frontend directly to `http://localhost:18003` |

---

## 2. Starting the Dashboard

Run the backend and frontend in two terminals from the repository root:

```bash
# Terminal 1 — Start FastAPI Backend (port 18003)
./start_backend.sh

# Terminal 2 — Start React + Vite Frontend (port 3000, or next available port e.g. 3018)
./start_frontend.sh
```

- **Frontend UI**: `http://localhost:3000` (or the port printed by Vite, e.g. `http://localhost:3018`)
- **Backend OpenAPI / Swagger Docs**: `http://localhost:18003/docs`
- **Health Check**: `http://localhost:18003/health`

---

## 3. Navigation & Pages Overview

The sidebar provides direct access to 7 core modules:

```
FedIDS Sidebar
├── 1. Dashboard       (/)               — Platform KPIs, quick actions, and live backend/DB connectivity
├── 2. Organisations   (/organisations)  — Manage participating federated client organisations & local datasets
├── 3. Experiments     (/experiments)    — Configure, launch, pause, and monitor FedAvg / FedProx / DP-FedAvg training
├── 4. Models          (/models)         — Global & federated model registry with accuracy/F1 metrics
├── 5. Inference       (/inference)      — 6-step Universal Dataset Intelligence & 18-Stage Adaptive Detection Wizard
├── 6. Detections      (/detections)     — Threat record explorer, multi-filter search, record inspector, CSV export
└── 7. Settings        (/settings)       — Theme switcher (Dark / Light / System), API/WebSocket & DB configuration
```

---

## 4. Detailed Page Reference

### 4.1 Dashboard (`/`)
The landing page provides a real-time executive summary of the Federated IDS platform:
- **KPI Cards** (clickable shortcuts to each module):
  - **Organisations**: Total registered federated client nodes.
  - **Active Experiments**: Currently running federated training sessions.
  - **Inference Jobs**: Total batch detection jobs executed.
  - **Total Detections**: Cumulative network flows analyzed across all jobs.
  - **Malicious Detected**: Total flagged/attack records identified.
- **Quick Actions**: One-click launchers for *Add Organisation*, *Create Experiment*, *Run Inference*, and *View Detections*.
- **System Status**: Live health indicators for the FastAPI Backend API and SQLite Database (`federated_ids.db`).

---

### 4.2 Organisations (`/organisations`)
Manages the participating client institutions in the federation (e.g., `Client 1`, `Client 2`, regional SOC nodes):
- **Organisation Table**: Displays `Name`, `Org ID` (e.g., `org-001`), `Status` (`active`, `training`, `paused`, `inactive`), `Rounds Participated`, and `Total Samples`.
- **Add / Edit / Inspect Modal**: Register new client organisations, update their metadata, or bind a specific local `Dataset UUID` to an organisation.

---

### 4.3 Experiments (`/experiments`)
Orchestrates federated learning runs across participating organisations:
- **Experiment Controls**: Start, Pause, Resume, Stop, Inspect, or Delete federated training experiments.
- **New Experiment Modal**:
  - Select algorithm: **FedAvg**, **FedProx** (with proximal term $\mu$), **DP-FedAvg**, or **DP-FedProx** (with Differential Privacy noise multiplier $\sigma$, max gradient clipping norm $C$, and target $\delta$).
  - Configure `Total Rounds`, `Local Epochs`, `Batch Size`, and `Learning Rate`.
  - Select 2 or more participating organisations.

---

### 4.4 Global Models (`/models`)
Displays the versioned registry of trained PyTorch checkpoints (`MLPIDS` and `ImprovedMLPIDS`):
- **Pre-Registered Checkpoints**:
  1. **Federated v2 (Improved FedAvg, Round 20)** (`federated_v2_r20` | `fedavg` | `ImprovedMLPIDS` | **99.44% Acc**, **98.36% F1**) — *Default Federated Model*
  2. **Federated v2 (Improved FedAvg, Round 10)** (`federated_v2_r10` | `fedavg` | `ImprovedMLPIDS` | **99.33% Acc**, **98.03% F1**)
  3. **Manual FedAvg (50 rounds)** (`federated_manual_v1` | `fedavg` | `MLPIDS` | **95.87% Acc**, **87.70% F1**)
  4. **DP-FedAvg (30 rounds, $\varepsilon=0.1457$)** (`dp_federated_v1` | `dp_fedavg` | `MLPIDS` | **93.92% Acc**, **99.48% Recall**)
  5. **Centralized Improved MLP** (`centralized_improved_v1` | `centralized` | `ImprovedMLPIDS` | **99.72% Acc**, **99.16% F1**)
  6. **Centralized MLP Baseline** (`centralized_v1` | `centralized` | `MLPIDS` | **97.71% Acc**, **92.98% F1**)
- **Register Model Modal**: Allows registering additional `.pth` checkpoints with custom version tags, algorithms, and training round metadata.

---

### 4.5 Universal Dataset Intelligence & Adaptive Inference (`/inference`)
The core analytical workbench of the platform, organized into a transparent **6-Step Interactive Wizard** backed by the **18-Stage Adaptive Pipeline**:

#### Step 1 — Upload or Select Dataset
- **Built-in Benchmark & Synthetic Datasets**: One-click selection for *CIC-IDS2017 Batch 01 (Unlabeled)*, *CIC-IDS2017 Batch 02 (Unlabeled)*, *CIC-IDS2017 Labeled (5,000 rows)*, or *Anonymized Feature Matrix (`feature_1..70`)*.
- **Universal File Upload**: Drag-and-drop any network traffic dataset in **CSV, TSV, JSON, JSONL/NDJSON, or Parquet** format (`CIC-IDS2017/2018/2019`, `UNSW-NB15`, `NSL-KDD`, `TON_IoT`, `Edge-IIoTset`, Zeek, NetFlow, or custom schemas).

#### Step 2 — Overview & Interactive Preview
- **Universal Dataset Profile Banner**: Displays detected **Schema Family** (e.g., `CICIDS2017`, `UNSW_NB15`, `CUSTOM`), **Labeled vs. Unlabeled** status, detected file format/encoding, clean row count, memory footprint in RAM, **0–100 Data Quality Score** with 5-dimension quality breakdown bars, missing/infinite cell counts, and column type distribution.
- **Interactive Data Preview Table**: Paginated, full-text searchable, and column-sortable table showing data types and per-column null badges.
- **Column & Distribution Statistics Table**: Filterable per-column breakdown of `Role / Semantic Type`, `Min`, `Median`, `Max`, `Mean ± Std`, `3-IQR Outlier Count (%)`, and `Null / Inf` counts.

#### Step 3 — Validate & Map Features
- **Validation Status Banner & Checks Grid**: Non-rigid 6-point validation check (`Row Count`, `Schema Compatibility`, `Missing Values`, `Infinite Values`, `Duplicates & Variance`, `Label Column Status`). Datasets with partial or non-CICIDS schemas are never crashed or blindly rejected—they are flagged for adaptive routing.
- **Visual Semantic Feature Mapping Table**:
  - Shows how uploaded columns map to the **77 Canonical Network Features** (and 70 CIC-IDS2017 features).
  - Filter pills by mapping classification: **Exact Match**, **Semantic Match**, **Derived Feature**, **Approximate Match**, and **Unavailable**.
  - Displays confidence percentage (`0–100%`), feature group (`Flow`, `Directional`, `Packet Stats`, `Timing`, `Protocol`, `TCP`, `Rate`), unit conversions (e.g., `seconds_to_microseconds`), mathematical derivation formulas, and a **Manual Override** dropdown per canonical feature.

#### Step 4 — Model Routing & Checkpoint Selection
- **Global & Federated Model Selection Grid**: Interactive cards allowing you to choose which trained checkpoint to execute:
  - *Federated v2 (Improved FedAvg, Round 20)* (default)
  - *Federated v2 (Improved FedAvg, Round 10)*
  - *Manual FedAvg (50 rounds)*
  - *DP-FedAvg (30 rounds, $\varepsilon=0.1457$)*
  - *Centralized Improved MLP*
  - *Centralized MLP Baseline*
- **Adaptive Detection Mode Router**: Displays the 4 detection strategies with compatibility badges and lets you inspect or override the active mode:
  - **Mode 1 (`supervised`)**: Full 70-feature `StandardScaler` + PyTorch checkpoint classification ($\ge 90\%$ effective coverage).
  - **Mode 2 (`hybrid`)**: Weighted ensemble combining the selected PyTorch checkpoint (with neutral training-mean baseline for unavailable non-core features), `IsolationForest` + Modified Z-Score anomaly detection, and the 8-Rule Network Heuristic Engine ($\ge 55\%$ coverage).
  - **Mode 3 (`anomaly`)**: Unsupervised `IsolationForest` + Robust Modified Z-Score + Network Rules for schemas with low CIC-IDS2017 overlap (`UNSW-NB15`, `NSL-KDD`, custom IoT logs).
  - **Mode 4 (`rule_based`)**: Deterministic 8-signature network intrusion rule evaluation for minimal schemas.
- **Transparency & Technical Honesty Cards**: Explains *Why This Detection Mode Was Selected*, active ensemble weights (`Supervised %`, `Anomaly %`, `Rule %`), and *Known Limitations*.

#### Step 5 — 18-Stage Traceable Pipeline Execution
- Live execution monitor polling `/api/inference/{job_id}/stages` every second.
- Displays status (`completed`, `warning`, `running`), execution time in milliseconds, `Input → Output` record counts, and structured telemetry metadata across all 18 stages (`1. Dataset Ingestion` through `18. Export Readiness`).

#### Step 6 — Detection Analytics & Record Investigation
- **Top KPI Cards**: `Total Processed`, `Normal Records`, `Confirmed Attacks`, `Probable Suspicious`, `Statistical Anomalies`, and `Rule Alerts`.
- **Ground-Truth Evaluation Banner** (automatically shown when the uploaded dataset contains a ground-truth label column): Displays post-inference `Accuracy`, `Precision`, `Recall`, `F1 Score`, and Confusion Matrix (`TP`, `TN`, `FP`, `FN`).
- **Distribution Charts**:
  - *Severity Breakdown* (`Critical`, `High`, `Medium`, `Low`, `Informational`)
  - *Detection Category & Mode Breakdown*
  - *Risk Score & Confidence Histograms*
- **Paginated Records Table & Flow Record Investigation Modal**: Click any record to inspect its `Risk Score`, `Supervised Prob`, `Anomaly Score`, `Rule Score`, human-readable `Detection Explanation`, `Triggered Network Rules`, `Top Deviating Statistical Features`, and all original flow feature values.
- **Recent Adaptive Detection Jobs Table**: Located at the bottom of `/inference` to reload analytics for any previous job or delete old runs.

---

### 4.6 Detections Explorer (`/detections`)
A dedicated SOC threat-hunting interface across all completed inference jobs:
- **Summary KPI Bar**: `Total Loaded`, `Total Records`, `Benign`, `Malicious`, and `Threat Rate (%)`.
- **Multi-Dimensional Filter Bar**:
  - Full-text search across job name, model name, and dataset name.
  - Prediction filter (`All Predictions`, `Benign Only`, `Malicious Only`).
  - **Model filter** (`All Models`, `Federated v2 (Improved FedAvg, Round 20)`, `DP-FedAvg (30 rounds, ε=0.1457)`, `Manual FedAvg (50 rounds)`, `Centralized Improved MLP`, `Centralized MLP Baseline`).
  - **Dataset filter** (`All Datasets`).
  - **Advanced FiltersDrawer**: Date range presets (`All Time`, `Today`, `Last 7 Days`, `This Month`, `Custom Range`) and `Probability Min / Max` range sliders.
- **Detections Table**: Lists `# (Record Index)`, `Timestamp`, `Job`, `Model`, `Mode`, `Dataset`, `Predicted Category`, `Severity`, `Risk Score` (with visual progress bar), `Confidence`, and an `Inspect` action button.
- **Detection Details Modal**: Full drill-down with attack probability gauge, adaptive explanation, individual `Supervised / Anomaly / Rule` scores, triggered heuristic rule evidence, job metadata, and downloadable per-record feature JSON.
- **CSV Export**: One-click export of filtered detection records (`detections_YYYYMMDD_HHMMss.csv`).

---

### 4.7 Settings (`/settings`)
- **Appearance**: Switch between `Light`, `Dark`, and `System` themes.
- **API & Database Configuration**: Inspect active Backend REST URL (`http://localhost:18003`), WebSocket URL (`ws://localhost:18003`), and SQLite database URI (`sqlite+aiosqlite:///federated_ids.db`).

---

## 5. Backend API Endpoints Powering the Dashboard

| Endpoint | Method | Description |
|---|---|---|
| `/api/stats` | `GET` | Platform-wide KPI counts for the Dashboard overview |
| `/api/models` | `GET` / `POST` | List or register global & federated model checkpoints |
| `/api/models/compatible` | `GET` | List active models compatible with a dataset |
| `/api/inference/analyze-csv` | `POST` | Upload & profile any CSV/TSV/JSON/JSONL/Parquet dataset |
| `/api/inference/analyze-sample` | `POST` | Profile a built-in benchmark sample dataset |
| `/api/inference/preview/{dataset_id}` | `GET` | Paginated, searchable, sortable dataset row preview |
| `/api/inference/validate/{dataset_id}` | `GET` | Run 6-point schema & data quality validation |
| `/api/inference/feature-mapping/{dataset_id}` | `GET` / `POST` | Inspect semantic feature mappings or apply manual overrides |
| `/api/inference/compatibility/{dataset_id}` | `GET` | Evaluate model compatibility & 4-mode routing decision |
| `/api/inference/preprocess/{dataset_id}` | `POST` | Execute feature derivation & adaptive scaling check |
| `/api/inference/run-on-dataset` | `POST` | Launch the 18-stage adaptive inference pipeline with a selected `model_id` and `detection_mode` |
| `/api/inference/{job_id}/stages` | `GET` | Fetch real-time 18-stage execution telemetry and summary metrics |
| `/api/inference/{job_id}/results` | `GET` | Fetch enriched per-record detection results for a job |
| `/api/inference/{job_id}/records/{idx}` | `GET` | Deep-dive investigation payload for a single flow record |
| `/api/inference/{job_id}/export` | `GET` | Stream enriched detection results as a downloadable CSV |
| `/api/detections` | `GET` | Global paginated detections search across all jobs, models, and datasets |
