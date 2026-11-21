# Deployment Guide: Vercel (Frontend) & Railway (Backend) with GitHub

This guide documents the complete process for deploying the **Federated Intrusion Detection System (Federated IDS)** to **Vercel** (React SPA dashboard) and **Railway** (FastAPI ML & Federated Training Backend) using **GitHub**.

---

## 1. Architecture & Deployment Topology

```
┌────────────────────────────────┐         HTTPS / WSS          ┌────────────────────────────────┐
│         Vercel (Edge)          │ ───────────────────────────> │        Railway (Cloud)         │
│  React 18 + Vite Dashboard     │                              │   FastAPI + PyTorch Backend    │
│  SPA Routing (`vercel.json`)   │ <─────────────────────────── │   `uvicorn --port ${PORT}`     │
└────────────────────────────────┘                              └────────────────────────────────┘
                 │                                                               │
                 │ Git Push to GitHub                                            │ Railway PostgreSQL / SQLite
                 ▼                                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   GitHub Repository                                            │
│  /frontend  ───────────> Root Directory for Vercel Project                                     │
│  /          ───────────> Root Directory for Railway Project (`Dockerfile` & `railway.json`)   │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Railway Deployment (Backend API & Training Engine)

Railway deploys the FastAPI backend using Docker with automatic port binding, background training, and PyTorch inference.

### Step 2.1 — Connect GitHub Repository to Railway
1. Log in to [Railway](https://railway.app/).
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Select this repository.
4. Railway will automatically detect `railway.json` and `Dockerfile` in the root directory.

### Step 2.2 — Configure Environment Variables on Railway
In Railway under **Project Settings $\rightarrow$ Variables**, configure:

| Variable | Recommended Value | Description |
|---|---|---|
| `PORT` | *(Provided automatically by Railway)* | Dynamic HTTP port |
| `DATABASE_URL` | `postgresql+asyncpg://...` or `sqlite+aiosqlite:///./federated_ids.db` | Auto-converts Railway PostgreSQL `postgres://` URLs to async |
| `CORS_ORIGINS` | `https://your-app.vercel.app,http://localhost:3000` | Allowed origins for API and WebSocket connections |
| `DB_ECHO` | `false` | Disable verbose SQL query logging in production |
| `PYTHONUNBUFFERED` | `1` | Ensures real-time console and training logs |

### Step 2.3 — Add PostgreSQL (Optional)
If you want persistent managed cloud database storage rather than SQLite:
1. In your Railway project, click **New** $\rightarrow$ **Database** $\rightarrow$ **PostgreSQL**.
2. Railway provides `DATABASE_URL`. The backend automatically converts `postgres://` to `postgresql+asyncpg://` on startup in `src/dashboard/db/session.py`.
3. Tables and default organisations are automatically seeded on startup via `lifespan()`.

### Step 2.4 — Verify Backend Health
Once deployed, open Railway's generated domain (e.g. `https://federated-ids-production.up.railway.app`):
```bash
curl https://your-backend.up.railway.app/api/health
```
Response:
```json
{"status":"healthy","timestamp":"2026-11-21T...","version":"1.0.0"}
```

---

## 3. Vercel Deployment (React Frontend Dashboard)

The frontend is a modern React 18 SPA styled with Tailwind CSS, utilizing `@tanstack/react-query`, `lucide-react`, and `recharts`.

### Step 3.1 — Create Vercel Project
1. Log in to [Vercel](https://vercel.com/).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Import your GitHub repository.

### Step 3.2 — Configure Build Settings
In the Vercel project configuration:
- **Framework Preset**: `Vite`
- **Root Directory**: `frontend` *(Click Edit and select the `frontend` folder)*
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`

### Step 3.3 — Configure Environment Variables on Vercel
In the Vercel **Environment Variables** panel, add:

| Variable | Example Value | Description |
|---|---|---|
| `VITE_API_URL` | `https://your-backend.up.railway.app` | Railway backend public URL |
| `VITE_WS_URL` | `wss://your-backend.up.railway.app` | WebSocket endpoint for live training rounds & stages |

> **Note on SPA Routing**: The included [frontend/vercel.json](file:///Users/admin/Desktop/federated-ids%202/frontend/vercel.json) automatically routes all browser navigation paths (`/inference`, `/experiments`, `/detections`, `/models`) to `index.html` to prevent 404 errors on browser refresh.

---

## 4. Datasets Included in the Repository

All original mock datasets have been preserved and supplemented with the latest Kaggle intrusion datasets:

1. **`data/mock/inference/kaggle_cybersecurity_intrusion_2025.csv`**:
   - 9,537 network flow records with 9 features: `Network Packet Size`, `Protocol Type`, `Login Attempts`, `Session Duration`, `Encryption Used`, `IP Reputation Score`, `Failed Logins`, `Browser/User-Agent Anomaly`, `Intrusion Detected (Label)`.
2. **`data/mock/inference/kaggle_unsw_nb15_cloud_ids_2025.csv`**:
   - 2,000 cloud network flow records with 20 behavioral attributes.
3. **`data/mock/inference/kaggle_threat_intel_malicious_ips.csv`**:
   - 200 threat intelligence IP records.
4. **Preserved Existing Mock Datasets**:
   - `mock_inference_batch_01.csv` through `06.csv`
   - `mock_cic_ids_5000_labelled.csv`
   - `mock_cic_ids_5000.csv`
   - `test_inference.csv`

---

## 5. Universal Dataset Training in the Experiments Section

Users can train centralized or federated models on **any dataset** with an arbitrary schema:

1. Navigate to **Experiments** $\rightarrow$ Click **New Experiment**.
2. Select your algorithm (`Centralized Improved MLP`, `FedAvg`, `FedProx`, `DP-FedAvg`, `DP-FedProx`).
3. Select **Training Dataset**:
   - Choose any preloaded Kaggle dataset, OR
   - Select any uploaded dataset uploaded through the dashboard.
4. Select 1 organisation (for Centralized / single-tenant models) or 2+ organisations (for Federated consortium training).
5. Click **Create** $\rightarrow$ Click **Start**.
6. The backend PyTorch engine:
   - Fits the `UniversalDatasetEncoder` on the dataset schema.
   - Dynamically instantiates `CustomSchemaMLP(D)` for dimension $D$.
   - Executes multi-round federated training with local client shards.
   - Computes local & global loss, accuracy, precision, recall, and F1 score.
   - Saves checkpoint in `models/custom_trained/exp_<id>_model.pth`.
   - Registers a new active `GlobalModel` in the database.
7. Click **Detect** on the completed experiment row to immediately launch adaptive inference using the trained model on similar-schema network data.

---

## 6. Multi-Organisation Attribution in Detections & Inference

- **Detections Page**:
  - Displays a dedicated **Organisation** column showing which consortium member owned or ran each detection.
  - Filter detections by specific organisation using the **All Organisations** dropdown filter.
  - View detailed organisation attribution in the **Detection Details** investigation modal.
- **Inference Page**:
  - Run inference on behalf of any sponsoring organisation.
  - Automated fallback ensures single-tenant and consortium detection jobs always carry valid organizational provenance.

---

## 7. Verification Commands

Test your local or staging setup before pushing:
```bash
# 1. Run Universal Dataset Pipeline test suite
./.venv/bin/python -m unittest tests/test_universal_pipeline.py

# 2. Build the production React frontend
cd frontend && npm run build

# 3. Test backend health
curl http://localhost:18003/api/health
```
