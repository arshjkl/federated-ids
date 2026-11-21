# FedIDS Frontend Dashboard (`frontend/`)

React 18 + TypeScript + Vite + TailwindCSS web interface for the **Federated Learning Intrusion Detection System (FedIDS)** and **Universal Network Dataset Intelligence & Adaptive Detection Engine**.

For the full visual and functional walkthrough of every screen, wizard step, and API endpoint, see [`../docs/DASHBOARD_INTERFACE.md`](../docs/DASHBOARD_INTERFACE.md).

---

## Quick Start

```bash
# From the repository root:
./start_backend.sh   # Starts FastAPI on http://localhost:18003
./start_frontend.sh  # Starts Vite dev server on http://localhost:3000 (or next free port)
```

Or run directly inside `frontend/`:

```bash
cd frontend
npm install
npm run dev          # Starts development server with hot module replacement
npx tsc --noEmit     # Runs TypeScript type checking
npm run build        # Builds production bundle into frontend/dist/
```

---

## Directory Structure

```
frontend/
├── .env                          # Vite environment config (proxies /api, /health, /ws to :18003)
├── index.html                    # Root HTML shell
├── package.json                  # Dependencies & scripts
├── tailwind.config.js            # Dark/light cyber-analytics color palette
├── vite.config.ts                # Dev server proxy to http://localhost:18003
└── src/
    ├── App.tsx                   # Router & sidebar navigation layout
    ├── main.tsx                  # React root mount
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.tsx       # Left navigation sidebar
    │   │   └── Header.tsx        # Top bar with theme switcher
    │   └── providers/
    │       └── ThemeProvider.tsx # Dark / Light / System theme provider
    ├── hooks/
    │   ├── useDashboard.ts       # React Query hooks (stats, orgs, models, datasets, jobs, experiments)
    │   └── useTheme.ts           # Theme state hook
    ├── pages/
    │   ├── Dashboard.tsx         # Executive KPI overview & system health
    │   ├── Organisations.tsx     # Federated client organisation manager
    │   ├── Experiments.tsx       # FedAvg / FedProx / DP-FedAvg training controller
    │   ├── Models.tsx            # Global & federated model registry
    │   ├── Inference.tsx         # 6-step Universal Dataset Intelligence & 18-Stage Adaptive Pipeline UI
    │   ├── Detections.tsx        # Threat record explorer, model/dataset filtering, record inspector & CSV export
    │   ├── Privacy.tsx           # Differential privacy (ε, δ) budget analytics
    │   └── Settings.tsx          # Appearance, API, WebSocket & database settings
    └── services/
        └── api.ts                # Typed Axios client for all FastAPI endpoints
```
