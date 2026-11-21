# Complete Deployment Guide: Vercel (Frontend) + Render / Railway / Koyeb (Backend)

This guide documents how to deploy the **Federated Intrusion Detection System (Federated IDS)** using GitHub.

- **Frontend Dashboard**: Deployed on **Vercel** (Global Edge React 18 SPA).
- **Backend ML & Training Engine**: Deployed on **Render** (Recommended Free Railway Alternative) or **Railway** / **Koyeb**.

---

## Architecture & Topology

```
┌────────────────────────────────┐         HTTPS / WSS          ┌──────────────────────────────────────────────┐
│         Vercel (Edge)          │ ───────────────────────────> │     Render / Railway / Koyeb (Cloud)         │
│  React 18 + Vite Dashboard     │                              │        FastAPI + PyTorch Backend             │
│  SPA Routing (`vercel.json`)   │ <─────────────────────────── │         `uvicorn --port ${PORT}`             │
└────────────────────────────────┘                              └──────────────────────────────────────────────┘
                 │                                                                     │
                 │ Git Push to GitHub                                                  │ PostgreSQL / SQLite
                 ▼                                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       GitHub Repository: arshjkl/federated-ids                               │
│  /frontend  ───────────> Root Directory for Vercel Project (`vercel.json`)                                   │
│  /          ───────────> Root Directory for Backend (`Dockerfile`, `render.yaml`, `railway.json`)           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## OPTION A (Recommended Alternative): Deploy Backend on Render (100% Free)

[Render](https://render.com/) is the top free alternative to Railway. It natively supports Docker, custom ports (`$PORT`), FastAPI, WebSockets (`wss://`), and persistent databases.

### Step 1: Sign up & Connect GitHub
1. Go to **[render.com](https://render.com/)** and sign in with GitHub.
2. In the Render Dashboard, click **New +** $\rightarrow$ select **Web Service**.
3. Under *Connect a repository*, choose **`arshjkl/federated-ids`**.

### Step 2: Configure Service Settings
Fill in the configuration:
- **Name**: `federated-ids-backend`
- **Region**: Choose the closest to you (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
- **Language / Runtime**: **Docker** *(Render detects the root `Dockerfile` automatically)*
- **Instance Type**: **Free**

### Step 3: Add Environment Variables in Render
Scroll down to the **Environment Variables** section and add:

| Key | Value | Description |
|---|---|---|
| `PORT` | `10000` | Port for the FastAPI server |
| `DATABASE_URL` | `sqlite+aiosqlite:///./federated_ids.db` | Default SQLite or Render Postgres URL |
| `CORS_ORIGINS` | `*` | Allows your Vercel frontend domain |
| `PYTHONUNBUFFERED` | `1` | Stream real-time training and inference logs |

*(Alternatively, you can click **New +** $\rightarrow$ **Blueprint** and Render will automatically configure everything from the included `render.yaml`!)*

### Step 4: Health Check & Deploy
- **Health Check Path**: `/api/health`
- Click **Create Web Service**.
- Render will build the container, install dependencies, and launch your API at a public URL:
  `https://federated-ids-backend-xxxx.onrender.com`

---

## OPTION B: Deploy Backend on Koyeb (Fast Serverless MicroVMs)

[Koyeb](https://www.koyeb.com/) is another excellent alternative offering generous free allowances, native Docker container support, and built-in global edge routing.

1. Sign up on **[koyeb.com](https://www.koyeb.com/)** with GitHub.
2. Click **Create App** $\rightarrow$ select **GitHub**.
3. Pick **`arshjkl/federated-ids`**.
4. In **Builder**, select **Dockerfile**.
5. In **Environment Variables**, set:
   - `CORS_ORIGINS` = `*`
   - `PYTHONUNBUFFERED` = `1`
6. Click **Deploy**. Koyeb will output your public URL:
   `https://federated-ids-xxxx.koyeb.app`

---

## OPTION C: Deploy Backend on Railway

If you prefer Railway:
1. Log in to [railway.app](https://railway.app/) with GitHub.
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo** $\rightarrow$ Select `arshjkl/federated-ids`.
3. In **Variables**, add `CORS_ORIGINS` = `*` and `PYTHONUNBUFFERED` = `1`.
4. In **Settings** $\rightarrow$ **Networking**, click **Generate Domain**.

---

## Deploy Frontend to Vercel (React 18 Dashboard)

Once your backend is running on **Render** (or Koyeb/Railway), deploy the frontend to Vercel and connect them:

### Step 1: Create Vercel Project
1. Log in to **[vercel.com](https://vercel.com/)** with GitHub.
2. Click **Add New...** $\rightarrow$ **Project**.
3. Select **`arshjkl/federated-ids`**.

### Step 2: Configure Project Settings
- **Framework Preset**: `Vite`
- **Root Directory**: Click **Edit** and choose **`frontend`**
- **Build Command**: `npm run build`
- **Output Directory**: `dist`
- **Install Command**: `npm install`

### Step 3: Set Backend Environment Variables in Vercel
In the **Environment Variables** panel, add your backend URL (from Render, Koyeb, or Railway):

| Variable Name | Value (Example with Render) |
|---|---|
| `VITE_API_URL` | `https://federated-ids-backend.onrender.com` |
| `VITE_WS_URL` | `wss://federated-ids-backend.onrender.com` |

> *Note: If using HTTP, Vercel will connect via `https://` and WebSocket via `wss://` automatically.*

### Step 4: Click "Deploy"
Vercel builds your React app and deploys it worldwide at `https://federated-ids-xxxx.vercel.app`.
All pages (`/inference`, `/experiments`, `/detections`, `/models`) will work out of the box with zero 404 errors on refresh due to `frontend/vercel.json`.

---

## Verification & Testing After Deployment

1. **Verify Backend Health**:
   ```bash
   curl https://<YOUR_BACKEND_URL>/api/health
   # Returns: {"status":"healthy","database":"connected","version":"1.0.0"}
   ```
2. **Verify Frontend**:
   Open your Vercel URL in your browser:
   - **Inference**: Test running detection on any of the preloaded Kaggle datasets (`Kaggle Cybersecurity Intrusion 2025`, `UNSW-NB15`, etc.).
   - **Experiments**: Train a model on any custom dataset with 1 or more organisations. Click **Detect** to run inference with the newly trained model!
   - **Detections**: Filter detections by **Organisation** to verify multi-tenant isolation.
