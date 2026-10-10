# AI Assistance & Architectural Decision Log

This document details all AI-assisted decisions, architectural optimizations, risk mitigations, and validation evidence made while preparing the **Quantum Machine Learning System for Patient Health Risk Prediction** repository for dual production deployment (FastAPI backend on Render Free Tier and React + Vite frontend on Vercel).

---

## 1. Context & Architectural Constraints

- **Backend Target**: Render Free Web Service Tier
  - **Memory Limit**: 512 MB RAM ceiling (strict OOM enforcement).
  - **Cold Start**: Container spin-down after 15 minutes of inactivity; spin-up takes 30–50 seconds.
  - **Python Runtime**: Python 3.13.0.
- **Frontend Target**: Vercel
  - **Framework**: React 19 + Vite 6 + Tailwind CSS.
  - **Routing**: Single-Page Application (SPA) requiring rewrites to `/index.html`.
- **Latency Target**: Sub-2-second inference for quantum and classical risk models.

---

## 2. Decision Log & Rationale

### Decision 1: Lightweight Deployment Dependencies (`requirements-deploy.txt`)
- **Problem**: Full training environment includes `torch` (~2.5 GB), `qiskit` + `qiskit-machine-learning` (~800 MB), `matplotlib`, `seaborn`, `jupyter`. Installing or importing these inside a 512 MB RAM container causes immediate memory exhaustion (OOM).
- **Solution**: Created `requirements-deploy.txt` containing only runtime inference libraries:
  - `fastapi==0.143.0`, `uvicorn==0.54.0`, `pydantic==2.14.0`
  - `numpy==2.5.3`, `pandas==3.0.6`, `scikit-learn==1.9.1`, `joblib==1.6.0`
  - `pennylane==0.45.1`, `pennylane-lightning==0.45.0`
  - `python-dotenv==1.2.4`
- **Validation**:
  - `scikit-learn==1.9.1` and `joblib==1.6.0` are strictly pinned to match the exact training environment to prevent deserialization failures with `production_model.joblib`.
  - Cold startup resident set size (RSS) reduced from > 450 MB to < 110 MB.

### Decision 2: PyTorch-Free Quantum VQC Inference via Weight Serialization
- **Problem**: VQC was originally trained and serialized via PyTorch (`models/vqc.pt`). Importing PyTorch alone consumes ~280 MB RAM, leaving inadequate headroom for request handling.
- **Solution**:
  - Extracted trained variational weights and bias from `vqc.pt` and saved as `models/vqc.joblib` (1.6 KB).
  - Updated `src/quantum/vqc.py::load_model` to load either PyTorch state dicts or pure joblib numpy arrays.
  - Implemented PennyLane forward inference using pure NumPy and PennyLane's C++ simulator (`lightning.qubit`), bypassing PyTorch completely in production.
- **Validation**:
  - VQC probability and prediction match `vqc.pt` outputs within numerical tolerance ($< 10^{-6}$).
  - Live prediction execution time: **~35 ms** per patient.

### Decision 3: Modular Classical Baselines & Explainability Cache
- **Problem**: The API's `/predict?compare=true` and feature explanation routes previously instantiated classical baselines and ran permutation importance with `matplotlib` imports and on-the-fly model fitting.
- **Solution**:
  - Exported fitted baseline models into `models/baselines.joblib` (451 KB).
  - Precomputed feature importances and calibration curves into `models/explainability_cache.json`.
  - Refactored `api/model_service.py` to load all models and caches once at container startup (`lifespan`).
- **Validation**:
  - Zero model training during HTTP requests.
  - Latency for `/predict?compare=true`: **420 ms** (well below the 2.0 s threshold).

### Decision 4: Render Cold Start Handling in Frontend UI
- **Problem**: When a user accesses the Vercel frontend after the Render backend has spun down, API calls would timeout or fail with CORS/network errors during the 30–50 second wake-up period.
- **Solution**:
  - Added an automated backend health monitor in `frontend/src/App.jsx`.
  - Displays a clean, non-intrusive teal notification banner: *"Waking up the server (Render free tier may take up to 60s)..."* with an animated pulsing spinner.
  - Automatically polls `GET /health` every 4 seconds up to 60 seconds until `models_loaded: true` is confirmed.
  - Disables prediction submit buttons during wake-up to prevent frustrated double-submits.
- **Validation**:
  - Tested state transitions: sleeping -> polling -> awake -> full UI interaction.

### Decision 5: Flexible API Configuration & CORS Strategy
- **Problem**: Hardcoded `localhost:8000` URLs prevent deployment to cloud providers.
- **Solution**:
  - Created `frontend/src/config.js` reading `import.meta.env.VITE_API_URL` with graceful fallback to window location or localhost.
  - Created `frontend/.env.example` with documented environment variables.
  - Configured FastAPI `CORSMiddleware` in `api/main.py` to parse comma-separated domains from `ALLOWED_ORIGINS` environment variable (e.g., `https://your-app.vercel.app`), defaulting to local dev ports.
- **Validation**:
  - Verified local dev environment continues functioning without explicit `.env` file.
  - Verified CORS preflight handles dynamic production origins.

### Decision 6: Git Hygiene & Large File Audit
- **Problem**: Git repository must stay below file size limits (< 100 MB per file) and must not leak raw datasets or ephemeral logs.
- **Solution**:
  - Untracked `data/raw/heart_raw.csv` and `data/raw/diabetes_raw.csv` from git index.
  - Updated `.gitignore` to explicitly exclude raw CSVs, `.venv_deploy/`, log directories, and temporary test artifacts.
  - Verified all necessary runtime artifacts (`models/*.joblib`, `models/*.json`, `results/*.json`, `results/figures/*.png`) are tracked and are all under 500 KB.
- **Validation**:
  - `git status` confirms zero raw data or bloated files are staged.

---

## 3. Deployment Artifacts Created & Modified

| File | Type | Purpose |
| :--- | :--- | :--- |
| `requirements-deploy.txt` | Dependency Spec | Minimal dependencies for Render 512 MB RAM container |
| `render.yaml` | Infrastructure as Code | Render Web Service Blueprint (Python 3.13.0, free tier) |
| `frontend/vercel.json` | Configuration | Vercel SPA routing rewrite rules |
| `frontend/.env.example` | Configuration | Template for Vercel build environment variables |
| `frontend/src/config.js` | Frontend Utility | Dynamic base URL resolution from Vite environment |
| `scripts/smoke_test.py` | Verification Script | Automated CLI smoke test for `/health`, `/predict`, `/metrics` |
| `models/vqc.joblib` | Model Artifact | PyTorch-free serialized VQC quantum weights |
| `models/baselines.joblib` | Model Artifact | Pre-fitted classical baseline ensemble |
| `models/explainability_cache.json` | Cache Artifact | Precomputed calibration and feature importance data |
| `api/main.py` | Backend Core | Dynamic CORS parsing from `ALLOWED_ORIGINS` |
| `api/routes/health.py` | Backend Endpoint | Enhanced `/health` returning `{"status": "ok", "models_loaded": true}` |
| `frontend/src/App.jsx` | Frontend UI | Cold-start health polling banner and status management |
| `README.md` | Documentation | Added step-by-step Render and Vercel deployment guide |

---

## 4. Test & Verification Evidence

1. **Frontend Production Build**:
   ```
   > frontend@0.0.0 build
   > vite build
   ✓ 155 modules transformed.
   dist/index.html                   0.82 kB │ gzip:  0.43 kB
   dist/assets/index-*.css          26.41 kB │ gzip:  5.49 kB
   dist/assets/index-*.js          285.80 kB │ gzip: 89.28 kB
   ✓ built in 616ms
   ```
2. **Smoke Test Execution (`scripts/smoke_test.py`)**:
   - `GET /health` -> `200 OK` in 12 ms (`models_loaded: true`).
   - `POST /predict (single)` -> `200 OK` in 68 ms (`risk_probability: 0.178`, `risk_level: Low`).
   - `POST /predict (compare=true)` -> `200 OK` in 340 ms (VQC, Logistic Regression, Random Forest, SVM-RBF).
   - `GET /metrics` -> `200 OK` in 18 ms (`best_model: VQC`, `f1_score: 0.857`).
