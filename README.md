# Quantum Machine Learning System for Patient Health Risk Prediction

> **MEDICAL DISCLAIMER**:  
> This software system is an educational and experimental Quantum Machine Learning prototype. It is **NOT** certified for medical diagnostics or clinical decision-making. Always consult qualified healthcare professionals.

## Overview
This repository implements an end-to-end Quantum Machine Learning (QML) benchmarking and deployment system for tabular clinical risk prediction (predicting binary risk: Low vs. High).

The system integrates:
- **Variational Quantum Classifier (VQC)** powered by PennyLane on `default.qubit` simulator.
- **Quantum Kernel Support Vector Machine (QSVC)** powered by Qiskit and `qiskit-machine-learning`.
- **Classical Baselines**: Logistic Regression, Random Forest, Support Vector Classifier, and Gradient Boosting.
- **Full-Stack Serving**: FastAPI backend with Pydantic validation schemas and a modern React + Vite + Tailwind CSS frontend.

---

## Repository Structure

```
qml-health-risk/
├── data/
│   ├── raw/                # Cached raw clinical datasets (UCI Heart, Pima Diabetes)
│   └── processed/          # Preprocessed train/val/test splits (.npz)
├── notebooks/              # Jupyter notebooks for EDA and experiment tracking
├── src/
│   ├── data/               # loader.py, preprocess.py, eda.py
│   ├── classical/          # Classical baseline model implementations
│   ├── quantum/            # vqc.py, quantum_kernel.py, feature_maps.py
│   ├── evaluation/         # metrics.py, compare.py
│   └── utils/              # config.py (registry, constants), logger.py
├── api/                    # FastAPI backend endpoints, schemas, model service
├── frontend/               # React + Vite + Tailwind frontend application
├── models/                 # Saved fitted model artifacts (*.joblib)
├── results/
│   └── eda/                # Generated EDA figures & clinical summaries
├── logs/                   # System execution logs
└── tests/                  # Pytest unit and integration test suite
```

---

## Setup & Environment

1. **Create and Activate Virtual Environment**:
   ```bash
   py -3 -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python check_env.py
   # or
   make check
   ```

4. **Run Unit Tests**:
   ```bash
   pytest tests/ -v
   ```

---

## Dataset & Preprocessing

- **Datasets Supported**:
  - `heart`: UCI Heart Disease (Cleveland Clinic, 303 samples, target binarized 0 vs $\ge 1$).
  - `diabetes`: Pima Indians Diabetes (768 samples).
- Switch datasets easily via `DATASET=heart|diabetes` environment variable or in `src/utils/config.py`.
- **Zero-Leakage Pipeline**:
  - Imputation (Median / Mode) $\rightarrow$ Standard Scaling $\rightarrow$ Feature Reduction (PCA / SelectKBest to $\le 8$ features) $\rightarrow$ MinMax Scaling to $[0, \pi]$ for quantum angle encoding.
  - Pipeline is fit strictly on the 70% training split.

---

## Deployment Guide

This project is architected for dual-cloud deployment:
- **Backend API**: Hosted on **Render** (Free Tier Web Service, 512 MB RAM ceiling, Python 3.13.0).
- **Frontend UI**: Hosted on **Vercel** (Global Edge CDN, React 19 + Vite).

### Live URLs (Placeholders)
- **Live Web Application**: [https://your-project.vercel.app](https://your-project.vercel.app)
- **Live API Documentation (Swagger)**: [https://your-api.onrender.com/docs](https://your-api.onrender.com/docs)
- **API Health Check**: [https://your-api.onrender.com/health](https://your-api.onrender.com/health)

---

### Backend Deployment (Render)

1. Create a **New Web Service** connected to your repository on [render.com](https://render.com).
2. Configure the service settings:
   - **Root Directory**: `.` (leave as root)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
3. Add the following **Environment Variables**:
   | Variable | Value | Purpose |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.13.0` | Ensures runtime compatibility |
   | `ALLOWED_ORIGINS` | `https://<your-vercel-app>.vercel.app` | CORS authorization for your frontend |

> [!NOTE]
> Alternatively, Render will automatically detect [`render.yaml`](file:///d:/Quantum%20ML/qml-health-risk/render.yaml) if deployed via Render Blueprints.

#### Render Free-Tier Cold Starts Explained
Render free-tier web services spin down automatically after 15 minutes of inactivity to conserve cloud resources. When a new visitor accesses the application after an idle period:
- The first request triggers a container wake-up, taking **30 to 50 seconds**.
- The frontend features an automated **"Waking up the server..."** status banner that polls `/health` and notifies users until the backend is fully initialized.
- Once awake, quantum and classical predictions execute in **< 500 ms**.

---

### Frontend Deployment (Vercel)

1. Import your GitHub repository on [vercel.com](https://vercel.com).
2. Set the project configuration:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Add the following **Environment Variable**:
   | Variable | Value |
   | :--- | :--- |
   | `VITE_API_URL` | `https://<your-render-app>.onrender.com` |

4. Deploy! Vercel handles single-page routing rewrites automatically via [`frontend/vercel.json`](file:///d:/Quantum%20ML/qml-health-risk/frontend/vercel.json).

---

### Local Verification & Smoke Testing

To test the deployment environment locally without bloated training packages:
```bash
# 1. Create a clean deployment virtual environment
py -3 -m venv .venv_deploy
.\.venv_deploy\Scripts\pip install -r requirements-deploy.txt

# 2. Run backend locally on port 8000
.\.venv_deploy\Scripts\uvicorn api.main:app --host 127.0.0.1 --port 8000

# 3. Run automated smoke tests
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

