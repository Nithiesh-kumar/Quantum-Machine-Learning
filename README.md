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
