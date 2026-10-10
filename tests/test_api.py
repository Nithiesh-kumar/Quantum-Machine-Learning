"""Tests for FastAPI clinical risk prediction service using TestClient."""
import time
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient running startup and shutdown lifespans."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Test Data Fixtures
# ---------------------------------------------------------------------------
SAMPLE_HIGH_RISK_PATIENT = {
    "age": 63.0,
    "sex": 1,
    "cp": 4,
    "trestbps": 145.0,
    "chol": 260.0,
    "fbs": 0,
    "restecg": 2,
    "thalach": 142.0,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 2.0,
    "thal": 7.0,
}

SAMPLE_LOW_RISK_PATIENT = {
    "age": 41.0,
    "sex": 0,
    "cp": 2,
    "trestbps": 120.0,
    "chol": 195.0,
    "fbs": 0,
    "restecg": 0,
    "thalach": 178.0,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 0.0,
    "thal": 3.0,
}


# ---------------------------------------------------------------------------
# Health & Status Tests
# ---------------------------------------------------------------------------
def test_health_endpoint(client):
    """GET /health must return healthy status and loaded models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True
    assert data["loaded_models_count"] > 0
    assert "production_model" in data
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# Datasets Metadata Tests
# ---------------------------------------------------------------------------
def test_datasets_endpoint(client):
    """GET /datasets must return supported conditions and field specifications."""
    response = client.get("/datasets")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    conditions = data["supported_conditions"]
    assert any(c["dataset_key"] == "heart" for c in conditions)

    heart = next(c for c in conditions if c["dataset_key"] == "heart")
    assert len(heart["fields"]) == 13
    field_names = [f["name"] for f in heart["fields"]]
    assert "age" in field_names
    assert "trestbps" in field_names
    assert "chol" in field_names
    assert "oldpeak" in field_names

    # Check field schema completeness
    age_field = next(f for f in heart["fields"] if f["name"] == "age")
    assert age_field["type"] == "float"
    assert age_field["units"] == "years"
    assert "range" in age_field


# ---------------------------------------------------------------------------
# Prediction Endpoint Tests
# ---------------------------------------------------------------------------
def test_predict_flat_high_risk_patient(client):
    """POST /predict with flat patient fields returns high risk prediction."""
    start = time.time()
    response = client.post("/predict", json=SAMPLE_HIGH_RISK_PATIENT)
    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 2.0  # Acceptance: response in < 2 seconds

    data = response.json()
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_label"] in [0, 1]
    assert data["risk_level"] in ["Low", "Moderate", "High"]
    assert "model_used" in data
    assert "model_type" in data
    assert isinstance(data["quantum_representation"], list)
    assert len(data["quantum_representation"]) == 4
    assert len(data["top_contributing_features"]) > 0

    # Check top contributing features structure
    top_feat = data["top_contributing_features"][0]
    assert "feature" in top_feat
    assert "patient_value" in top_feat
    assert "baseline_mean" in top_feat
    assert "importance" in top_feat
    assert "directional_impact" in top_feat
    assert "medical_disclaimer" in data


def test_predict_wrapped_low_risk_patient(client):
    """POST /predict with wrapped payload returns valid response."""
    payload = {
        "patient": SAMPLE_LOW_RISK_PATIENT,
        "model_name": "production",
        "compare": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_label"] == 0
    assert data["risk_level"] in ["Low", "Moderate"]
    assert data["risk_probability"] < 0.5


def test_predict_with_compare_flag(client):
    """POST /predict with compare=True returns side-by-side predictions across all models."""
    payload = {
        **SAMPLE_HIGH_RISK_PATIENT,
        "compare": True,
    }
    start = time.time()
    response = client.post("/predict", json=payload)
    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 2.0  # Acceptance criterion: side-by-side evaluation in < 2 seconds

    data = response.json()
    assert data["comparison"] is not None
    comparison = data["comparison"]

    # Verify both quantum and classical models are present
    assert "MLP" in comparison
    assert "Random Forest" in comparison
    assert "Logistic Regression" in comparison
    assert "SVM-RBF" in comparison
    assert "VQC" in comparison

    for m_name, m_pred in comparison.items():
        assert "risk_probability" in m_pred
        assert "risk_label" in m_pred
        assert "risk_level" in m_pred
        assert "confidence_percentage" in m_pred
        assert m_pred["model_type"] in ["Classical", "Quantum"]


def test_predict_specific_vqc_model(client):
    """POST /predict with model_name='VQC' executes VQC quantum circuit."""
    payload = {
        "patient": SAMPLE_HIGH_RISK_PATIENT,
        "model_name": "VQC",
        "compare": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "VQC"
    assert data["model_type"] == "Quantum"
    assert 0.0 <= data["risk_probability"] <= 1.0


def test_predict_validation_errors(client):
    """POST /predict with invalid fields returns 422 with structured error details."""
    invalid_patient = {
        **SAMPLE_HIGH_RISK_PATIENT,
        "age": 10.0,      # ge=18.0 violated
        "trestbps": 500.0, # le=260.0 violated
        "cp": 99,         # le=4 violated
    }
    response = client.post("/predict", json=invalid_patient)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert "details" in data
    assert len(data["details"]) >= 3
    assert "medical_disclaimer" in data


# ---------------------------------------------------------------------------
# Metrics & Model Info Tests
# ---------------------------------------------------------------------------
def test_metrics_endpoint(client):
    """GET /metrics returns comparison table and model evaluation metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "comparison_table" in data
    assert len(data["comparison_table"]) == 6  # 6 models

    models_in_table = [m["model_name"] for m in data["comparison_table"]]
    assert "Logistic Regression" in models_in_table
    assert "SVM-RBF" in models_in_table
    assert "Random Forest" in models_in_table
    assert "MLP" in models_in_table
    assert "VQC" in models_in_table
    assert "QSVC" in models_in_table

    vqc_entry = next(m for m in data["comparison_table"] if m["model_name"] == "VQC")
    assert vqc_entry["model_type"] == "Quantum"
    assert vqc_entry["recall"] > 0.8
    assert vqc_entry["std_metrics"] is not None


def test_model_info_endpoint(client):
    """GET /model-info returns quantum circuit details and diagram URLs."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    # VQC specs
    assert data["quantum_vqc"]["qubits"] == 4
    assert data["quantum_vqc"]["circuit_depth"] == 2
    assert "StronglyEntanglingLayers" in data["quantum_vqc"]["ansatz"]
    assert "AngleEmbedding" in data["quantum_vqc"]["feature_map"]

    # QSVC specs
    assert data["quantum_qsvc"]["qubits"] == 4
    assert "ZZFeatureMap" in data["quantum_qsvc"]["feature_map"]

    # Production Model
    assert "production_model" in data
    assert "circuit_diagram_url" in data
    assert "figures" in data
    assert "medical_disclaimer" in data


# ---------------------------------------------------------------------------
# Static Assets & Swagger UI Tests
# ---------------------------------------------------------------------------
def test_static_circuit_diagram_access(client):
    """Static mount serves the quantum circuit diagram image."""
    response = client.get("/static/vqc_circuit.png")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/png", "application/octet-stream"]


def test_swagger_and_openapi_docs(client):
    """GET /docs and /openapi.json serve Swagger UI and OpenAPI specification."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200
    assert "swagger" in docs_resp.text.lower() or "html" in docs_resp.headers["content-type"]

    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    spec = openapi_resp.json()
    assert spec["info"]["title"] == "Quantum Machine Learning Health Risk API"
    assert "/health" in spec["paths"]
    assert "/datasets" in spec["paths"]
    assert "/predict" in spec["paths"]
    assert "/metrics" in spec["paths"]
    assert "/model-info" in spec["paths"]


# ---------------------------------------------------------------------------
# Explainability & Uncertainty Tests
# ---------------------------------------------------------------------------
def test_predict_confidence_interval_and_top3_features(client):
    """POST /predict returns exactly top-3 features and valid confidence interval."""
    response = client.post("/predict", json=SAMPLE_HIGH_RISK_PATIENT)
    assert response.status_code == 200
    data = response.json()

    # Acceptance: UI shows top-3 features
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) == 3
    for feat in data["top_contributing_features"]:
        assert "feature" in feat
        assert "patient_value" in feat
        assert "baseline_mean" in feat
        assert "importance" in feat
        assert feat["importance"] >= 0.0
        assert feat["directional_impact"] in ["Elevates Risk", "Reduces Risk"]

    # Acceptance: confidence range for every prediction
    assert "confidence_interval" in data
    ci = data["confidence_interval"]
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["confidence_level"] == 0.95
    assert ci["uncertainty_std"] > 0.0
    assert "%" in ci["range_display"]
    assert "CI" in ci["range_display"]

    # Brier score
    assert "brier_score" in data
    assert data["brier_score"] is not None
    assert 0.0 <= data["brier_score"] <= 1.0


def test_predict_vqc_uncertainty_quantification(client):
    """POST /predict with VQC estimates uncertainty via quantum finite shots & parameter variance."""
    payload = {
        "patient": SAMPLE_HIGH_RISK_PATIENT,
        "model_name": "VQC",
        "compare": False,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    ci = data["confidence_interval"]
    assert "Quantum Finite Shots" in ci["method"]
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert "%" in ci["range_display"]


def test_calibration_curve_and_brier_scores(client):
    """GET /model-info and /metrics expose Brier scores and calibration diagram."""
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    m_data = metrics_resp.json()
    assert "brier_scores" in m_data
    assert m_data["brier_scores"] is not None
    assert "MLP" in m_data["brier_scores"]

    info_resp = client.get("/model-info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()
    assert info_data["calibration_figure_url"] == "/static/figures/calibration_curves.png"
    assert "global_feature_importances" in info_data
    assert len(info_data["global_feature_importances"]) > 0

    # Static file check
    calib_resp = client.get("/static/figures/calibration_curves.png")
    assert calib_resp.status_code == 200

