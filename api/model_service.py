"""Model service managing single-instance loading, preprocessing, and inference."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from api.schemas import (
    ConfidenceInterval,
    FeatureContribution,
    PatientData,
    PredictResponse,
    SingleModelPrediction,
)
from src.data.preprocess import transform_single_patient
from src.evaluation.explainability import (
    compute_pca_mapped_permutation_importance,
    estimate_vqc_prediction_uncertainty,
    estimate_classical_prediction_uncertainty,
    compute_calibration_and_brier_scores,
)
from src.quantum.vqc import VariationalQuantumClassifier
from src.quantum.quantum_kernel import build_zz_feature_map, build_quantum_kernel
from src.utils.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    MEDICAL_DISCLAIMER,
)
from src.utils.logger import get_logger

logger = get_logger("model_service")


# Baseline Cleveland population averages for clinical attribution
FEATURE_BASELINES: Dict[str, Dict[str, Any]] = {
    "ca": {"mean": 0.67, "elevates_if": "greater", "weight": 0.28, "unit": "vessels"},
    "oldpeak": {"mean": 1.04, "elevates_if": "greater", "weight": 0.24, "unit": "mm"},
    "thal": {"mean": 4.73, "elevates_if": "greater", "weight": 0.22, "unit": "status"},
    "thalach": {"mean": 149.6, "elevates_if": "less", "weight": 0.18, "unit": "bpm"},
    "cp": {"mean": 3.16, "elevates_if": "greater", "weight": 0.16, "unit": "type"},
    "exang": {"mean": 0.33, "elevates_if": "greater", "weight": 0.15, "unit": "binary"},
    "age": {"mean": 54.4, "elevates_if": "greater", "weight": 0.12, "unit": "years"},
    "trestbps": {"mean": 131.6, "elevates_if": "greater", "weight": 0.10, "unit": "mm Hg"},
    "chol": {"mean": 246.3, "elevates_if": "greater", "weight": 0.08, "unit": "mg/dl"},
    "slope": {"mean": 1.60, "elevates_if": "greater", "weight": 0.08, "unit": "slope"},
    "sex": {"mean": 0.68, "elevates_if": "greater", "weight": 0.07, "unit": "binary"},
    "restecg": {"mean": 0.99, "elevates_if": "greater", "weight": 0.05, "unit": "ecg"},
    "fbs": {"mean": 0.15, "elevates_if": "greater", "weight": 0.04, "unit": "binary"},
}


class ModelService:
    """
    Singleton service that preloads preprocessor and all machine learning models
    into memory at startup to guarantee sub-second prediction latency.
    """
    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        self.production_model_path = MODELS_DIR / "production_model.joblib"
        self.production_meta_path = MODELS_DIR / "production_model_metadata.json"
        self.vqc_path = MODELS_DIR / "vqc.pt"
        self.qsvc_path = MODELS_DIR / "qsvc_model.joblib"

        self.preprocessor = None
        self.production_model = None
        self.production_metadata: Dict[str, Any] = {}
        self.production_model_name: str = "MLP"

        self.vqc_model: Optional[VariationalQuantumClassifier] = None
        self.qsvc_bundle: Optional[Dict[str, Any]] = None
        self.qsvc_kernel = None
        self.X_train_reference: Optional[np.ndarray] = None
        self.X_test_reference: Optional[np.ndarray] = None
        self.y_test_reference: Optional[np.ndarray] = None

        # Explainability and calibration caches
        self.global_feature_importances: Dict[str, float] = {}
        self.brier_scores: Dict[str, float] = {}
        self.calibration_results: Dict[str, Any] = {}

        # Classical baseline registry for side-by-side comparisons
        self.models_registry: Dict[str, Any] = {}
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def initialize(self) -> None:
        """Loads all models, preprocessors, and reference datasets once into memory."""
        if self.is_loaded:
            logger.info("ModelService already initialized.")
            return

        logger.info("Initializing ModelService - loading models and preprocessor...")
        start_time = time.time()

        # 1. Load Preprocessor
        if self.preprocessor_path.exists():
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Loaded preprocessor from: %s", self.preprocessor_path)
        else:
            logger.warning("Preprocessor not found at: %s", self.preprocessor_path)

        # 2. Load Production Model & Metadata
        if self.production_model_path.exists():
            self.production_model = joblib.load(self.production_model_path)
            logger.info("Loaded production model from: %s", self.production_model_path)
        if self.production_meta_path.exists():
            with open(self.production_meta_path, "r", encoding="utf-8") as f:
                self.production_metadata = json.load(f)
            self.production_model_name = self.production_metadata.get("production_model_name", "MLP")
            logger.info("Production model name: %s", self.production_model_name)

        # 3. Load VQC Quantum Model
        if self.vqc_path.exists():
            try:
                self.vqc_model = VariationalQuantumClassifier.load_model(self.vqc_path)
                logger.info("Loaded VQC model from: %s", self.vqc_path)
                self.models_registry["VQC"] = self.vqc_model
            except Exception as e:
                logger.warning("Failed to load VQC checkpoint (%s). Continuing...", e)

        # 4. Load Reference Training & Test Data
        processed_data_path = PROCESSED_DATA_DIR / "heart_processed.npz"
        if processed_data_path.exists():
            data = np.load(processed_data_path)
            self.X_train_reference = data["X_train"][:, :4]
            y_train_ref = data["y_train"]
            if "X_test" in data and "y_test" in data:
                self.X_test_reference = data["X_test"][:, :4]
                self.y_test_reference = data["y_test"]
            logger.info("Loaded reference training features (shape: %s)", self.X_train_reference.shape)

            # Fit classical baselines for side-by-side comparison
            self._initialize_classical_baselines(self.X_train_reference, y_train_ref)

        # 5. Load QSVC Quantum Model Bundle
        if self.qsvc_path.exists():
            try:
                self.qsvc_bundle = joblib.load(self.qsvc_path)
                fmap = build_zz_feature_map(n_qubits=4, reps=2, entanglement="linear")
                self.qsvc_kernel = build_quantum_kernel(fmap, "fidelity_statevector")
                self.models_registry["QSVC"] = self.qsvc_bundle["model"]
                logger.info("Loaded QSVC model bundle from: %s", self.qsvc_path)
            except Exception as e:
                logger.warning("Failed to load QSVC model bundle (%s). Continuing...", e)

        # Register Production Model under its specific name as well
        if self.production_model is not None:
            self.models_registry[self.production_model_name] = self.production_model
            self.models_registry["production"] = self.production_model

        # 6. Compute Global PCA-Mapped Permutation Feature Importance
        if self.production_model is not None and self.preprocessor is not None and self.X_test_reference is not None:
            try:
                explain_res = compute_pca_mapped_permutation_importance(
                    model=self.production_model,
                    X_reduced_test=self.X_test_reference,
                    y_test=self.y_test_reference,
                    preprocessor_pipeline=self.preprocessor,
                    n_components=4,
                    n_repeats=10,
                )
                self.global_feature_importances = explain_res.get("clinical_importances", {})
                logger.info("PCA-mapped permutation importance calculated for %d features", len(self.global_feature_importances))
            except Exception as e:
                logger.warning("Failed to compute PCA-mapped feature importances (%s). Using baselines.", e)

        # 7. Compute Calibration Curves and Brier Scores
        if self.X_test_reference is not None and self.y_test_reference is not None and self.models_registry:
            try:
                calib_dict = compute_calibration_and_brier_scores(
                    models_dict=self.models_registry,
                    X_test=self.X_test_reference,
                    y_test=self.y_test_reference,
                )
                self.calibration_results = calib_dict
                self.brier_scores = {k: v["brier_score"] for k, v in calib_dict.items() if "brier_score" in v}
                logger.info("Calibration & Brier scores computed: %s", self.brier_scores)
            except Exception as e:
                logger.warning("Failed to compute calibration curves and Brier scores (%s)", e)

        self.is_loaded = True
        logger.info("ModelService initialized successfully in %.2f seconds.", time.time() - start_time)

    def _initialize_classical_baselines(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains or initializes standard classical baselines for fast side-by-side comparison."""
        try:
            lr = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
            svm_rbf = SVC(kernel="rbf", probability=True, random_state=42).fit(X_train, y_train)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
            mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42).fit(X_train, y_train)

            self.models_registry["Logistic Regression"] = lr
            self.models_registry["SVM-RBF"] = svm_rbf
            self.models_registry["Random Forest"] = rf
            self.models_registry["MLP"] = mlp
            logger.info("Initialized 4 classical baseline models in memory for comparison.")
        except Exception as e:
            logger.warning("Error fitting baseline models: %s", e)

    def transform_patient(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a patient dict into the 4-qubit [0, pi] representation."""
        return transform_single_patient(patient_dict, preprocessor_path=self.preprocessor_path)[:, :4]

    def _calculate_feature_contributions(self, patient_dict: Dict[str, Any]) -> List[FeatureContribution]:
        """Calculates clinical biomarker attributions using PCA-mapped importance and patient deviation."""
        contributions = []

        for feat, info in FEATURE_BASELINES.items():
            val = float(patient_dict.get(feat, info["mean"]))
            mean = float(info["mean"])
            weight = float(self.global_feature_importances.get(feat, info["weight"]))

            # Compute normalized departure from mean
            diff = val - mean
            if info["elevates_if"] == "greater":
                elevates = diff > 0
                magnitude = max(0.0, diff) * weight
            else:
                elevates = diff < 0
                magnitude = max(0.0, -diff) * weight

            importance = round(float(magnitude), 4)
            impact_str = "Elevates Risk" if elevates else "Reduces Risk"

            contributions.append(
                FeatureContribution(
                    feature=feat,
                    patient_value=round(val, 2),
                    baseline_mean=round(mean, 2),
                    importance=importance,
                    directional_impact=impact_str,
                )
            )

        # Sort by importance descending and return top 3 contributing features
        contributions.sort(key=lambda c: c.importance, reverse=True)
        return contributions[:3]

    def _compute_uncertainty(
        self, model_name: str, x_vector: np.ndarray, prob: float
    ) -> ConfidenceInterval:
        """Estimates prediction uncertainty and confidence interval for the given model."""
        name_clean = model_name.strip()
        if name_clean == "VQC" and self.vqc_model is not None:
            ci_data = estimate_vqc_prediction_uncertainty(self.vqc_model, x_vector, n_shots=1000)
        else:
            raw_model = self.models_registry.get(name_clean, self.production_model)
            if isinstance(raw_model, dict) and "model" in raw_model:
                raw_model = raw_model["model"]
            ci_data = estimate_classical_prediction_uncertainty(raw_model, x_vector)

        return ConfidenceInterval(
            lower=ci_data["ci_lower"],
            upper=ci_data["ci_upper"],
            confidence_level=ci_data.get("confidence_level", 0.95),
            uncertainty_std=ci_data["uncertainty_std"],
            range_display=ci_data["range_display"],
            method=ci_data["method"],
        )

    def _predict_with_model(
        self, model_name: str, x_vector: np.ndarray
    ) -> Tuple[float, int, str]:
        """
        Executes prediction with a specified model.
        Returns (probability, label, risk_level).
        """
        name_clean = model_name.strip()

        # 1. Quantum VQC
        if name_clean == "VQC":
            if self.vqc_model is None:
                raise RuntimeError("VQC model is not available or failed to load.")
            probs = self.vqc_model.predict_proba(x_vector)[0]
            prob = float(probs[1])

        # 2. Quantum QSVC
        elif name_clean == "QSVC":
            if self.qsvc_bundle is None or self.X_train_reference is None or self.qsvc_kernel is None:
                raise RuntimeError("QSVC model bundle is not available or reference data is missing.")
            k_vec = self.qsvc_kernel.evaluate(x_vector, self.X_train_reference)
            svc_model = self.qsvc_bundle["model"]
            if hasattr(svc_model, "predict_proba"):
                prob = float(svc_model.predict_proba(k_vec)[0, 1])
            else:
                dec = float(svc_model.decision_function(k_vec)[0])
                prob = float(1.0 / (1.0 + np.exp(-dec)))

        # 3. Classical Models or Production Model
        else:
            model = self.models_registry.get(name_clean, self.production_model)
            if model is None:
                raise RuntimeError(f"Requested model '{name_clean}' is not available.")
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(x_vector)[0, 1])
            else:
                dec = float(model.decision_function(x_vector)[0])
                prob = float(1.0 / (1.0 + np.exp(-dec)))

        prob = min(max(prob, 0.0), 1.0)
        label = 1 if prob >= 0.5 else 0

        if prob < 0.35:
            risk_level = "Low"
        elif prob <= 0.65:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return prob, label, risk_level

    def predict(
        self, patient: PatientData, model_name: str = "production", compare: bool = False
    ) -> PredictResponse:
        """
        Full end-to-end prediction pipeline for a patient record.
        """
        if not self.is_loaded:
            self.initialize()

        start_time = time.time()
        patient_dict = patient.model_dump()

        # 1. Quantum Feature Preprocessing
        x_vector = self.transform_patient(patient_dict)
        quantum_rep = [round(float(v), 4) for v in x_vector[0]]

        # 2. Primary Model Prediction
        target_name = self.production_model_name if model_name.lower() in ["production", "default", ""] else model_name
        prob, label, risk_level = self._predict_with_model(target_name, x_vector)
        is_quantum = target_name in ["VQC", "QSVC"]
        model_type_str = "Quantum" if is_quantum else "Classical"

        # 3. Uncertainty Quantification & Calibration
        ci = self._compute_uncertainty(target_name, x_vector, prob)
        brier = self.brier_scores.get(target_name, self.brier_scores.get(self.production_model_name, None))

        # 4. Feature Attribution / Explanations (Top 3 Clinical Features)
        top_features = self._calculate_feature_contributions(patient_dict)

        # 5. Optional Side-by-Side Model Comparison
        comparison_dict = None
        if compare:
            comparison_dict = {}
            candidate_models = ["MLP", "Random Forest", "Logistic Regression", "SVM-RBF", "VQC", "QSVC"]
            for cand in candidate_models:
                try:
                    c_prob, c_label, c_risk = self._predict_with_model(cand, x_vector)
                    c_type = "Quantum" if cand in ["VQC", "QSVC"] else "Classical"
                    conf = round(float(c_prob if c_label == 1 else (1.0 - c_prob)) * 100.0, 1)
                    c_ci = self._compute_uncertainty(cand, x_vector, c_prob)
                    c_brier = self.brier_scores.get(cand, None)

                    comparison_dict[cand] = SingleModelPrediction(
                        model_name=cand,
                        model_type=c_type,
                        risk_probability=round(c_prob, 4),
                        risk_label=c_label,
                        risk_level=c_risk,
                        confidence_percentage=conf,
                        confidence_interval=c_ci,
                        brier_score=c_brier,
                    )
                except Exception as e:
                    logger.warning("Side-by-side comparison for '%s' failed: %s", cand, e)

        duration_ms = round((time.time() - start_time) * 1000.0, 2)

        return PredictResponse(
            risk_probability=round(prob, 4),
            risk_label=label,
            risk_level=risk_level,
            model_used=target_name,
            model_type=model_type_str,
            confidence_interval=ci,
            brier_score=brier,
            quantum_representation=quantum_rep,
            top_contributing_features=top_features,
            comparison=comparison_dict,
            processing_time_ms=duration_ms,
            medical_disclaimer=MEDICAL_DISCLAIMER,
        )
