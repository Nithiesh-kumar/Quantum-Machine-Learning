"""Explainability, PCA loading mapping, quantum uncertainty, and calibration module."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, ModuleNotFoundError):
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from src.utils.config import RESULTS_DIR
from src.utils.logger import get_logger

logger = get_logger("explainability")


# ---------------------------------------------------------------------------
# 1. Permutation Importance Mapped Through PCA Loadings
# ---------------------------------------------------------------------------
def compute_pca_mapped_permutation_importance(
    model: Any,
    X_reduced_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor_pipeline: Pipeline,
    n_components: int = 4,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes permutation feature importance on the reduced quantum feature space
    and projects the importance scores back onto the original clinical biomarkers
    via the PCA loading matrix and OneHot column mappings.

    Returns:
        Dictionary containing:
          - 'reduced_importances': Permutation importance on reduced components.
          - 'clinical_importances': Mapped importance per original clinical biomarker.
          - 'ranked_features': List of (feature, importance) sorted descending.
    """
    # 1. Permutation importance on reduced features (shape: N, 4)
    r = permutation_importance(
        model,
        X_reduced_test[:, :n_components],
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    reduced_imp = np.maximum(0.0, r.importances_mean)

    # 2. Extract PCA components and ColumnTransformer preprocessed feature names
    ct = preprocessor_pipeline.named_steps["preprocessor"]
    pca = preprocessor_pipeline.named_steps["reducer"]
    preprocessed_names = ct.get_feature_names_out()

    # V matrix: shape (n_components, n_preprocessed_features)
    # V[j, k] represents the weight of preprocessed feature k in principal component j
    V = np.abs(pca.components_[:n_components, :])

    # 3. Project reduced importance through PCA loadings: W = I * V
    preprocessed_importance = np.dot(reduced_imp, V)

    # 4. Map preprocessed column names back to original clinical features
    # e.g. 'num__age' -> 'age', 'cat__cp_4.0' -> 'cp', 'cat__thal_7.0' -> 'thal'
    clinical_importances: Dict[str, float] = {}
    for name, weight in zip(preprocessed_names, preprocessed_importance):
        # Extract base clinical feature name
        clean_name = name.replace("num__", "").replace("cat__", "")
        base_feature = clean_name.split("_")[0]
        clinical_importances[base_feature] = clinical_importances.get(base_feature, 0.0) + float(weight)

    # Normalize clinical importances so they sum to 1.0 (relative attribution)
    total_weight = sum(clinical_importances.values())
    if total_weight > 0:
        for k in clinical_importances:
            clinical_importances[k] = round(clinical_importances[k] / total_weight, 4)

    ranked_features = sorted(clinical_importances.items(), key=lambda x: x[1], reverse=True)

    logger.info("Top PCA-mapped clinical features: %s", ranked_features[:5])
    return {
        "reduced_importances": [round(float(v), 4) for v in reduced_imp],
        "clinical_importances": clinical_importances,
        "ranked_features": ranked_features,
    }


# ---------------------------------------------------------------------------
# 2. Quantum VQC & Classical Prediction Uncertainty Estimation
# ---------------------------------------------------------------------------
def estimate_vqc_prediction_uncertainty(
    vqc_model: Any,
    x_vector: np.ndarray,
    n_shots: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Estimates prediction uncertainty for a Variational Quantum Classifier (VQC)
    by combining quantum finite-shot Bernoulli variance with parameter calibration variance.

    For a Pauli-Z expectation mapped to probability P:
      - Quantum shot variance: sigma_shot^2 = P * (1 - P) / n_shots
      - Parameter calibration variance: simulated via small gate jitter
    """
    # 1. Base probability
    probs = vqc_model.predict_proba(x_vector)[0]
    base_p = float(probs[1])

    # 2. Quantum finite-shot variance (Bernoulli trial variance)
    # sigma_shot = sqrt(p * (1 - p) / n_shots)
    sigma_shot = np.sqrt(max(1e-5, base_p * (1.0 - base_p)) / n_shots)

    # 3. Gate perturbation sampling (5 micro-evaluations for parameter drift uncertainty)
    sample_probs = [base_p]
    if hasattr(vqc_model, "weights_") and vqc_model.weights_ is not None:
        orig_weights = vqc_model.weights_.copy()
        for _ in range(8):
            try:
                # Add micro gate calibration noise (std=0.01 rad)
                jittered_weights = orig_weights + np.random.normal(0, 0.015, orig_weights.shape)
                vqc_model.weights_ = jittered_weights
                sample_p = float(vqc_model.predict_proba(x_vector)[0, 1])
                sample_probs.append(sample_p)
            except Exception:
                pass
            finally:
                vqc_model.weights_ = orig_weights

    # Combined standard error
    sigma_param = np.std(sample_probs) if len(sample_probs) > 1 else 0.0
    total_std = float(np.sqrt(sigma_shot**2 + sigma_param**2))

    # 95% Confidence Interval (z = 1.96)
    z_score = 1.96 if confidence_level == 0.95 else 2.576
    ci_lower = max(0.0, float(base_p - z_score * total_std))
    ci_upper = min(1.0, float(base_p + z_score * total_std))

    return {
        "probability": round(base_p, 4),
        "uncertainty_std": round(total_std, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": confidence_level,
        "range_display": f"{round(ci_lower * 100)}% - {round(ci_upper * 100)}% (95% CI)",
        "method": f"Quantum Finite Shots ({n_shots}) + Gate Calibration Variance",
    }


def estimate_classical_prediction_uncertainty(
    model: Any,
    x_vector: np.ndarray,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Estimates prediction uncertainty for classical models (MLP, RF, LogReg)
    via ensemble variance or local perturbation.
    """
    if hasattr(model, "predict_proba"):
        base_p = float(model.predict_proba(x_vector)[0, 1])
    else:
        dec = float(model.decision_function(x_vector)[0])
        base_p = float(1.0 / (1.0 + np.exp(-dec)))

    # For Random Forest, use individual tree variance
    if hasattr(model, "estimators_"):
        tree_preds = [float(e.predict_proba(x_vector)[0, 1]) for e in model.estimators_]
        std = float(np.std(tree_preds))
    elif hasattr(model, "coefs_"):  # MLP
        # Empirical perturbation uncertainty
        perturb_preds = [base_p]
        for _ in range(10):
            jitter = np.random.normal(0, 0.02, x_vector.shape)
            p_j = float(model.predict_proba(x_vector + jitter)[0, 1])
            perturb_preds.append(p_j)
        std = max(0.015, float(np.std(perturb_preds)))
    else:
        # Standard analytical binomial standard error
        std = float(np.sqrt(max(1e-5, base_p * (1.0 - base_p)) / 212))

    z_score = 1.96 if confidence_level == 0.95 else 2.576
    ci_lower = max(0.0, float(base_p - z_score * std))
    ci_upper = min(1.0, float(base_p + z_score * std))

    return {
        "probability": round(base_p, 4),
        "uncertainty_std": round(std, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": confidence_level,
        "range_display": f"{round(ci_lower * 100)}% - {round(ci_upper * 100)}% (95% CI)",
        "method": "Empirical Prediction Interval (95% CI)",
    }


# ---------------------------------------------------------------------------
# 3. Model Calibration Curve & Brier Score Computation
# ---------------------------------------------------------------------------
def compute_calibration_and_brier_scores(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 5,
    save_figure_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes Brier scores and calibration curves (reliability diagrams) across
    all benchmark models and generates a publication comparison plot.
    """
    calibration_results: Dict[str, Any] = {}

    fig, ax = (plt.subplots(figsize=(8.5, 6.5), dpi=300)) if HAS_MATPLOTLIB else (None, None)
    if HAS_MATPLOTLIB and ax is not None:
        # Perfect calibration reference line (y = x)
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated (Ideal)", alpha=0.7)

    colors = {
        "MLP": "#9467bd",
        "Random Forest": "#ff7f0e",
        "Logistic Regression": "#1f77b4",
        "SVM-RBF": "#2ca02c",
        "VQC": "#d62728",
        "QSVC": "#8c564b",
    }

    for name, raw_model in models_dict.items():
        try:
            model = raw_model["model"] if isinstance(raw_model, dict) and "model" in raw_model else raw_model
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                dec = model.decision_function(X_test)
                probs = 1.0 / (1.0 + np.exp(-dec))

            brier = float(brier_score_loss(y_test, probs))
            fop, mpv = calibration_curve(y_test, probs, n_bins=n_bins, strategy="uniform")

            calibration_results[name] = {
                "brier_score": round(brier, 4),
                "fraction_of_positives": [round(float(v), 4) for v in fop],
                "mean_predicted_values": [round(float(v), 4) for v in mpv],
            }

            if HAS_MATPLOTLIB and ax is not None:
                color = colors.get(name, "#333333")
                ax.plot(mpv, fop, "s-", label=f"{name} (Brier: {brier:.3f})", color=color, linewidth=2, markersize=5)
        except Exception as e:
            logger.warning("Calibration calculation for %s failed: %s", name, e)

    if HAS_MATPLOTLIB and ax is not None and fig is not None:
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Observed Fraction of Positives (True Risk)", fontsize=11)
        ax.set_title("Probability Calibration Curves & Brier Scores\n(UCI Cleveland Test Set)", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        fig_path = save_figure_path or (RESULTS_DIR / "figures" / "calibration_curves.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        logger.info("Saved calibration curves figure to: %s", fig_path)

    return calibration_results
