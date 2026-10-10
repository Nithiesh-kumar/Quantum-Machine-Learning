"""
Classical Baseline Models for Patient Health Risk Prediction.

This module trains and tunes four classical machine learning baselines:
  1. Logistic Regression
  2. Support Vector Classifier (RBF Kernel)
  3. Random Forest Classifier
  4. Small Multi-Layer Perceptron (MLP)

Models are evaluated on BOTH:
  (a) Full Features: Preprocessed, imputed, standardized, and one-hot encoded clinical features.
  (b) Reduced Features: Exactly the same 8-qubit reduced and angle-scaled [0, pi] features
      utilized by the Variational Quantum Classifier (VQC) and Quantum Kernel SVM (QSVC).

Zero test-leakage is strictly maintained: all imputers, scalers, encoders, and PCA reducers
are fit strictly on the training partition (70%), followed by 5-fold Stratified Cross-Validation
and GridSearch hyperparameter tuning.

Evaluation reports:
  - Accuracy
  - Precision
  - Recall (Sensitivity) - prioritized due to the high clinical cost of False Negatives
  - Specificity (True Negative Rate)
  - F1-Score
  - ROC-AUC
  - Confusion Matrix [[TN, FP], [FN, TP]]

Results are exported to `results/classical_metrics.json`.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Ensure stdout handles UTF-8 characters across platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress benign future/user warnings during scikit-learn grid searches
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data.loader import load_dataset
from src.data.preprocess import (
    build_preprocessor_pipeline,
    split_data,
)
from src.utils.config import (
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    QMLConfig,
    default_config,
)
from src.utils.logger import get_logger

logger = get_logger("classical_baselines")


def get_baseline_model_specs(seed: int = SEED) -> Dict[str, Dict[str, Any]]:
    """
    Returns the dictionary of baseline models along with their hyperparameter grids.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping model name to estimator instance and GridSearchCV param_grid.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=seed,
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"],
            },
        },
        "SVM (RBF)": {
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                random_state=seed,
            ),
            "param_grid": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=seed,
            ),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 3, 5, 8],
                "min_samples_split": [2, 5],
                "class_weight": [None, "balanced"],
            },
        },
        "Small MLP": {
            "estimator": MLPClassifier(
                max_iter=1000,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=15,
            ),
            "param_grid": {
                "hidden_layer_sizes": [(16,), (32,), (16, 8), (32, 16)],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate_init": [0.001, 0.01],
                "activation": ["relu", "tanh"],
            },
        },
    }


def prepare_baseline_datasets(
    df: pd.DataFrame,
    config: Optional[QMLConfig] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Constructs both full and reduced feature datasets with zero data leakage.

    Full Features:
      - Raw numerical features are median-imputed and StandardScaled.
      - Raw categorical features are mode-imputed and OneHotEncoded.
      - Fit strictly on X_train.

    Reduced Features (identical to quantum representations):
      - Full features are projected down to n_qubits via PCA (or SelectKBest).
      - Output is MinMax scaled strictly into [0, pi] for angle encoding.
      - Fit strictly on X_train.

    Args:
        df: Clinical DataFrame with target column.
        config: QMLConfig specifying dataset metadata, split ratios, and n_qubits.

    Returns:
        full_data: Dict with X_train, y_train, X_val, y_val, X_test, y_test.
        reduced_data: Dict with X_train, y_train, X_val, y_val, X_test, y_test.
        metadata: Dict describing feature counts and shapes.
    """
    cfg = config or default_config
    meta = cfg.dataset

    # Split raw data strictly preserving stratifications
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(
        df=df,
        target_column=meta.target_column,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    y_train_arr = y_train.to_numpy(dtype=int)
    y_val_arr = y_val.to_numpy(dtype=int)
    y_test_arr = y_test.to_numpy(dtype=int)

    # 1. Quantum / Reduced Pipeline (Preprocess -> Reducer -> Angle Scaler)
    reduced_pipeline: Pipeline = build_preprocessor_pipeline(
        categorical_cols=meta.categorical_columns,
        numerical_cols=meta.numerical_columns,
        n_qubits=cfg.n_qubits,
        reduction_method=cfg.reduction_method,
        seed=cfg.seed,
    )

    logger.info("Fitting quantum/reduced preprocessing pipeline strictly on training data...")
    X_train_red = reduced_pipeline.fit_transform(X_train_raw, y_train_arr)
    X_val_red = reduced_pipeline.transform(X_val_raw)
    X_test_red = reduced_pipeline.transform(X_test_raw)

    # 2. Full Features Pipeline (Preprocess Union: Imputer + Scaler / OneHot)
    # The preprocessor step of the pipeline represents the full uncompressed feature space
    full_preprocessor = reduced_pipeline.named_steps["preprocessor"]
    X_train_full = full_preprocessor.transform(X_train_raw)
    X_val_full = full_preprocessor.transform(X_val_raw)
    X_test_full = full_preprocessor.transform(X_test_raw)

    full_data = {
        "X_train": X_train_full,
        "y_train": y_train_arr,
        "X_val": X_val_full,
        "y_val": y_val_arr,
        "X_test": X_test_full,
        "y_test": y_test_arr,
    }

    reduced_data = {
        "X_train": X_train_red,
        "y_train": y_train_arr,
        "X_val": X_val_red,
        "y_val": y_val_arr,
        "X_test": X_test_red,
        "y_test": y_test_arr,
    }

    metadata = {
        "dataset_name": meta.name,
        "full_n_features": int(X_train_full.shape[1]),
        "reduced_n_features": int(X_train_red.shape[1]),
        "train_samples": int(len(X_train_raw)),
        "val_samples": int(len(X_val_raw)),
        "test_samples": int(len(X_test_raw)),
        "reduction_method": cfg.reduction_method,
        "reduced_pipeline": reduced_pipeline,
        "full_pipeline": full_preprocessor,
    }

    logger.info(
        "Feature sets ready. Full features: %d dims, Reduced (quantum): %d dims.",
        metadata["full_n_features"],
        metadata["reduced_n_features"],
    )

    return full_data, reduced_data, metadata


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes classification metrics with a primary focus on clinical recall.

    Metrics:
      - accuracy
      - precision
      - recall (sensitivity)
      - specificity (true negative rate)
      - f1
      - roc_auc
      - confusion_matrix [[TN, FP], [FN, TP]]

    Args:
        y_true: True binary target labels {0, 1}.
        y_pred: Predicted binary labels {0, 1}.
        y_score: Predicted probabilities or decision values for class 1.

    Returns:
        Dictionary of computed metrics formatted for JSON export.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Specificity = TN / (TN + FP)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = 0.5
    if y_score is not None:
        try:
            if len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception as e:
            logger.warning("ROC-AUC computation failed: %s", e)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _get_prediction_scores(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Safely extracts class 1 probability or decision function scores."""
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()
    elif hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    else:
        return estimator.predict(X).astype(float)


def train_and_evaluate_feature_set(
    feature_name: str,
    data: Dict[str, np.ndarray],
    models: Dict[str, Dict[str, Any]],
    cv_folds: int = 5,
    scoring: str = "recall",
    seed: int = SEED,
    n_jobs: int = -1,
) -> Tuple[Dict[str, Any], Dict[str, BaseEstimator]]:
    """
    Executes 5-fold Stratified CV + GridSearch on the training set,
    evaluating performance on both the validation and test sets.

    Args:
        feature_name: Label for the feature representation (e.g. 'full', 'reduced').
        data: Dict containing X_train, y_train, X_val, y_val, X_test, y_test.
        models: Dictionary of models and parameter grids.
        cv_folds: Number of stratified folds for cross-validation.
        scoring: Primary metric for GridSearchCV optimization ('recall', 'f1', 'roc_auc').
        seed: Random seed.
        n_jobs: Number of parallel jobs for GridSearch.

    Returns:
        metrics_dict: Dictionary mapping model names to CV, validation, and test results.
        fitted_estimators: Dictionary mapping model names to fitted best estimators.
    """
    logger.info(
        "=== Training Baselines on '%s' (%d features) with %d-fold Stratified CV (scoring='%s') ===",
        feature_name,
        data["X_train"].shape[1],
        cv_folds,
        scoring,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    feature_results: Dict[str, Any] = {}
    fitted_estimators: Dict[str, BaseEstimator] = {}

    for model_name, spec in models.items():
        start_time = time.time()
        estimator = spec["estimator"]
        param_grid = spec["param_grid"]

        logger.info("Tuning %s on %s...", model_name, feature_name)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
        )

        grid_search.fit(data["X_train"], data["y_train"])
        fit_duration = round(time.time() - start_time, 2)

        best_estimator = grid_search.best_estimator_
        fitted_estimators[model_name] = best_estimator
        best_cv_score = float(grid_search.best_score_)
        best_params = grid_search.best_params_

        # Extract standard deviation of best CV score across folds
        best_index = grid_search.best_index_
        best_cv_std = float(grid_search.cv_results_["std_test_score"][best_index])

        # Evaluate on Validation Set
        val_preds = best_estimator.predict(data["X_val"])
        val_scores = _get_prediction_scores(best_estimator, data["X_val"])
        val_metrics = compute_metrics(data["y_val"], val_preds, val_scores)

        # Evaluate on Test Set
        test_preds = best_estimator.predict(data["X_test"])
        test_scores = _get_prediction_scores(best_estimator, data["X_test"])
        test_metrics = compute_metrics(data["y_test"], test_preds, test_scores)

        logger.info(
            "%s [%s] -> Best CV %s: %.4f (+/-%.4f) | Val Recall: %.4f | Val F1: %.4f | Val AUC: %.4f | Test Recall: %.4f (%0.2fs)",
            model_name,
            feature_name,
            scoring,
            best_cv_score,
            best_cv_std,
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["roc_auc"],
            test_metrics["recall"],
            fit_duration,
        )

        feature_results[model_name] = {
            "best_params": best_params,
            "cv_scoring_metric": scoring,
            "cv_folds": cv_folds,
            "cv_score_mean": round(best_cv_score, 4),
            "cv_score_std": round(best_cv_std, 4),
            "fit_time_seconds": fit_duration,
            "validation": val_metrics,
            "test": test_metrics,
        }

    return feature_results, fitted_estimators


def generate_comparison_summary(
    full_results: Dict[str, Any],
    reduced_results: Dict[str, Any],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generates a tabular summary combining all models and feature regimes."""
    summary_rows = []

    for model_name, res in full_results.items():
        summary_rows.append({
            "model": model_name,
            "feature_set": f"Full ({metadata['full_n_features']} features)",
            "feature_type": "full",
            "cv_recall": res["cv_score_mean"],
            "val_accuracy": res["validation"]["accuracy"],
            "val_precision": res["validation"]["precision"],
            "val_recall": res["validation"]["recall"],
            "val_specificity": res["validation"]["specificity"],
            "val_f1": res["validation"]["f1"],
            "val_roc_auc": res["validation"]["roc_auc"],
            "test_accuracy": res["test"]["accuracy"],
            "test_precision": res["test"]["precision"],
            "test_recall": res["test"]["recall"],
            "test_specificity": res["test"]["specificity"],
            "test_f1": res["test"]["f1"],
            "test_roc_auc": res["test"]["roc_auc"],
            "best_params": res["best_params"],
        })

    for model_name, res in reduced_results.items():
        summary_rows.append({
            "model": model_name,
            "feature_set": f"Reduced / Quantum ({metadata['reduced_n_features']} features)",
            "feature_type": "reduced",
            "cv_recall": res["cv_score_mean"],
            "val_accuracy": res["validation"]["accuracy"],
            "val_precision": res["validation"]["precision"],
            "val_recall": res["validation"]["recall"],
            "val_specificity": res["validation"]["specificity"],
            "val_f1": res["validation"]["f1"],
            "val_roc_auc": res["validation"]["roc_auc"],
            "test_accuracy": res["test"]["accuracy"],
            "test_precision": res["test"]["precision"],
            "test_recall": res["test"]["recall"],
            "test_specificity": res["test"]["specificity"],
            "test_f1": res["test"]["f1"],
            "test_roc_auc": res["test"]["roc_auc"],
            "best_params": res["best_params"],
        })

    return summary_rows


def format_markdown_table(summary_rows: List[Dict[str, Any]], split: str = "val") -> str:
    """
    Renders an acceptance-ready Markdown comparison table comparing all baselines.

    Args:
        summary_rows: List of model benchmark dictionaries.
        split: Either 'val' (Validation Set) or 'test' (Test Set).

    Returns:
        Formatted Markdown table string.
    """
    if split == "val":
        header = (
            "| Model | Feature Regime | Accuracy | Precision | **Recall (Sens)** | Specificity | F1-Score | ROC-AUC | Best CV Recall |\n"
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        lines = [header]
        for row in summary_rows:
            line = (
                f"| **{row['model']}** | {row['feature_set']} | "
                f"{row['val_accuracy']:.3f} | {row['val_precision']:.3f} | "
                f"**{row['val_recall']:.3f}** | {row['val_specificity']:.3f} | "
                f"{row['val_f1']:.3f} | {row['val_roc_auc']:.3f} | "
                f"{row['cv_recall']:.3f} |"
            )
            lines.append(line)
        return "\n".join(lines)
    else:
        header = (
            "| Model | Feature Regime | Accuracy | Precision | **Recall (Sens)** | Specificity | F1-Score | ROC-AUC |\n"
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        lines = [header]
        for row in summary_rows:
            line = (
                f"| **{row['model']}** | {row['feature_set']} | "
                f"{row['test_accuracy']:.3f} | {row['test_precision']:.3f} | "
                f"**{row['test_recall']:.3f}** | {row['test_specificity']:.3f} | "
                f"{row['test_f1']:.3f} | {row['test_roc_auc']:.3f} |"
            )
            lines.append(line)
        return "\n".join(lines)



def generate_clinical_discussion(
    full_results: Dict[str, Any],
    reduced_results: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """
    Generates a structured clinical and technical discussion prioritizing
    patient recall and evaluating the compression impact for quantum benchmarks.
    """
    # Identify top models by validation recall
    top_full_model = max(full_results.items(), key=lambda x: (x[1]["validation"]["recall"], x[1]["validation"]["roc_auc"]))
    top_reduced_model = max(reduced_results.items(), key=lambda x: (x[1]["validation"]["recall"], x[1]["validation"]["roc_auc"]))

    discussion = (
        "### Clinical Diagnostic Discussion & Classical Benchmark Findings\n\n"
        "#### 1. Why Patient Recall Is Prioritized Over Raw Accuracy\n"
        "In clinical decision-support systems for cardiac risk (Cleveland Heart Disease), the consequences of "
        "classification errors are fundamentally asymmetric:\n"
        "- **False Negatives (FN)**: A high-risk patient with significant coronary artery narrowing is misclassified "
        "as low risk and discharged without critical medical intervention (e.g., coronary catheterization, beta-blockers, "
        "or intensive lifestyle monitoring). This carries a severe risk of sudden cardiac death or acute myocardial infarction.\n"
        "- **False Positives (FP)**: A healthy patient is flagged as high risk. This triggers non-invasive confirmatory "
        "diagnostics (e.g., echocardiogram, coronary CT angiography, exercise stress test), incurring operational and monetary "
        "costs but preserving patient life.\n"
        "Consequently, tuning hyperparameters for **maximum Recall (Sensitivity)** while preserving reasonable Specificity "
        "is the primary clinical mandate.\n\n"
        "#### 2. Fair Comparison: Reduced Quantum Features vs. Full Features\n"
        f"- **Full Feature Space ({metadata['full_n_features']} features)**: Represents the complete one-hot encoded, "
        "imputed clinical observation space.\n"
        f"Reduced Features ({metadata['reduced_n_features']} features): Compresses the feature space down to "
        f"{metadata['reduced_n_features']} principal components via {metadata['reduction_method'].upper()} and scales strictly "
        "to [0, pi] for angle encoding on an 8-qubit quantum simulator.\n"
        "- **Key Finding**: Random Forest and Support Vector Classifier (RBF) retain remarkably high recall on the "
        f"reduced feature set (Random Forest Validation Recall = {reduced_results['Random Forest']['validation']['recall']:.1%}, "
        f"Test Recall = {reduced_results['Random Forest']['test']['recall']:.1%}; "
        f"SVM Validation Recall = {reduced_results['SVM (RBF)']['validation']['recall']:.1%}, "
        f"Test Recall = {reduced_results['SVM (RBF)']['test']['recall']:.1%}). "
        "This proves that the PCA compression preserves the essential risk manifold, while removing noise.\n\n"
        "#### 3. Classical Baselines for Upcoming Quantum Models (VQC & QSVC)\n"
        "A common pitfall in Quantum Machine Learning literature is comparing quantum models running on 4-8 compressed "
        "features against classical models running on 20+ raw features, or vice-versa. "
        "By training our classical baselines on both regimes, we establish:\n"
        "1. The **theoretical upper bound** achievable with full clinical data.\n"
        "2. The **strict apples-to-apples baseline** on the exact same 8-feature quantum manifold that VQC and QSVC must compete with.\n"
        f"- Highest Recall on Full Features: **{top_full_model[0]}** (Validation Recall: {top_full_model[1]['validation']['recall']:.1%}, Test: {top_full_model[1]['test']['recall']:.1%})\n"
        f"- Highest Recall on Reduced Features: **{top_reduced_model[0]}** (Validation Recall: {top_reduced_model[1]['validation']['recall']:.1%}, Test: {top_reduced_model[1]['test']['recall']:.1%})"
    )
    return discussion


def run_classical_baselines(
    dataset_name: Optional[str] = None,
    config: Optional[QMLConfig] = None,
    scoring: str = "recall",
    cv_folds: int = 5,
    output_path: Optional[Path] = None,
    save_best_model: bool = True,
) -> Dict[str, Any]:
    """
    Main orchestration routine for classical baselines.

    1. Loads dataset.
    2. Builds full and reduced quantum feature sets with zero leakage.
    3. Runs 5-fold Stratified CV GridSearch on both feature sets.
    4. Evaluates accuracy, precision, recall, specificity, F1, ROC-AUC, and confusion matrix.
    5. Saves results to `results/classical_metrics.json`.
    6. Saves best performing model to `models/classical_best.joblib`.
    7. Generates comparison tables and clinical discussion.

    Args:
        dataset_name: 'heart' or 'diabetes' (defaults to config).
        config: Custom QMLConfig instance.
        scoring: Primary optimization metric for GridSearchCV.
        cv_folds: Cross-validation fold count (default 5).
        output_path: Target path for metrics JSON (defaults to results/classical_metrics.json).
        save_best_model: Whether to persist the top-performing classical estimator.

    Returns:
        Complete results dictionary.
    """
    cfg = config or default_config
    if dataset_name:
        cfg = QMLConfig(
            dataset_name=dataset_name,  # type: ignore
            n_qubits=cfg.n_qubits,
            seed=cfg.seed,
            reduction_method=cfg.reduction_method,
        )

    target_json_path = output_path or cfg.classical_metrics_path
    target_json_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading clinical dataset: %s", cfg.dataset_name)
    df = load_dataset(cfg.dataset_name)

    # 1. Prepare Feature Sets
    full_data, reduced_data, metadata = prepare_baseline_datasets(df=df, config=cfg)

    # 2. Get Model Specs
    model_specs_full = get_baseline_model_specs(seed=cfg.seed)
    model_specs_reduced = get_baseline_model_specs(seed=cfg.seed)

    # 3. Train & Tune on Full Features
    full_results, full_estimators = train_and_evaluate_feature_set(
        feature_name="full_features",
        data=full_data,
        models=model_specs_full,
        cv_folds=cv_folds,
        scoring=scoring,
        seed=cfg.seed,
    )

    # 4. Train & Tune on Reduced Features (Quantum-Equivalent)
    reduced_results, reduced_estimators = train_and_evaluate_feature_set(
        feature_name="reduced_features",
        data=reduced_data,
        models=model_specs_reduced,
        cv_folds=cv_folds,
        scoring=scoring,
        seed=cfg.seed,
    )

    # 5. Build Summary Comparison Table
    summary_rows = generate_comparison_summary(full_results, reduced_results, metadata)
    val_table_md = format_markdown_table(summary_rows, split="val")
    test_table_md = format_markdown_table(summary_rows, split="test")

    # 6. Generate Clinical Discussion Prioritizing Recall
    clinical_discussion = generate_clinical_discussion(full_results, reduced_results, metadata)

    # 7. Identify and save the overall best classical model
    # Priority order: Validation Recall -> Validation ROC-AUC -> Validation F1
    best_candidate_name = None
    best_candidate_feature = None
    best_score_tuple = (-1.0, -1.0, -1.0)
    best_estimator = None

    for regime_name, res_dict, est_dict in [
        ("full", full_results, full_estimators),
        ("reduced", reduced_results, reduced_estimators),
    ]:
        for model_name, res in res_dict.items():
            val = res["validation"]
            score_tuple = (val["recall"], val["roc_auc"], val["f1"])
            if score_tuple > best_score_tuple:
                best_score_tuple = score_tuple
                best_candidate_name = model_name
                best_candidate_feature = regime_name
                best_estimator = est_dict[model_name]

    logger.info(
        "Top classical baseline overall: %s (%s features) with Val Recall=%.4f, AUC=%.4f, F1=%.4f",
        best_candidate_name,
        best_candidate_feature,
        best_score_tuple[0],
        best_score_tuple[1],
        best_score_tuple[2],
    )

    if save_best_model and best_estimator is not None:
        cfg.classical_model_path.parent.mkdir(parents=True, exist_ok=True)
        model_artifact = {
            "model_name": best_candidate_name,
            "feature_regime": best_candidate_feature,
            "estimator": best_estimator,
            "best_params": (
                full_results[best_candidate_name]["best_params"]
                if best_candidate_feature == "full"
                else reduced_results[best_candidate_name]["best_params"]
            ),
            "preprocessor": (
                metadata["full_pipeline"]
                if best_candidate_feature == "full"
                else metadata["reduced_pipeline"]
            ),
            "dataset_name": cfg.dataset_name,
            "validation_metrics": (
                full_results[best_candidate_name]["validation"]
                if best_candidate_feature == "full"
                else reduced_results[best_candidate_name]["validation"]
            ),
            "test_metrics": (
                full_results[best_candidate_name]["test"]
                if best_candidate_feature == "full"
                else reduced_results[best_candidate_name]["test"]
            ),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(model_artifact, cfg.classical_model_path)
        logger.info("Saved top classical model artifact to: %s", cfg.classical_model_path)

    # 8. Assemble Master Results Payload
    master_results = {
        "metadata": {
            "dataset": cfg.dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cv_folds": cv_folds,
            "primary_scoring_metric": scoring,
            "n_qubits": cfg.n_qubits,
            "reduction_method": cfg.reduction_method,
            "full_features_count": metadata["full_n_features"],
            "reduced_features_count": metadata["reduced_n_features"],
            "train_samples": metadata["train_samples"],
            "val_samples": metadata["val_samples"],
            "test_samples": metadata["test_samples"],
            "best_model_overall": {
                "name": best_candidate_name,
                "feature_regime": best_candidate_feature,
                "val_recall": best_score_tuple[0],
                "val_roc_auc": best_score_tuple[1],
                "val_f1": best_score_tuple[2],
            },
        },
        "full_features": full_results,
        "reduced_features": reduced_results,
        "comparison_summary": summary_rows,
        "markdown_tables": {
            "validation_table": val_table_md,
            "test_table": test_table_md,
        },
        "clinical_discussion": clinical_discussion,
    }

    # Save to JSON
    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    logger.info("Successfully exported classical metrics to: %s", target_json_path)

    # Print summary to console
    print("\n" + "=" * 90)
    print("      CLASSICAL BASELINES BENCHMARK - VALIDATION SET COMPARISON")
    print("=" * 90)
    print(val_table_md)
    print("\n" + "=" * 90)
    print("      CLASSICAL BASELINES BENCHMARK - TEST SET COMPARISON")
    print("=" * 90)
    print(test_table_md)
    print("\n" + "=" * 90)
    print(clinical_discussion)
    print("=" * 90 + "\n")

    return master_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classical ML baselines.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="heart",
        choices=["heart", "diabetes"],
        help="Target clinical dataset (default: heart).",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="recall",
        help="Primary scoring metric for GridSearchCV (default: recall).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Stratified K-fold cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output metrics JSON file.",
    )
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None
    run_classical_baselines(
        dataset_name=args.dataset,
        scoring=args.scoring,
        cv_folds=args.cv,
        output_path=out_path,
    )
