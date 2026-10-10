"""Unit tests for Comparative Evaluation Suite (src/evaluation/compare.py)."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation.compare import (
    calculate_metrics,
    count_model_parameters,
    plot_f1_recall_comparison,
    plot_model_size_comparison,
    plot_roc_curves_overlay,
    plot_training_time_comparison,
    select_and_save_production_model,
)


def test_calculate_metrics():
    """Verify calculation of clinical metrics and recall-weighted score."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_score)

    assert metrics["accuracy"] == 0.6
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == round(2 / 3, 4)
    assert "f2" in metrics
    assert "recall_weighted_f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["recall_weighted_f1"] == round(0.6 * metrics["recall"] + 0.4 * metrics["f1"], 4)


def test_count_model_parameters():
    """Verify parameter count for classical models."""
    X = np.random.randn(20, 4)
    y = np.array([0] * 10 + [1] * 10)
    lr = LogisticRegression().fit(X, y)

    params = count_model_parameters(lr, "Logistic Regression")
    assert params == 5  # 4 features + 1 bias


def test_visualization_generators(tmp_path):
    """Verify that plotting utilities create valid image files."""
    dummy_results = {
        "Logistic Regression": {
            "model_type": "classical",
            "roc_fpr": [0.0, 0.2, 1.0],
            "roc_tpr": [0.0, 0.8, 1.0],
            "param_count": 5,
            "summary": {
                "recall_mean": 0.8,
                "recall_std": 0.0,
                "f1_mean": 0.75,
                "f1_std": 0.0,
                "roc_auc_mean": 0.85,
                "roc_auc_std": 0.0,
                "train_time_mean": 0.01,
            },
        },
        "VQC": {
            "model_type": "quantum",
            "roc_fpr": [0.0, 0.3, 1.0],
            "roc_tpr": [0.0, 0.75, 1.0],
            "param_count": 24,
            "summary": {
                "recall_mean": 0.75,
                "recall_std": 0.05,
                "f1_mean": 0.72,
                "f1_std": 0.04,
                "roc_auc_mean": 0.80,
                "roc_auc_std": 0.03,
                "train_time_mean": 2.5,
            },
        },
    }

    roc_path = tmp_path / "roc.png"
    f1_path = tmp_path / "f1.png"
    time_path = tmp_path / "time.png"
    size_path = tmp_path / "size.png"

    plot_roc_curves_overlay(dummy_results, roc_path)
    plot_f1_recall_comparison(dummy_results, f1_path)
    plot_training_time_comparison(dummy_results, time_path)
    plot_model_size_comparison(dummy_results, size_path)

    for p in [roc_path, f1_path, time_path, size_path]:
        assert p.exists()
        assert p.stat().st_size > 0


def test_select_and_save_production_model(tmp_path):
    """Verify production model selection by recall-weighted F1."""
    X = np.random.randn(20, 4)
    y = np.array([0] * 10 + [1] * 10)
    lr = LogisticRegression().fit(X, y)

    dummy_results = {
        "Model A": {
            "model_type": "classical",
            "best_estimator": lr,
            "param_count": 5,
            "summary": {
                "recall_weighted_f1_mean": 0.70,
                "recall_mean": 0.65,
                "f1_mean": 0.75,
                "roc_auc_mean": 0.78,
            },
        },
        "Model B": {
            "model_type": "classical",
            "best_estimator": lr,
            "param_count": 5,
            "summary": {
                "recall_weighted_f1_mean": 0.85,
                "recall_mean": 0.90,
                "f1_mean": 0.80,
                "roc_auc_mean": 0.88,
            },
        },
    }

    winner, meta = select_and_save_production_model(
        dummy_results, X, y, n_qubits=4, models_dir=tmp_path, results_dir=tmp_path
    )
    assert winner == "Model B"
    assert meta["production_model_name"] == "Model B"
    assert (tmp_path / "production_model.joblib").exists()
    assert (tmp_path / "production_model_metadata.json").exists()
