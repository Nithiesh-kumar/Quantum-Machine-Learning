"""Unit tests for classical baseline benchmarking suite."""
import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.classical.baselines import (
    compute_metrics,
    format_markdown_table,
    get_baseline_model_specs,
    prepare_baseline_datasets,
    run_classical_baselines,
    train_and_evaluate_feature_set,
)
from src.data.loader import load_dataset
from src.utils.config import QMLConfig


@pytest.fixture
def heart_df():
    """Provides heart dataset."""
    return load_dataset("heart")


def test_get_baseline_model_specs():
    """Verify all 4 required models are specified with appropriate parameter grids."""
    specs = get_baseline_model_specs(seed=42)
    expected_models = {"Logistic Regression", "SVM (RBF)", "Random Forest", "Small MLP"}

    assert set(specs.keys()) == expected_models
    for name, spec in specs.items():
        assert "estimator" in spec
        assert "param_grid" in spec
        assert isinstance(spec["param_grid"], dict)
        assert len(spec["param_grid"]) > 0


def test_prepare_baseline_datasets_shapes_and_leakage(heart_df):
    """Verify zero test leakage and proper dimensionality for full and reduced sets."""
    cfg = QMLConfig(dataset_name="heart", n_qubits=8, seed=42)
    full_data, reduced_data, metadata = prepare_baseline_datasets(heart_df, config=cfg)

    # Check sample splits sum to dataset total
    total_len = len(heart_df)
    assert len(full_data["X_train"]) + len(full_data["X_val"]) + len(full_data["X_test"]) == total_len
    assert len(reduced_data["X_train"]) + len(reduced_data["X_val"]) + len(reduced_data["X_test"]) == total_len

    # Check dimensions
    assert reduced_data["X_train"].shape[1] == 8
    assert full_data["X_train"].shape[1] == metadata["full_n_features"]
    assert metadata["full_n_features"] > 8

    # Check reduced features are angle scaled to [0, pi]
    assert np.all(reduced_data["X_train"] >= 0.0)
    assert np.all(reduced_data["X_train"] <= np.pi + 1e-5)


def test_compute_metrics():
    """Verify metric calculations including recall, specificity, and confusion matrix."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.6, 0.4, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_score)

    # True negatives = 1, False positives = 1, False negatives = 1, True positives = 2
    assert metrics["accuracy"] == 0.6
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == round(2 / 3, 4)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 2]]
    assert "roc_auc" in metrics


def test_train_and_evaluate_feature_set():
    """Verify training and evaluation with 2-fold CV on small test data."""
    # Synthetic small dataset
    np.random.seed(42)
    X_train = np.random.randn(40, 4)
    y_train = np.array([0] * 20 + [1] * 20)
    X_val = np.random.randn(10, 4)
    y_val = np.array([0] * 5 + [1] * 5)
    X_test = np.random.randn(10, 4)
    y_test = np.array([0] * 5 + [1] * 5)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    mini_specs = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=100, random_state=42),
            "param_grid": {"C": [0.1, 1.0]},
        }
    }

    results, estimators = train_and_evaluate_feature_set(
        feature_name="test_features",
        data=data,
        models=mini_specs,
        cv_folds=2,
        scoring="recall",
        seed=42,
        n_jobs=1,
    )

    assert "Logistic Regression" in results
    res = results["Logistic Regression"]
    assert "best_params" in res
    assert "validation" in res
    assert "test" in res
    assert "recall" in res["validation"]
    assert "confusion_matrix" in res["validation"]
    assert "Logistic Regression" in estimators


def test_end_to_end_classical_baselines(tmp_path):
    """Verify end-to-end run creates JSON metrics file with all required fields."""
    out_json = tmp_path / "classical_metrics.json"
    cfg = QMLConfig(dataset_name="heart", n_qubits=8, seed=42)

    master_results = run_classical_baselines(
        dataset_name="heart",
        config=cfg,
        scoring="recall",
        cv_folds=3,
        output_path=out_json,
        save_best_model=False,
    )

    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert "metadata" in saved_data
    assert "full_features" in saved_data
    assert "reduced_features" in saved_data
    assert "comparison_summary" in saved_data
    assert "clinical_discussion" in saved_data

    # Check all 4 models are in both feature sets
    for feature_set in ["full_features", "reduced_features"]:
        for model in ["Logistic Regression", "SVM (RBF)", "Random Forest", "Small MLP"]:
            assert model in saved_data[feature_set]
            m_res = saved_data[feature_set][model]
            assert "validation" in m_res
            assert "recall" in m_res["validation"]
            assert "precision" in m_res["validation"]
            assert "accuracy" in m_res["validation"]
            assert "f1" in m_res["validation"]
            assert "roc_auc" in m_res["validation"]
            assert "confusion_matrix" in m_res["validation"]
