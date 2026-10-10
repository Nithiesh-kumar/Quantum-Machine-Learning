"""Unit tests for Variational Quantum Classifier module (src/quantum/vqc.py)."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.quantum.vqc import (
    VariationalQuantumClassifier,
    VQCConfig,
    compute_vqc_metrics,
    initialize_weights,
    run_vqc_pipeline,
)


def test_vqc_config_validation():
    """Verify configuration validation bounds."""
    # Valid config
    cfg = VQCConfig(n_qubits=4, n_layers=2)
    assert cfg.n_qubits == 4
    assert cfg.n_layers == 2

    # Invalid qubits
    with pytest.raises(ValueError, match="n_qubits"):
        VQCConfig(n_qubits=10)

    # Invalid layers
    with pytest.raises(ValueError, match="n_layers"):
        VQCConfig(n_layers=0)

    # Invalid feature map
    with pytest.raises(ValueError, match="Unsupported feature_map"):
        VQCConfig(feature_map="quantum_fourier_transform")

    # Invalid ansatz
    with pytest.raises(ValueError, match="Unsupported ansatz"):
        VQCConfig(ansatz="random_clifford")


def test_weight_initialization():
    """Verify weight shapes for different ansatz architectures."""
    w_se = initialize_weights("strongly_entangling", n_layers=3, n_qubits=4)
    assert w_se.shape == (3, 4, 3)

    w_ra = initialize_weights("real_amplitudes", n_layers=2, n_qubits=5)
    assert w_ra.shape == (3, 5)


def test_vqc_probability_range():
    """Verify that predictions always output valid probabilities in [0, 1]."""
    cfg = VQCConfig(n_qubits=4, n_layers=1, feature_map="angle", ansatz="strongly_entangling")
    vqc = VariationalQuantumClassifier(config=cfg)

    # Manually assign weights to test prediction without fitting
    vqc.weights_ = initialize_weights(cfg.ansatz, cfg.n_layers, cfg.n_qubits)
    X_test = np.random.uniform(0.0, np.pi, size=(5, 4))

    probs = vqc.predict_proba(X_test)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Check probabilities sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)

    preds = vqc.predict(X_test)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})


def test_draw_circuit(tmp_path):
    """Verify circuit diagram drawing generates an image file."""
    img_path = tmp_path / "test_circuit.png"
    cfg = VQCConfig(n_qubits=3, n_layers=1, circuit_save_path=img_path)
    vqc = VariationalQuantumClassifier(config=cfg)
    vqc.draw_circuit(save_path=img_path)

    assert img_path.exists()
    assert img_path.stat().st_size > 0


def test_vqc_fit_and_early_stopping(tmp_path):
    """Verify VQC end-to-end fit, early stopping, and history tracking."""
    model_path = tmp_path / "vqc_test.pt"
    curves_path = tmp_path / "vqc_curves_test.png"
    metrics_path = tmp_path / "vqc_metrics_test.json"

    cfg = VQCConfig(
        n_qubits=3,
        n_layers=1,
        feature_map="angle",
        ansatz="real_amplitudes",
        epochs=4,
        batch_size=8,
        patience=2,
        model_save_path=model_path,
        curves_save_path=curves_path,
        metrics_save_path=metrics_path,
    )

    # Synthetic separable data
    np.random.seed(42)
    X_train = np.random.uniform(0.0, np.pi, size=(16, 3))
    y_train = np.array([0] * 8 + [1] * 8)
    X_val = np.random.uniform(0.0, np.pi, size=(8, 3))
    y_val = np.array([0] * 4 + [1] * 4)

    vqc = VariationalQuantumClassifier(config=cfg)
    vqc.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    assert vqc.weights_ is not None
    assert len(vqc.history_["train_loss"]) > 0
    assert len(vqc.history_["val_loss"]) > 0
    assert model_path.exists()
    assert curves_path.exists()

    # Test load
    loaded_vqc = VariationalQuantumClassifier.load_model(model_path)
    np.testing.assert_allclose(vqc.weights_, loaded_vqc.weights_, atol=1e-6)


def test_noisy_simulator_execution():
    """Verify depolarizing noise model execution on default.mixed."""
    cfg = VQCConfig(
        n_qubits=2,
        n_layers=1,
        noise_model=True,
        noise_probability=0.02,
        diff_method="parameter-shift",
    )
    vqc = VariationalQuantumClassifier(config=cfg)
    vqc.weights_ = initialize_weights(cfg.ansatz, cfg.n_layers, cfg.n_qubits)

    X = np.array([[0.5, 1.0]])
    probs = vqc.predict_proba(X)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_compute_vqc_metrics():
    """Verify clinical metrics computation."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.7, 0.4, 0.8])

    metrics = compute_vqc_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
