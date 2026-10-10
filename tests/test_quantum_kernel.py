"""Unit tests for Quantum Support Vector Classifier module (src/quantum/quantum_kernel.py)."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.quantum.quantum_kernel import (
    QSVCConfig,
    build_quantum_kernel,
    build_zz_feature_map,
    compute_kernel_alignment,
    compute_or_load_kernel_matrices,
    compute_qsvc_metrics,
    train_and_tune_qsvc,
    visualize_and_compare_kernels,
)


def test_qsvc_config_validation():
    """Verify configuration validation bounds."""
    cfg = QSVCConfig(n_qubits=4, reps=2, entanglement="linear")
    assert cfg.n_qubits == 4
    assert cfg.reps == 2
    assert cfg.entanglement == "linear"

    with pytest.raises(ValueError, match="n_qubits"):
        QSVCConfig(n_qubits=1)

    with pytest.raises(ValueError, match="reps"):
        QSVCConfig(reps=0)

    with pytest.raises(ValueError, match="entanglement"):
        QSVCConfig(entanglement="invalid_topology")


def test_build_zz_feature_map():
    """Verify ZZ feature map circuits for linear and full entanglement."""
    circ_linear = build_zz_feature_map(n_qubits=3, reps=2, entanglement="linear")
    assert circ_linear.num_qubits == 3

    circ_full = build_zz_feature_map(n_qubits=3, reps=1, entanglement="full")
    assert circ_full.num_qubits == 3


def test_build_quantum_kernel():
    """Verify kernel constructors."""
    circ = build_zz_feature_map(n_qubits=2, reps=1, entanglement="linear")
    kernel_q = build_quantum_kernel(circ, kernel_type="fidelity_quantum")
    assert kernel_q is not None

    kernel_sv = build_quantum_kernel(circ, kernel_type="fidelity_statevector")
    assert kernel_sv is not None


def test_kernel_caching_and_reloading(tmp_path):
    """Verify disk caching of kernel matrices and cache hits on second invocation."""
    cache_dir = tmp_path / "cache"
    cfg = QSVCConfig(
        n_qubits=2,
        reps=1,
        kernel_type="fidelity_statevector",
        cache_dir=cache_dir,
        use_cache=True,
    )

    circ = build_zz_feature_map(n_qubits=2, reps=1)
    kernel = build_quantum_kernel(circ, kernel_type="fidelity_statevector")

    X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    X_val = np.array([[0.7, 0.8]])
    X_test = np.array([[0.9, 1.0]])

    # 1st call: compute and save to cache
    K_tr1, K_v1, K_te1, meta1 = compute_or_load_kernel_matrices(
        kernel=kernel,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        config=cfg,
    )
    assert meta1["cache_status"] == "computed_and_cached"
    assert cfg.cache_filename.exists()
    assert K_tr1.shape == (3, 3)
    assert K_v1.shape == (1, 3)
    assert K_te1.shape == (1, 3)

    # 2nd call: load directly from disk cache
    K_tr2, K_v2, K_te2, meta2 = compute_or_load_kernel_matrices(
        kernel=kernel,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        config=cfg,
    )
    assert meta2["cache_status"] == "hit"
    np.testing.assert_allclose(K_tr1, K_tr2, atol=1e-7)
    np.testing.assert_allclose(K_v1, K_v2, atol=1e-7)
    np.testing.assert_allclose(K_te1, K_te2, atol=1e-7)


def test_visualize_and_compare_kernels(tmp_path):
    """Verify kernel heatmap generation and comparison with RBF."""
    heatmap_path = tmp_path / "heatmap_test.png"
    K_quantum = np.eye(8)
    X_train = np.random.uniform(0.0, np.pi, size=(8, 3))

    fig = visualize_and_compare_kernels(
        K_quantum=K_quantum,
        X_train=X_train,
        save_path=heatmap_path,
    )

    assert heatmap_path.exists()
    assert heatmap_path.stat().st_size > 0


def test_train_and_tune_qsvc():
    """Verify SVC fitting with kernel='precomputed' and hyperparameter tuning."""
    K_train = np.eye(6)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    K_val = np.ones((2, 6)) * 0.5
    y_val = np.array([0, 1])

    cfg = QSVCConfig(tune_c=True, candidate_c_values=[0.1, 1.0, 10.0])
    svc, best_c, meta = train_and_tune_qsvc(
        K_train=K_train,
        y_train=y_train,
        K_val=K_val,
        y_val=y_val,
        config=cfg,
    )

    assert svc is not None
    assert best_c in [0.1, 1.0, 10.0]
    assert "candidates_evaluated" in meta


def test_compute_qsvc_metrics():
    """Verify metrics calculation format."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_qsvc_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
