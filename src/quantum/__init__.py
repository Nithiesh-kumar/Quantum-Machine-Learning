"""Quantum Machine Learning module for Health Risk Prediction."""

from src.quantum.feature_maps import angle_feature_map, get_feature_map, zz_feature_map
from src.quantum.vqc import VariationalQuantumClassifier, VQCConfig, run_vqc_pipeline

try:
    from src.quantum.quantum_kernel import (
        QSVCConfig,
        build_quantum_kernel,
        build_zz_feature_map,
        compute_or_load_kernel_matrices,
        run_qsvc_pipeline,
        visualize_and_compare_kernels,
    )
except ImportError:
    # Qiskit and matplotlib are not included in lightweight deployment
    QSVCConfig = None
    build_quantum_kernel = None
    build_zz_feature_map = None
    compute_or_load_kernel_matrices = None
    run_qsvc_pipeline = None
    visualize_and_compare_kernels = None

__all__ = [
    # VQC
    "VariationalQuantumClassifier",
    "VQCConfig",
    "run_vqc_pipeline",
    "angle_feature_map",
    "zz_feature_map",
    "get_feature_map",
    # QSVC
    "QSVCConfig",
    "run_qsvc_pipeline",
    "build_zz_feature_map",
    "build_quantum_kernel",
    "compute_or_load_kernel_matrices",
    "visualize_and_compare_kernels",
]
