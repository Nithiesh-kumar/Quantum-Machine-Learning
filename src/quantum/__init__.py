"""Quantum Machine Learning module for Health Risk Prediction."""

from src.quantum.feature_maps import angle_feature_map, get_feature_map, zz_feature_map
from src.quantum.quantum_kernel import (
    QSVCConfig,
    build_quantum_kernel,
    build_zz_feature_map,
    compute_or_load_kernel_matrices,
    run_qsvc_pipeline,
    visualize_and_compare_kernels,
)
from src.quantum.vqc import VariationalQuantumClassifier, VQCConfig, run_vqc_pipeline

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
