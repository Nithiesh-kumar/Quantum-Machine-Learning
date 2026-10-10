"""Evaluation and comparative benchmarking module for Quantum ML Health Risk System."""

try:
    from src.evaluation.compare import (
        calculate_metrics,
        run_ablation_experiments,
        run_benchmark_models,
        run_comparison_pipeline,
    )
    __all__ = [
        "calculate_metrics",
        "run_benchmark_models",
        "run_ablation_experiments",
        "run_comparison_pipeline",
    ]
except ImportError:
    # Heavy training dependencies (qiskit, matplotlib) not installed in deployment
    __all__ = []
