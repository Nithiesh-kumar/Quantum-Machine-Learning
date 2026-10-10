"""Classical baseline models module."""
from src.classical.baselines import (
    compute_metrics,
    format_markdown_table,
    get_baseline_model_specs,
    prepare_baseline_datasets,
    run_classical_baselines,
    train_and_evaluate_feature_set,
)

__all__ = [
    "compute_metrics",
    "format_markdown_table",
    "get_baseline_model_specs",
    "prepare_baseline_datasets",
    "run_classical_baselines",
    "train_and_evaluate_feature_set",
]
