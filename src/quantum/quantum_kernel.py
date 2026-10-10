"""
Quantum Support Vector Classifier (QSVC) with Qiskit Machine Learning.

Key Capabilities:
- ZZFeatureMap (reps=2, linear/full entanglement configurable) paired with FidelityQuantumKernel.
- Precomputed kernel matrix evaluation with scikit-learn's SVC(kernel="precomputed").
- Persistent disk caching of quantum kernel matrices to avoid redundant circuit evaluations on small datasets.
- Kernel comparison heatmap visualizing the Quantum ZZ Kernel vs. Classical RBF Kernel with alignment analysis.
- End-to-end evaluation on validation and test sets matching the standardized clinical metrics schema.
- Notes and adaptations for Qiskit 2.x and Qiskit Machine Learning 0.9.x API changes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

# Ensure UTF-8 console output across operating systems
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress benign future/deprecation warnings from scikit-learn and qiskit
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

# Qiskit 2.x and Qiskit Machine Learning modern imports
try:
    # Qiskit 2.1+ modern function replacement
    from qiskit.circuit.library import zz_feature_map
    IS_ZZ_FUNCTION = True
except ImportError:
    # Fallback for earlier Qiskit versions
    from qiskit.circuit.library import ZZFeatureMap as zz_feature_map
    IS_ZZ_FUNCTION = False

import qiskit
import qiskit_machine_learning
from qiskit_machine_learning.kernels import (
    BaseKernel,
    FidelityQuantumKernel,
    FidelityStatevectorKernel,
)

from src.data.loader import load_dataset
from src.data.preprocess import prepare_and_fit_data
from src.utils.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SEED,
    QMLConfig,
    default_config,
)
from src.utils.logger import get_logger

logger = get_logger("qsvc")


# ---------------------------------------------------------------------------
# Qiskit Version & API Changes Documentation
# ---------------------------------------------------------------------------
QISKIT_API_CHANGES_SUMMARY: Dict[str, str] = {
    "qiskit_version": getattr(qiskit, "__version__", "unknown"),
    "qiskit_machine_learning_version": getattr(qiskit_machine_learning, "__version__", "unknown"),
    "zz_feature_map_migration": (
        "In Qiskit 2.1+, the class 'qiskit.circuit.library.ZZFeatureMap' was deprecated in favor of the "
        "functional constructor 'qiskit.circuit.library.zz_feature_map'. The function returns a standard "
        "QuantumCircuit rather than a BlueprintCircuit, avoiding legacy decomposition overhead."
    ),
    "fidelity_quantum_kernel_architecture": (
        "In qiskit-machine-learning >= 0.7.0, the legacy backend-based 'QuantumKernel' class was removed. "
        "It was replaced by 'FidelityQuantumKernel' and 'FidelityStatevectorKernel', which rely on Qiskit's "
        "primitives architecture (BaseStateFidelity / ComputeUncompute) for shot-based or statevector fidelity evaluation."
    ),
    "qiskit_primitives_v2": (
        "Qiskit 2.x standardized on SamplerV2 and EstimatorV2 primitives, with ComputeUncompute managing "
        "quantum state overlap circuits |<psi(x)|psi(y)>|^2 without requiring qiskit-aer or legacy Aer backends."
    ),
}


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------
@dataclass
class QSVCConfig:
    """Configuration settings for Quantum Support Vector Classifier (QSVC)."""
    dataset_name: str = "heart"
    n_qubits: int = 4                    # Number of quantum features / qubits (4–6 recommended)
    reps: int = 2                        # Repetitions of ZZFeatureMap (default 2)
    entanglement: str = "linear"         # 'linear' or 'full'
    C: float = 1.0                       # Regularization parameter for SVC
    tune_c: bool = True                  # True to perform validation search over candidate C values
    candidate_c_values: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    scoring_metric: str = "recall"       # Optimization objective ('recall' prioritized for clinical safety)
    kernel_type: str = "fidelity_quantum" # 'fidelity_quantum' or 'fidelity_statevector'
    use_cache: bool = True               # Cache precomputed kernel matrices to disk
    force_recompute: bool = False        # If True, recomputes kernels ignoring existing cache
    seed: int = SEED

    # Destination Paths
    cache_dir: Path = field(default_factory=lambda: PROCESSED_DATA_DIR)
    model_save_path: Path = field(default_factory=lambda: MODELS_DIR / "qsvc_model.joblib")
    heatmap_save_path: Path = field(default_factory=lambda: RESULTS_DIR / "qsvc_kernel_heatmap.png")
    metrics_save_path: Path = field(default_factory=lambda: RESULTS_DIR / "qsvc_metrics.json")

    def __post_init__(self):
        if not (2 <= self.n_qubits <= 8):
            raise ValueError(f"n_qubits ({self.n_qubits}) must be between 2 and 8.")
        if self.reps < 1:
            raise ValueError(f"reps ({self.reps}) must be >= 1.")
        if self.entanglement not in ("linear", "full", "circular"):
            raise ValueError(f"Unsupported entanglement '{self.entanglement}'. Choose 'linear' or 'full'.")
        if self.kernel_type not in ("fidelity_quantum", "fidelity_statevector"):
            raise ValueError(f"Unsupported kernel_type '{self.kernel_type}'.")

        self.cache_dir = Path(self.cache_dir)
        self.model_save_path = Path(self.model_save_path)
        self.heatmap_save_path = Path(self.heatmap_save_path)
        self.metrics_save_path = Path(self.metrics_save_path)

    @property
    def cache_filename(self) -> Path:
        """Constructs a deterministic cache filename for kernel matrices."""
        name = f"kernel_cache_{self.dataset_name}_{self.n_qubits}q_r{self.reps}_{self.entanglement}_{self.kernel_type}.npz"
        return self.cache_dir / name


# ---------------------------------------------------------------------------
# Feature Map & Kernel Builders
# ---------------------------------------------------------------------------
def build_zz_feature_map(
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
) -> Any:
    """
    Constructs a Qiskit ZZFeatureMap circuit, adapting to Qiskit 2.x API changes.

    Args:
        n_qubits: Number of features / qubits.
        reps: Number of repetitions (layers) of the feature map.
        entanglement: Entanglement topology ('linear' or 'full').

    Returns:
        QuantumCircuit implementing the ZZ-interaction feature encoding.
    """
    if IS_ZZ_FUNCTION:
        # Modern Qiskit 2.1+ functional API
        circuit = zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    else:
        # Legacy class-based API
        circuit = zz_feature_map(
            feature_dimension=n_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    return circuit


def build_quantum_kernel(
    feature_map: Any,
    kernel_type: str = "fidelity_quantum",
) -> BaseKernel:
    """
    Instantiates a Qiskit Machine Learning quantum kernel.

    Args:
        feature_map: Quantum circuit feature map.
        kernel_type: 'fidelity_quantum' (standard) or 'fidelity_statevector' (fast simulation).

    Returns:
        Configured BaseKernel instance.
    """
    if kernel_type == "fidelity_quantum":
        return FidelityQuantumKernel(feature_map=feature_map)
    elif kernel_type == "fidelity_statevector":
        return FidelityStatevectorKernel(feature_map=feature_map)
    else:
        raise ValueError(f"Unknown kernel_type '{kernel_type}'.")


# ---------------------------------------------------------------------------
# Disk Caching and Evaluation of Kernel Matrices
# ---------------------------------------------------------------------------
def compute_or_load_kernel_matrices(
    kernel: BaseKernel,
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    config: Optional[QSVCConfig] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Computes or loads precomputed quantum kernel matrices to/from disk cache.

    Because clinical tabular datasets are compact, caching $K_{train}$, $K_{val}$,
    and $K_{test}$ avoids redundant circuit evaluations and enables instantaneous
    hyperparameter tuning.

    Args:
        kernel: Configured quantum kernel object.
        X_train: Training feature array (N_train, n_qubits).
        X_val: Optional validation feature array (N_val, n_qubits).
        X_test: Optional test feature array (N_test, n_qubits).
        config: QSVC configuration dataclass.

    Returns:
        Tuple of (K_train, K_val, K_test, cache_metadata).
    """
    cfg = config or QSVCConfig()
    cache_path = cfg.cache_filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Attempt Cache Load
    if cfg.use_cache and not cfg.force_recompute and cache_path.exists():
        try:
            logger.info("Loading precomputed quantum kernel matrices from cache: %s", cache_path)
            cached_data = np.load(cache_path)
            K_train = cached_data["K_train"]
            K_val = cached_data["K_val"] if "K_val" in cached_data else None
            K_test = cached_data["K_test"] if "K_test" in cached_data else None

            # Verify dimensionality matches input data
            if K_train.shape == (len(X_train), len(X_train)):
                meta = {
                    "cache_status": "hit",
                    "cache_path": str(cache_path),
                    "computation_time_seconds": 0.0,
                    "K_train_shape": list(K_train.shape),
                }
                logger.info("Cache hit! Loaded K_train (%s), K_val (%s), K_test (%s).",
                            K_train.shape,
                            K_val.shape if K_val is not None else None,
                            K_test.shape if K_test is not None else None)
                return K_train, K_val, K_test, meta
            else:
                logger.warning("Cached matrix dimensions (%s) do not match current X_train (%d, %d). Recomputing.",
                               K_train.shape, len(X_train), len(X_train))
        except Exception as e:
            logger.warning("Failed to load cached kernel matrices (%s). Recomputing.", e)

    # 2. Compute Kernel Matrices
    logger.info("Evaluating quantum kernel matrix on training set (%d samples)...", len(X_train))
    t0 = time.time()
    K_train = kernel.evaluate(X_train)
    t_train = time.time() - t0
    logger.info("Evaluated K_train in %.2fs. Shape: %s, Min: %.4f, Max: %.4f, Mean: %.4f",
                t_train, K_train.shape, K_train.min(), K_train.max(), K_train.mean())

    K_val = None
    if X_val is not None:
        logger.info("Evaluating rectangular quantum kernel matrix on validation set (%d x %d)...",
                    len(X_val), len(X_train))
        t1 = time.time()
        K_val = kernel.evaluate(X_val, X_train)
        logger.info("Evaluated K_val in %.2fs. Shape: %s", time.time() - t1, K_val.shape)

    K_test = None
    if X_test is not None:
        logger.info("Evaluating rectangular quantum kernel matrix on test set (%d x %d)...",
                    len(X_test), len(X_train))
        t2 = time.time()
        K_test = kernel.evaluate(X_test, X_train)
        logger.info("Evaluated K_test in %.2fs. Shape: %s", time.time() - t2, K_test.shape)

    total_time = round(time.time() - t0, 2)

    # 3. Cache to Disk
    if cfg.use_cache:
        save_dict = {"K_train": K_train}
        if K_val is not None:
            save_dict["K_val"] = K_val
        if K_test is not None:
            save_dict["K_test"] = K_test

        np.savez_compressed(cache_path, **save_dict)
        logger.info("Cached quantum kernel matrices to: %s (Total eval time: %.2fs)", cache_path, total_time)

    meta = {
        "cache_status": "computed_and_cached",
        "cache_path": str(cache_path),
        "computation_time_seconds": total_time,
        "K_train_shape": list(K_train.shape),
    }

    return K_train, K_val, K_test, meta


# ---------------------------------------------------------------------------
# Kernel Heatmap Visualization & Comparison with RBF
# ---------------------------------------------------------------------------
def compute_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    Computes the Frobenius inner-product kernel alignment between two matrices:
    A(K1, K2) = <K1, K2>_F / (||K1||_F * ||K2||_F).
    """
    norm1 = np.linalg.norm(K1, "fro")
    norm2 = np.linalg.norm(K2, "fro")
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.sum(K1 * K2) / (norm1 * norm2))


def visualize_and_compare_kernels(
    K_quantum: np.ndarray,
    X_train: np.ndarray,
    save_path: Optional[Path] = None,
    gamma: Optional[float] = None,
    max_display: int = 50,
) -> plt.Figure:
    """
    Generates a high-quality comparison heatmap between the Quantum ZZ Kernel
    and the Classical RBF Kernel, along with difference residual analysis.

    Args:
        K_quantum: Precomputed quantum kernel matrix (N, N).
        X_train: Training features used to generate classical RBF kernel.
        save_path: Destination PNG image path.
        gamma: RBF kernel bandwidth parameter (defaults to 1 / n_features).
        max_display: Maximum number of rows/columns to render on the heatmap for clarity.

    Returns:
        Matplotlib Figure.
    """
    target_path = Path(save_path or RESULTS_DIR / "qsvc_kernel_heatmap.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    n_features = X_train.shape[1]
    rbf_gamma = gamma if gamma is not None else (1.0 / n_features)
    K_rbf = rbf_kernel(X_train, gamma=rbf_gamma)

    # Subset display window if matrix is large for clean visual readability
    n_samples = min(len(X_train), max_display)
    K_q_sub = K_quantum[:n_samples, :n_samples]
    K_rbf_sub = K_rbf[:n_samples, :n_samples]
    K_diff_sub = np.abs(K_q_sub - K_rbf_sub)

    # Compute alignment across the full matrices
    alignment = compute_kernel_alignment(K_quantum, K_rbf)
    mean_q_offdiag = float(np.mean(K_quantum[~np.eye(len(K_quantum), dtype=bool)]))
    mean_rbf_offdiag = float(np.mean(K_rbf[~np.eye(len(K_rbf), dtype=bool)]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.suptitle(
        f"Kernel Matrix Comparison: Quantum ZZ Kernel vs. Classical RBF Kernel\n"
        f"Alignment Score: {alignment:.4f} | Mean Off-Diagonal (Quantum: {mean_q_offdiag:.3f}, RBF: {mean_rbf_offdiag:.3f})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # 1. Quantum ZZ Kernel
    im1 = axes[0].imshow(K_q_sub, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title(f"Quantum ZZ Kernel Matrix ({n_samples}x{n_samples})\n(Fidelity State Overlap)", fontsize=11)
    axes[0].set_xlabel("Sample Index", fontsize=10)
    axes[0].set_ylabel("Sample Index", fontsize=10)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Fidelity Overlap")

    # 2. Classical RBF Kernel
    im2 = axes[1].imshow(K_rbf_sub, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title(f"Classical RBF Kernel Matrix ({n_samples}x{n_samples})\n(gamma={rbf_gamma:.3f})", fontsize=11)
    axes[1].set_xlabel("Sample Index", fontsize=10)
    axes[1].set_ylabel("Sample Index", fontsize=10)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="RBF Similarity")

    # 3. Absolute Difference Residual
    im3 = axes[2].imshow(K_diff_sub, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")
    axes[2].set_title(f"Absolute Residual |K_quantum - K_rbf|\n(Max Diff: {K_diff_sub.max():.3f})", fontsize=11)
    axes[2].set_xlabel("Sample Index", fontsize=10)
    axes[2].set_ylabel("Sample Index", fontsize=10)
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="|Difference|")

    plt.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved kernel comparison heatmap to: %s", target_path)

    return fig


# ---------------------------------------------------------------------------
# Training & Tuning Precomputed QSVC
# ---------------------------------------------------------------------------
def train_and_tune_qsvc(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[QSVCConfig] = None,
) -> Tuple[SVC, float, Dict[str, Any]]:
    """
    Fits and tunes a Support Vector Classifier with kernel='precomputed'.

    Args:
        K_train: Precomputed training kernel matrix (N_train, N_train).
        y_train: Training labels (N_train,).
        K_val: Precomputed rectangular validation kernel matrix (N_val, N_train).
        y_val: Validation labels (N_val,).
        config: QSVC configuration settings.

    Returns:
        Tuple of (fitted_svc, best_c, val_tuning_records).
    """
    cfg = config or QSVCConfig()
    candidate_c_values = cfg.candidate_c_values if cfg.tune_c and K_val is not None and y_val is not None else [cfg.C]

    best_score = -1.0
    best_c = cfg.C
    best_svc: Optional[SVC] = None
    tuning_log: List[Dict[str, Any]] = []

    logger.info("Training QSVC with kernel='precomputed'. Candidate C values: %s", candidate_c_values)

    for c_val in candidate_c_values:
        # Fit SVC on training kernel matrix
        svc = SVC(
            C=c_val,
            kernel="precomputed",
            probability=True,
            random_state=cfg.seed,
        )
        svc.fit(K_train, y_train)

        # Evaluate on validation set if provided
        if K_val is not None and y_val is not None:
            val_preds = svc.predict(K_val)
            val_probs = svc.predict_proba(K_val)[:, 1] if hasattr(svc, "predict_proba") else svc.decision_function(K_val)

            rec = float(recall_score(y_val, val_preds, zero_division=0))
            f1 = float(f1_score(y_val, val_preds, zero_division=0))
            acc = float(accuracy_score(y_val, val_preds))
            try:
                auc = float(roc_auc_score(y_val, val_probs))
            except Exception:
                auc = 0.5

            record = {"C": c_val, "val_recall": rec, "val_f1": f1, "val_acc": acc, "val_roc_auc": auc}
            tuning_log.append(record)

            # Prioritize recall (with AUC and F1 tiebreakers for clinical safety)
            eval_score = rec if cfg.scoring_metric == "recall" else auc
            if eval_score > best_score:
                best_score = eval_score
                best_c = c_val
                best_svc = svc

            logger.info("Candidate C=%.3f -> Val Recall: %.4f, Val AUC: %.4f, Val F1: %.4f, Val Acc: %.4f",
                        c_val, rec, auc, f1, acc)
        else:
            best_c = c_val
            best_svc = svc

    logger.info("Optimal regularization parameter selected: C=%.3f", best_c)

    # Refit final best model
    if best_svc is None:
        best_svc = SVC(C=best_c, kernel="precomputed", probability=True, random_state=cfg.seed)
        best_svc.fit(K_train, y_train)

    tuning_summary = {
        "best_C": best_c,
        "optimization_metric": cfg.scoring_metric,
        "candidates_evaluated": tuning_log,
    }

    return best_svc, best_c, tuning_summary


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_qsvc_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Computes standardized clinical performance metrics matching the system schema."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = 0.5
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


# ---------------------------------------------------------------------------
# Master QSVC Pipeline Execution
# ---------------------------------------------------------------------------
def run_qsvc_pipeline(
    config: Optional[QSVCConfig] = None,
    dataset_name: str = "heart",
) -> Dict[str, Any]:
    """
    Executes the full Quantum Support Vector Classifier (QSVC) pipeline:
      1. Builds ZZFeatureMap + FidelityQuantumKernel.
      2. Computes and disk-caches kernel matrices for train, val, and test partitions.
      3. Fits SVC with kernel='precomputed' and tunes regularization parameter C.
      4. Visualizes and saves the quantum vs. classical RBF kernel comparison heatmap.
      5. Evaluates test set performance with identical metrics schema.
      6. Exports artifacts to /results/qsvc_metrics.json, heatmap PNG, and models/qsvc_model.joblib.

    Args:
        config: QSVC configuration dataclass.
        dataset_name: Target clinical dataset ('heart' or 'diabetes').

    Returns:
        Dictionary containing execution metadata, validation metrics, and test metrics.
    """
    cfg = config or QSVCConfig(dataset_name=dataset_name)

    logger.info("=" * 70)
    logger.info("QUANTUM SUPPORT VECTOR CLASSIFIER (QSVC) PIPELINE")
    logger.info("=" * 70)
    logger.info("Dataset: %s | Qubits: %d | Feature Map: ZZFeatureMap (reps=%d, %s)",
                cfg.dataset_name, cfg.n_qubits, cfg.reps, cfg.entanglement)
    logger.info("Kernel: %s | Precomputed SVC C: %s", cfg.kernel_type, cfg.C)

    # 1. Load Preprocessed Clinical Data
    processed_npz = PROCESSED_DATA_DIR / f"{cfg.dataset_name}_processed.npz"
    if processed_npz.exists():
        logger.info("Loading preprocessed dataset from: %s", processed_npz)
        data = np.load(processed_npz)
        X_train = data["X_train"][:, :cfg.n_qubits]
        y_train = data["y_train"]
        X_val = data["X_val"][:, :cfg.n_qubits]
        y_val = data["y_val"]
        X_test = data["X_test"][:, :cfg.n_qubits]
        y_test = data["y_test"]
    else:
        logger.info("Processed data not found. Preparing raw '%s' data...", cfg.dataset_name)
        df_raw = load_dataset(cfg.dataset_name)
        qml_cfg = QMLConfig(dataset_name=cfg.dataset_name, n_qubits=cfg.n_qubits, seed=cfg.seed)
        X_tr, y_tr, X_v, y_v, X_te, y_te, _ = prepare_and_fit_data(df_raw, config=qml_cfg)
        X_train, y_train, X_val, y_val, X_test, y_test = (
            X_tr[:, :cfg.n_qubits], y_tr, X_v[:, :cfg.n_qubits], y_v, X_te[:, :cfg.n_qubits], y_te
        )

    pipeline_start = time.time()

    # 2. Build Quantum Feature Map & Kernel
    feature_map = build_zz_feature_map(
        n_qubits=cfg.n_qubits,
        reps=cfg.reps,
        entanglement=cfg.entanglement,
    )
    quantum_kernel = build_quantum_kernel(
        feature_map=feature_map,
        kernel_type=cfg.kernel_type,
    )

    # 3. Compute or Load Cached Kernel Matrices
    K_train, K_val, K_test, cache_meta = compute_or_load_kernel_matrices(
        kernel=quantum_kernel,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        config=cfg,
    )

    # 4. Visualize Kernel Heatmap and Compare with RBF
    visualize_and_compare_kernels(
        K_quantum=K_train,
        X_train=X_train,
        save_path=cfg.heatmap_save_path,
    )

    # 5. Train & Tune Precomputed SVC
    svc_model, best_c, tuning_meta = train_and_tune_qsvc(
        K_train=K_train,
        y_train=y_train,
        K_val=K_val,
        y_val=y_val,
        config=cfg,
    )

    # 6. Evaluate Validation Metrics
    if K_val is not None:
        val_preds = svc_model.predict(K_val)
        val_scores = svc_model.predict_proba(K_val)[:, 1] if hasattr(svc_model, "predict_proba") else svc_model.decision_function(K_val)
        val_metrics = compute_qsvc_metrics(y_val, val_preds, val_scores)
    else:
        val_metrics = {}

    # 7. Evaluate Test Metrics
    test_preds = svc_model.predict(K_test)
    test_scores = svc_model.predict_proba(K_test)[:, 1] if hasattr(svc_model, "predict_proba") else svc_model.decision_function(K_test)
    test_metrics = compute_qsvc_metrics(y_test, test_preds, test_scores)

    total_pipeline_duration = round(time.time() - pipeline_start, 2)

    # 8. Save Fitted Model Bundle
    model_artifact = {
        "model": svc_model,
        "config": asdict(cfg),
        "best_C": best_c,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cache_metadata": cache_meta,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    cfg.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, cfg.model_save_path)
    logger.info("Saved fitted QSVC model artifact to: %s", cfg.model_save_path)

    # 9. Assemble and Export Master Results JSON
    results_payload = {
        "metadata": {
            "model_type": "QuantumSupportVectorClassifier",
            "framework": f"qiskit {qiskit.__version__} | qiskit-machine-learning {qiskit_machine_learning.__version__}",
            "dataset": cfg.dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_qubits": cfg.n_qubits,
            "feature_map": "ZZFeatureMap",
            "reps": cfg.reps,
            "entanglement": cfg.entanglement,
            "kernel_class": cfg.kernel_type,
            "optimal_C": best_c,
            "tuning_summary": tuning_meta,
            "cache_metadata": cache_meta,
            "pipeline_time_seconds": total_pipeline_duration,
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
            "test_samples": len(X_test),
        },
        "qiskit_api_changes": QISKIT_API_CHANGES_SUMMARY,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "saved_artifacts": {
            "metrics_json": str(cfg.metrics_save_path),
            "kernel_heatmap": str(cfg.heatmap_save_path),
            "model_bundle": str(cfg.model_save_path),
            "kernel_cache": str(cfg.cache_filename),
        },
    }

    cfg.metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Saved QSVC evaluation metrics to: %s", cfg.metrics_save_path)

    # Output formatted report
    print("\n" + "=" * 70)
    print("      QUANTUM SUPPORT VECTOR CLASSIFIER (QSVC) - TEST METRICS")
    print("=" * 70)
    print(f" Dataset:            {cfg.dataset_name.upper()}")
    print(f" Feature Map:        ZZFeatureMap (reps={cfg.reps}, entanglement={cfg.entanglement})")
    print(f" Qubits / Features:  {cfg.n_qubits}")
    print(f" Optimal SVC C:      {best_c}")
    print(f" Kernel Cache:       {cache_meta['cache_status']} ({cache_meta['cache_path']})")
    print(f" Pipeline Duration:  {total_pipeline_duration:.2f}s (~{total_pipeline_duration/60:.2f} mins)")
    print("-" * 70)
    print(f" Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f" Precision:          {test_metrics['precision']:.4f}")
    print(f" Recall:             {test_metrics['recall']:.4f}")
    print(f" Specificity:        {test_metrics['specificity']:.4f}")
    print(f" F1-Score:           {test_metrics['f1']:.4f}")
    print(f" ROC-AUC:            {test_metrics['roc_auc']:.4f}")
    print(f" Confusion Matrix:   {test_metrics['confusion_matrix']}")
    print("=" * 70 + "\n")

    return results_payload


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Quantum Support Vector Classifier (QSVC) with Qiskit ML.")
    parser.add_argument("--dataset", type=str, default="heart", choices=["heart", "diabetes"], help="Dataset name.")
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits (features).")
    parser.add_argument("--reps", type=int, default=2, help="Number of ZZ repetitions.")
    parser.add_argument("--entanglement", type=str, default="linear", choices=["linear", "full"], help="Entanglement strategy.")
    parser.add_argument("--c", type=float, default=1.0, help="Initial SVC C parameter.")
    parser.add_argument("--kernel-type", type=str, default="fidelity_quantum", choices=["fidelity_quantum", "fidelity_statevector"], help="Kernel backend.")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of kernel matrices ignoring disk cache.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")

    args = parser.parse_args()

    cfg = QSVCConfig(
        dataset_name=args.dataset,
        n_qubits=args.n_qubits,
        reps=args.reps,
        entanglement=args.entanglement,
        C=args.c,
        kernel_type=args.kernel_type,
        force_recompute=args.recompute,
        seed=args.seed,
    )

    run_qsvc_pipeline(config=cfg)


if __name__ == "__main__":
    main()
