"""
Variational Quantum Classifier (VQC) with PennyLane.

Key Capabilities:
- Feature encoding options: AngleEmbedding (RY) and Havlicek-style ZZ entangling feature map (selectable via config).
- Ansatz options: StronglyEntanglingLayers or custom RealAmplitudes-style ansatz, configurable depth (1–4 layers).
- Measurement: Expectation of PauliZ on qubit 0 mapped to probability via (1 + <Z>)/2.
- Objective: Binary cross-entropy (BCE) loss.
- Training: Mini-batch Adam optimizer, early stopping on validation loss, PennyLane + PyTorch/Autograd interface
  with parameter-shift or backprop differentiation.
- Visualizations: Loss and accuracy curves per epoch saved to /results/vqc_curves.png; circuit diagram drawn via qml.draw_mpl.
- Model Persistence: Optimal model checkpoint saved to /models/vqc.pt; evaluation metrics saved to /results/vqc_metrics.json.
- Robustness: Configurable depolarizing noise channel on default.mixed device to study robustness under quantum noise.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Ensure UTF-8 console output across operating systems
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, ModuleNotFoundError):
    matplotlib = None
    plt = None
    HAS_MATPLOTLIB = False

import numpy as original_np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# PyTorch integration
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.loader import load_dataset
from src.data.preprocess import prepare_and_fit_data
from src.quantum.feature_maps import angle_feature_map, get_feature_map, zz_feature_map
from src.utils.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    SEED,
    QMLConfig,
    default_config,
)
from src.utils.logger import get_logger

logger = get_logger("vqc")


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------
@dataclass
class VQCConfig:
    """Configuration settings for Variational Quantum Classifier (VQC)."""
    dataset_name: str = "heart"
    n_qubits: int = 6
    n_layers: int = 2
    feature_map: str = "angle"          # 'angle' (AngleEmbedding RY) or 'zz' (ZZ-feature-map)
    ansatz: str = "strongly_entangling"   # 'strongly_entangling' or 'real_amplitudes'
    entanglement: str = "linear"         # 'linear', 'full', or 'circular'
    diff_method: str = "backprop"        # 'backprop' or 'parameter-shift'
    interface: str = "torch" if HAS_TORCH else "autograd"  # 'torch' or 'autograd'
    optimizer: str = "adam"
    learning_rate: float = 0.02
    batch_size: int = 16
    epochs: int = 40
    patience: int = 8
    min_delta: float = 1e-4
    noise_model: bool = False            # True to run on noisy simulator (default.mixed)
    noise_probability: float = 0.01      # Depolarizing channel error rate
    seed: int = SEED

    # Destination Artifact Paths
    model_save_path: Path = field(default_factory=lambda: MODELS_DIR / "vqc.pt")
    curves_save_path: Path = field(default_factory=lambda: RESULTS_DIR / "vqc_curves.png")
    circuit_save_path: Path = field(default_factory=lambda: RESULTS_DIR / "vqc_circuit.png")
    metrics_save_path: Path = field(default_factory=lambda: RESULTS_DIR / "vqc_metrics.json")

    def __post_init__(self):
        if not (1 <= self.n_layers <= 6):
            raise ValueError(f"n_layers ({self.n_layers}) must be between 1 and 6.")
        if not (2 <= self.n_qubits <= 8):
            raise ValueError(f"n_qubits ({self.n_qubits}) must be between 2 and 8.")
        if self.feature_map.lower() not in ("angle", "angle_ry", "zz", "zz_feature_map"):
            raise ValueError(f"Unsupported feature_map '{self.feature_map}'. Use 'angle' or 'zz'.")
        if self.ansatz.lower() not in ("strongly_entangling", "real_amplitudes"):
            raise ValueError(f"Unsupported ansatz '{self.ansatz}'. Use 'strongly_entangling' or 'real_amplitudes'.")
        if self.diff_method not in ("backprop", "parameter-shift"):
            raise ValueError(f"Unsupported diff_method '{self.diff_method}'. Use 'backprop' or 'parameter-shift'.")
        if self.noise_model and self.diff_method == "backprop":
            logger.info("Noisy simulator (default.mixed) requires parameter-shift. Adjusting diff_method to parameter-shift.")
            self.diff_method = "parameter-shift"
        if self.interface == "torch" and not HAS_TORCH:
            logger.warning("PyTorch requested but not installed. Falling back to PennyLane autograd interface.")
            self.interface = "autograd"

        self.model_save_path = Path(self.model_save_path)
        self.curves_save_path = Path(self.curves_save_path)
        self.circuit_save_path = Path(self.circuit_save_path)
        self.metrics_save_path = Path(self.metrics_save_path)


# ---------------------------------------------------------------------------
# Quantum Circuit Building Blocks
# ---------------------------------------------------------------------------
def strongly_entangling_ansatz(weights: Any, wires: Sequence[Union[int, str]]) -> None:
    """
    StronglyEntanglingLayers ansatz from PennyLane.
    Weights shape: (n_layers, n_qubits, 3).
    """
    qml.StronglyEntanglingLayers(weights, wires=wires)


def real_amplitudes_ansatz(
    weights: Any,
    wires: Sequence[Union[int, str]],
    entanglement: str = "linear",
) -> None:
    """
    RealAmplitudes-style ansatz consisting of alternating layers of RY rotations
    and two-qubit entangling CNOT gates.
    Weights shape: (n_layers + 1, n_qubits).
    """
    n_qubits = len(wires)
    n_weight_layers = weights.shape[0]

    # Initial RY rotation layer
    for i, w in enumerate(wires):
        qml.RY(weights[0, i], wires=w)

    # Entangling and rotational layers
    for l in range(n_weight_layers - 1):
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        elif entanglement == "circular":
            for i in range(n_qubits):
                qml.CNOT(wires=[wires[i], wires[(i + 1) % n_qubits]])
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[wires[i], wires[j]])
        else:
            raise ValueError(f"Unsupported entanglement '{entanglement}'.")

        # RY rotation layer after entanglement
        for i, w in enumerate(wires):
            qml.RY(weights[l + 1, i], wires=w)


def initialize_weights(
    ansatz_name: str,
    n_layers: int,
    n_qubits: int,
    seed: int = SEED,
) -> original_np.ndarray:
    """
    Initializes trainable parameters for the chosen quantum ansatz.
    """
    rng = original_np.random.default_rng(seed)
    clean_ansatz = ansatz_name.lower().strip()

    if clean_ansatz in ("strongly_entangling", "stronglyentanglinglayers"):
        shape = (n_layers, n_qubits, 3)
        weights = rng.uniform(0.0, 2.0 * original_np.pi, size=shape)
    elif clean_ansatz in ("real_amplitudes", "realamplitudes"):
        shape = (n_layers + 1, n_qubits)
        weights = rng.uniform(0.0, 2.0 * original_np.pi, size=shape)
    else:
        raise ValueError(f"Unknown ansatz '{ansatz_name}'.")

    return weights


# ---------------------------------------------------------------------------
# Variational Quantum Classifier Estimator
# ---------------------------------------------------------------------------
class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier (VQC) implemented with PennyLane.

    Compatible with scikit-learn estimator interface (fit, predict, predict_proba).
    """

    def __init__(self, config: Optional[VQCConfig] = None):
        self.config = config or VQCConfig()
        self.weights_: Optional[original_np.ndarray] = None
        self.history_: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_epoch_: int = 0
        self.best_val_loss_: float = float("inf")
        self.device_: Optional[qml.Device] = None
        self.qnode_: Optional[Callable] = None
        self._init_qnode()

    def _init_qnode(self) -> None:
        """Constructs quantum device and compiled QNode based on configuration."""
        cfg = self.config
        n_qubits = cfg.n_qubits
        wires = list(range(n_qubits))

        # Select Device
        if cfg.noise_model:
            logger.info("Initializing noisy quantum simulator: default.mixed (p=%.4f)", cfg.noise_probability)
            self.device_ = qml.device("default.mixed", wires=n_qubits)
            diff_method = "parameter-shift"
        else:
            self.device_ = qml.device("default.qubit", wires=n_qubits)
            diff_method = cfg.diff_method

        # Select Feature Map
        feature_map_fn = get_feature_map(
            name=cfg.feature_map,
            rotation="Y",
            reps=1,
            entanglement=cfg.entanglement,
        )

        # Select Ansatz
        clean_ansatz = cfg.ansatz.lower().strip()
        if clean_ansatz in ("strongly_entangling", "stronglyentanglinglayers"):
            def _ansatz_call(w, wires_list):
                strongly_entangling_ansatz(w, wires_list)
        else:
            def _ansatz_call(w, wires_list):
                real_amplitudes_ansatz(w, wires_list, entanglement=cfg.entanglement)

        noise_model = cfg.noise_model
        noise_p = cfg.noise_probability

        # Define quantum circuit
        def _circuit(features: Any, weights: Any):
            # 1. Feature Map Encoding
            feature_map_fn(features, wires)

            # Optional depolarizing noise channel after encoding
            if noise_model and noise_p > 0.0:
                for w in wires:
                    qml.DepolarizingChannel(noise_p, wires=w)

            # 2. Parameterized Variational Ansatz
            _ansatz_call(weights, wires)

            # Optional depolarizing noise channel after ansatz
            if noise_model and noise_p > 0.0:
                for w in wires:
                    qml.DepolarizingChannel(noise_p, wires=w)

            # 3. Measurement: Expectation of PauliZ on qubit 0
            return qml.expval(qml.PauliZ(wires[0]))

        # Compile QNode with selected differentiation method and interface
        self.qnode_ = qml.QNode(
            _circuit,
            self.device_,
            diff_method=diff_method,
            interface=cfg.interface,
        )

    def draw_circuit(self, sample_x: Optional[original_np.ndarray] = None, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Draws the quantum circuit with qml.draw_mpl and saves it to disk for reports.

        Args:
            sample_x: Feature vector for drawing. Defaults to zeros.
            save_path: Destination image path. Defaults to config.circuit_save_path.
        """
        cfg = self.config
        target_path = Path(save_path or cfg.circuit_save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if sample_x is None:
            sample_x = original_np.zeros(cfg.n_qubits)
        else:
            sample_x = original_np.asarray(sample_x, dtype=float)[:cfg.n_qubits]

        weights = self.weights_ if self.weights_ is not None else initialize_weights(
            cfg.ansatz, cfg.n_layers, cfg.n_qubits, seed=cfg.seed
        )

        # Draw circuit using PennyLane draw_mpl
        fig, ax = qml.draw_mpl(self.qnode_, style="pennylane")(sample_x, weights)
        ax.set_title(
            f"VQC Circuit Architecture ({cfg.n_qubits} Qubits, {cfg.n_layers} Layers)\n"
            f"Encoding: {cfg.feature_map.upper()} | Ansatz: {cfg.ansatz.title()} | Noise: {cfg.noise_model}",
            fontsize=11,
            pad=15,
        )
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Quantum circuit diagram saved to: %s", target_path)
        return fig

    def _train_torch(
        self,
        X_train: original_np.ndarray,
        y_train: original_np.ndarray,
        X_val: Optional[original_np.ndarray] = None,
        y_val: Optional[original_np.ndarray] = None,
    ) -> None:
        """Executes mini-batch Adam training using the PennyLane + PyTorch interface."""
        cfg = self.config
        n_samples = len(X_train)
        device_dtype = torch.float64

        # Initialize weights as PyTorch parameter
        init_w = initialize_weights(cfg.ansatz, cfg.n_layers, cfg.n_qubits, seed=cfg.seed)
        weights_tensor = nn.Parameter(torch.tensor(init_w, dtype=device_dtype, requires_grad=True))

        optimizer = optim.Adam([weights_tensor], lr=cfg.learning_rate)

        # Convert full datasets to Torch tensors
        X_train_t = torch.tensor(X_train, dtype=device_dtype)
        y_train_t = torch.tensor(y_train, dtype=device_dtype)
        X_val_t = torch.tensor(X_val, dtype=device_dtype) if X_val is not None else None
        y_val_t = torch.tensor(y_val, dtype=device_dtype) if y_val is not None else None

        best_weights = weights_tensor.detach().clone()
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        self.history_ = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        start_time = time.time()
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()

            # Mini-batch shuffling
            permutation = torch.randperm(n_samples)

            for b_start in range(0, n_samples, cfg.batch_size):
                b_indices = permutation[b_start : min(b_start + cfg.batch_size, n_samples)]
                X_batch = X_train_t[b_indices]
                y_batch = y_train_t[b_indices]

                optimizer.zero_grad()
                expvals = self.qnode_(X_batch, weights_tensor)
                # Map <Z> -> Probability (1 + <Z>) / 2
                probs = (1.0 + expvals) / 2.0
                probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)

                loss = nn.functional.binary_cross_entropy(probs, y_batch)
                loss.backward()
                optimizer.step()

            # Train evaluation
            with torch.no_grad():
                train_expvals = self.qnode_(X_train_t, weights_tensor)
                train_probs = torch.clamp((1.0 + train_expvals) / 2.0, 1e-7, 1.0 - 1e-7)
                train_loss = float(nn.functional.binary_cross_entropy(train_probs, y_train_t).item())
                train_preds = (train_probs >= 0.5).long().cpu().numpy()
                train_acc = float(accuracy_score(y_train, train_preds))

            self.history_["train_loss"].append(round(train_loss, 4))
            self.history_["train_acc"].append(round(train_acc, 4))

            # Validation evaluation
            if X_val_t is not None and y_val_t is not None:
                with torch.no_grad():
                    val_expvals = self.qnode_(X_val_t, weights_tensor)
                    val_probs = torch.clamp((1.0 + val_expvals) / 2.0, 1e-7, 1.0 - 1e-7)
                    val_loss = float(nn.functional.binary_cross_entropy(val_probs, y_val_t).item())
                    val_preds = (val_probs >= 0.5).long().cpu().numpy()
                    val_acc = float(accuracy_score(y_val, val_preds))

                self.history_["val_loss"].append(round(val_loss, 4))
                self.history_["val_acc"].append(round(val_acc, 4))

                epoch_dur = time.time() - epoch_start
                logger.info(
                    "Epoch %02d/%02d [%.1fs] | Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
                    epoch, cfg.epochs, epoch_dur, train_loss, train_acc, val_loss, val_acc
                )

                # Early stopping tracking
                if val_loss < (best_val_loss - cfg.min_delta):
                    best_val_loss = val_loss
                    best_weights = weights_tensor.detach().clone()
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        logger.info(
                            "Early stopping triggered at epoch %d (best val_loss=%.4f at epoch %d).",
                            epoch, best_val_loss, best_epoch
                        )
                        break
            else:
                epoch_dur = time.time() - epoch_start
                logger.info(
                    "Epoch %02d/%02d [%.1fs] | Train Loss: %.4f, Train Acc: %.4f",
                    epoch, cfg.epochs, epoch_dur, train_loss, train_acc
                )
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_weights = weights_tensor.detach().clone()
                    best_epoch = epoch

        total_training_time = time.time() - start_time
        logger.info(
            "VQC Training completed in %.2f seconds (~%.2f mins). Best Epoch: %d",
            total_training_time, total_training_time / 60.0, best_epoch
        )

        self.weights_ = best_weights.cpu().numpy()
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = best_val_loss

    def _train_autograd(
        self,
        X_train: original_np.ndarray,
        y_train: original_np.ndarray,
        X_val: Optional[original_np.ndarray] = None,
        y_val: Optional[original_np.ndarray] = None,
    ) -> None:
        """Executes mini-batch Adam training using the PennyLane + Autograd interface."""
        cfg = self.config
        n_samples = len(X_train)

        init_w = initialize_weights(cfg.ansatz, cfg.n_layers, cfg.n_qubits, seed=cfg.seed)
        weights = pnp.array(init_w, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=cfg.learning_rate)

        best_weights = pnp.copy(weights)
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        self.history_ = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        start_time = time.time()
        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()

            indices = original_np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for b_start in range(0, n_samples, cfg.batch_size):
                b_end = min(b_start + cfg.batch_size, n_samples)
                X_batch = X_shuffled[b_start:b_end]
                y_batch = [float(val) for val in y_shuffled[b_start:b_end]]

                def cost_fn(w):
                    expvals = self.qnode_(X_batch, w)
                    probs = (1.0 + expvals) / 2.0
                    loss = 0.0
                    for i, y_scalar in enumerate(y_batch):
                        p_i = pnp.clip(probs[i], 1e-7, 1.0 - 1e-7)
                        loss = loss - (y_scalar * pnp.log(p_i) + (1.0 - y_scalar) * pnp.log(1.0 - p_i))
                    return loss / len(y_batch)

                weights = opt.step(cost_fn, weights)

            # Evaluate epoch
            train_expvals = self.qnode_(X_train, weights)
            train_probs = original_np.clip((1.0 + original_np.asarray(train_expvals)) / 2.0, 1e-7, 1.0 - 1e-7)
            train_loss = float(-original_np.mean(y_train * original_np.log(train_probs) + (1.0 - y_train) * original_np.log(1.0 - train_probs)))
            train_acc = float(accuracy_score(y_train, (train_probs >= 0.5).astype(int)))

            self.history_["train_loss"].append(round(train_loss, 4))
            self.history_["train_acc"].append(round(train_acc, 4))

            if X_val is not None and y_val is not None:
                val_expvals = self.qnode_(X_val, weights)
                val_probs = original_np.clip((1.0 + original_np.asarray(val_expvals)) / 2.0, 1e-7, 1.0 - 1e-7)
                val_loss = float(-original_np.mean(y_val * original_np.log(val_probs) + (1.0 - y_val) * original_np.log(1.0 - val_probs)))
                val_acc = float(accuracy_score(y_val, (val_probs >= 0.5).astype(int)))

                self.history_["val_loss"].append(round(val_loss, 4))
                self.history_["val_acc"].append(round(val_acc, 4))

                epoch_dur = time.time() - epoch_start
                logger.info(
                    "Epoch %02d/%02d [%.1fs] | Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
                    epoch, cfg.epochs, epoch_dur, train_loss, train_acc, val_loss, val_acc
                )

                if val_loss < (best_val_loss - cfg.min_delta):
                    best_val_loss = val_loss
                    best_weights = pnp.copy(weights)
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.patience:
                        logger.info("Early stopping triggered at epoch %d.", epoch)
                        break
            else:
                epoch_dur = time.time() - epoch_start
                logger.info("Epoch %02d/%02d [%.1fs] | Train Loss: %.4f, Train Acc: %.4f", epoch, cfg.epochs, epoch_dur, train_loss, train_acc)
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_weights = pnp.copy(weights)
                    best_epoch = epoch

        total_training_time = time.time() - start_time
        logger.info("VQC Training completed in %.2f seconds. Best Epoch: %d", total_training_time, best_epoch)

        self.weights_ = original_np.array(best_weights, copy=True)
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = best_val_loss

    def fit(
        self,
        X_train: original_np.ndarray,
        y_train: original_np.ndarray,
        X_val: Optional[original_np.ndarray] = None,
        y_val: Optional[original_np.ndarray] = None,
    ) -> "VariationalQuantumClassifier":
        """
        Trains the VQC using mini-batch Adam optimization and early stopping.

        Args:
            X_train: Training features of shape (N, n_qubits).
            y_train: Binary labels {0, 1} of shape (N,).
            X_val: Optional validation features.
            y_val: Optional validation binary labels.

        Returns:
            Self fitted estimator.
        """
        cfg = self.config
        n_samples = len(X_train)
        logger.info(
            "Fitting VQC model | Samples: %d, Qubits: %d, Layers: %d, Batch: %d, LR: %.4f, Diff: %s, Interface: %s",
            n_samples, cfg.n_qubits, cfg.n_layers, cfg.batch_size, cfg.learning_rate, cfg.diff_method, cfg.interface
        )

        # Draw circuit diagram prior to training
        self.draw_circuit(sample_x=X_train[0])

        if cfg.interface == "torch" and HAS_TORCH:
            self._train_torch(X_train, y_train, X_val, y_val)
        else:
            self._train_autograd(X_train, y_train, X_val, y_val)

        # Save training artifacts
        self.plot_training_curves()
        self.save_model()

        return self

    def predict_proba(self, X: original_np.ndarray) -> original_np.ndarray:
        """
        Predicts class probabilities for binary classification [P(y=0), P(y=1)].
        """
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() before predicting.")

        X = original_np.asarray(X, dtype=float)
        cfg = self.config

        if cfg.interface == "torch" and HAS_TORCH:
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float64)
                w_t = torch.tensor(self.weights_, dtype=torch.float64)
                expvals = self.qnode_(X_t, w_t)
                probs1 = ((1.0 + expvals) / 2.0).clamp(0.0, 1.0).cpu().numpy()
        else:
            expvals = self.qnode_(X, self.weights_)
            probs1 = original_np.clip((1.0 + original_np.asarray(expvals)) / 2.0, 0.0, 1.0)

        probs1 = original_np.asarray(probs1, dtype=float).ravel()
        probs0 = 1.0 - probs1
        return original_np.column_stack([probs0, probs1])

    def predict(self, X: original_np.ndarray) -> original_np.ndarray:
        """Predicts binary labels {0, 1} with 0.5 decision threshold."""
        prob1 = self.predict_proba(X)[:, 1]
        return (prob1 >= 0.5).astype(int)

    def plot_training_curves(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Generates and saves publication-grade loss and accuracy curves across epochs.
        """
        cfg = self.config
        target_path = Path(save_path or cfg.curves_save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        epochs_range = range(1, len(self.history_["train_loss"]) + 1)

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        fig.suptitle(
            f"VQC Optimization Curves ({cfg.dataset_name.upper()} Dataset | {cfg.n_qubits} Qubits, {cfg.n_layers} Layers)\n"
            f"Feature Map: {cfg.feature_map.upper()} | Ansatz: {cfg.ansatz.title()} | Noise: {cfg.noise_model}",
            fontsize=13,
            fontweight="bold",
            y=1.03,
        )

        # Loss Subplot
        ax_loss.plot(epochs_range, self.history_["train_loss"], label="Train BCE Loss", color="#1f77b4", linewidth=2)
        if self.history_["val_loss"]:
            ax_loss.plot(epochs_range, self.history_["val_loss"], label="Val BCE Loss", color="#ff7f0e", linewidth=2, linestyle="--")
            if self.best_epoch_ > 0 and self.best_epoch_ <= len(self.history_["val_loss"]):
                ax_loss.axvline(self.best_epoch_, color="green", linestyle=":", alpha=0.8, label=f"Best Checkpoint (Epoch {self.best_epoch_})")
        ax_loss.set_title("Binary Cross-Entropy Loss vs. Epoch", fontsize=11)
        ax_loss.set_xlabel("Epoch", fontsize=10)
        ax_loss.set_ylabel("BCE Loss", fontsize=10)
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        ax_loss.legend(loc="upper right", frameon=True)

        # Accuracy Subplot
        ax_acc.plot(epochs_range, self.history_["train_acc"], label="Train Accuracy", color="#2ca02c", linewidth=2)
        if self.history_["val_acc"]:
            ax_acc.plot(epochs_range, self.history_["val_acc"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
            if self.best_epoch_ > 0 and self.best_epoch_ <= len(self.history_["val_acc"]):
                ax_acc.axvline(self.best_epoch_, color="green", linestyle=":", alpha=0.8, label=f"Best Checkpoint (Epoch {self.best_epoch_})")
        ax_acc.set_title("Classification Accuracy vs. Epoch", fontsize=11)
        ax_acc.set_xlabel("Epoch", fontsize=10)
        ax_acc.set_ylabel("Accuracy", fontsize=10)
        ax_acc.set_ylim(-0.02, 1.02)
        ax_acc.grid(True, linestyle="--", alpha=0.5)
        ax_acc.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved to: %s", target_path)
        return fig

    def save_model(self, save_path: Optional[Path] = None) -> None:
        """
        Saves the fitted VQC weights, architecture metadata, and configuration to disk.
        """
        cfg = self.config
        target_path = Path(save_path or cfg.model_save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "weights": self.weights_,
            "config": asdict(cfg),
            "best_epoch": self.best_epoch_,
            "best_val_loss": self.best_val_loss_,
            "history": self.history_,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        if HAS_TORCH:
            try:
                torch.save(checkpoint, target_path)
                logger.info("Saved VQC checkpoint via torch.save to: %s", target_path)
                return
            except Exception as e:
                logger.warning("torch.save failed (%s). Falling back to pickle.", e)

        with open(target_path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved VQC checkpoint to: %s", target_path)

    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> "VariationalQuantumClassifier":
        """Loads a saved VQC model checkpoint from disk."""
        target_path = Path(model_path)
        # Check for alternative extensions if file does not exist directly
        if not target_path.exists():
            for alt_ext in [".joblib", ".pt", ".pkl"]:
                candidate = target_path.with_suffix(alt_ext)
                if candidate.exists():
                    target_path = candidate
                    break
        if not target_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {target_path}")

        loaded_data = None
        # 1. If joblib format or joblib checkpoint exists alongside .pt
        joblib_path = target_path if target_path.suffix == ".joblib" else target_path.with_suffix(".joblib")
        if joblib_path.exists():
            try:
                import joblib
                loaded_data = joblib.load(joblib_path)
            except Exception as e:
                logger.warning("joblib.load on %s failed: %s", joblib_path, e)

        # 2. PyTorch checkpoint loading if torch is available
        if loaded_data is None and HAS_TORCH:
            try:
                loaded_data = torch.load(target_path, map_location="cpu", weights_only=False)
            except TypeError:
                try:
                    loaded_data = torch.load(target_path, map_location="cpu")
                except Exception as e:
                    logger.warning("torch.load fallback failed: %s", e)
            except Exception as e:
                logger.warning("torch.load failed (%s). Attempting pickle fallback.", e)

        # 3. Standard pickle fallback
        if loaded_data is None:
            with open(target_path, "rb") as f:
                loaded_data = pickle.load(f)

        cfg_dict = loaded_data["config"]
        cfg_dict["model_save_path"] = Path(cfg_dict.get("model_save_path", MODELS_DIR / "vqc.pt"))
        cfg_dict["curves_save_path"] = Path(cfg_dict.get("curves_save_path", RESULTS_DIR / "vqc_curves.png"))
        cfg_dict["circuit_save_path"] = Path(cfg_dict.get("circuit_save_path", RESULTS_DIR / "vqc_circuit.png"))
        cfg_dict["metrics_save_path"] = Path(cfg_dict.get("metrics_save_path", RESULTS_DIR / "vqc_metrics.json"))

        config = VQCConfig(**cfg_dict)
        vqc = cls(config=config)
        vqc.weights_ = original_np.array(loaded_data["weights"], copy=True)
        vqc.best_epoch_ = loaded_data.get("best_epoch", 0)
        vqc.best_val_loss_ = loaded_data.get("best_val_loss", float("inf"))
        vqc.history_ = loaded_data.get("history", {})
        return vqc


# ---------------------------------------------------------------------------
# Metrics Evaluation and Benchmark Routine
# ---------------------------------------------------------------------------
def compute_vqc_metrics(
    y_true: original_np.ndarray,
    y_pred: original_np.ndarray,
    y_score: original_np.ndarray,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification clinical metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = 0.5
    try:
        if len(original_np.unique(y_true)) > 1:
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


def run_vqc_pipeline(
    config: Optional[VQCConfig] = None,
    dataset_name: str = "heart",
) -> Dict[str, Any]:
    """
    Executes end-to-end VQC training, validation, and test evaluation pipeline.
    """
    cfg = config or VQCConfig(dataset_name=dataset_name)

    logger.info("=" * 70)
    logger.info("VARIATIONAL QUANTUM CLASSIFIER (VQC) TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info("Dataset: %s | Qubits: %d | Layers: %d", cfg.dataset_name, cfg.n_qubits, cfg.n_layers)
    logger.info("Encoding: %s | Ansatz: %s | Interface: %s | Noise: %s", cfg.feature_map, cfg.ansatz, cfg.interface, cfg.noise_model)

    # 1. Load Data
    processed_npz = PROCESSED_DATA_DIR / f"{cfg.dataset_name}_processed.npz"
    if processed_npz.exists():
        logger.info("Loading preprocessed dataset from: %s", processed_npz)
        data = original_np.load(processed_npz)
        X_train = data["X_train"][:, :cfg.n_qubits]
        y_train = data["y_train"]
        X_val = data["X_val"][:, :cfg.n_qubits]
        y_val = data["y_val"]
        X_test = data["X_test"][:, :cfg.n_qubits]
        y_test = data["y_test"]
    else:
        logger.info("Processed data not found. Loading and preparing raw '%s' data...", cfg.dataset_name)
        df_raw = load_dataset(cfg.dataset_name)
        qml_cfg = QMLConfig(dataset_name=cfg.dataset_name, n_qubits=cfg.n_qubits, seed=cfg.seed)
        X_tr, y_tr, X_v, y_v, X_te, y_te, _ = prepare_and_fit_data(df_raw, config=qml_cfg)
        X_train, y_train, X_val, y_val, X_test, y_test = X_tr, y_tr, X_v, y_v, X_te, y_te

    # 2. Instantiate and Fit VQC
    start_train_wall = time.time()
    vqc = VariationalQuantumClassifier(config=cfg)
    vqc.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    train_duration = round(time.time() - start_train_wall, 2)

    # 3. Test Set Evaluation
    test_scores = vqc.predict_proba(X_test)[:, 1]
    test_preds = vqc.predict(X_test)
    test_metrics = compute_vqc_metrics(y_test, test_preds, test_scores)

    val_scores = vqc.predict_proba(X_val)[:, 1]
    val_preds = vqc.predict(X_val)
    val_metrics = compute_vqc_metrics(y_val, val_preds, val_scores)

    # 4. Assemble Results JSON
    results_payload = {
        "metadata": {
            "model_type": "VariationalQuantumClassifier",
            "framework": f"PennyLane {qml.__version__}",
            "dataset": cfg.dataset_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_qubits": cfg.n_qubits,
            "n_layers": cfg.n_layers,
            "feature_map": cfg.feature_map,
            "ansatz": cfg.ansatz,
            "measurement": "PauliZ_q0",
            "loss_function": "binary_cross_entropy",
            "optimizer": "adam",
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "epochs_configured": cfg.epochs,
            "best_epoch": vqc.best_epoch_,
            "early_stopping_patience": cfg.patience,
            "diff_method": cfg.diff_method,
            "interface": cfg.interface,
            "noise_model": cfg.noise_model,
            "noise_probability": cfg.noise_probability if cfg.noise_model else 0.0,
            "train_time_seconds": train_duration,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "saved_artifacts": {
            "model_checkpoint": str(cfg.model_save_path),
            "training_curves": str(cfg.curves_save_path),
            "circuit_diagram": str(cfg.circuit_save_path),
            "metrics_json": str(cfg.metrics_save_path),
        },
    }

    # Save to /results/vqc_metrics.json
    cfg.metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info("Test metrics exported to: %s", cfg.metrics_save_path)

    # Output formatted report
    print("\n" + "=" * 70)
    print("           VARIATIONAL QUANTUM CLASSIFIER - FINAL TEST METRICS")
    print("=" * 70)
    print(f" Dataset:            {cfg.dataset_name.upper()}")
    print(f" Circuit Topology:   {cfg.n_qubits} Qubits | {cfg.n_layers} Layers")
    print(f" Feature Map:        {cfg.feature_map.upper()} | Ansatz: {cfg.ansatz.title()}")
    print(f" Interface:          {cfg.interface.upper()} | Diff Method: {cfg.diff_method}")
    print(f" Noise Model:        {cfg.noise_model} (p={cfg.noise_probability if cfg.noise_model else 0.0})")
    print(f" Training Wall Time: {train_duration:.2f}s (~{train_duration/60:.2f} mins)")
    print(f" Best Epoch:         {vqc.best_epoch_}")
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
    parser = argparse.ArgumentParser(description="Variational Quantum Classifier (VQC) with PennyLane.")
    parser.add_argument("--dataset", type=str, default="heart", choices=["heart", "diabetes"], help="Dataset name.")
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits (4 to 6 recommended).")
    parser.add_argument("--n-layers", type=int, default=2, help="Depth of ansatz layers (1 to 4).")
    parser.add_argument("--feature-map", type=str, default="angle", choices=["angle", "zz"], help="Feature map type.")
    parser.add_argument("--ansatz", type=str, default="strongly_entangling", choices=["strongly_entangling", "real_amplitudes"], help="Ansatz type.")
    parser.add_argument("--diff-method", type=str, default="backprop", choices=["backprop", "parameter-shift"], help="Differentiation method.")
    parser.add_argument("--interface", type=str, default="torch" if HAS_TORCH else "autograd", choices=["torch", "autograd"], help="Interface framework.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.02, help="Adam learning rate.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--noise", action="store_true", help="Enable depolarizing noise on default.mixed device.")
    parser.add_argument("--noise-prob", type=float, default=0.01, help="Depolarizing noise probability.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")

    args = parser.parse_args()

    cfg = VQCConfig(
        dataset_name=args.dataset,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        feature_map=args.feature_map,
        ansatz=args.ansatz,
        diff_method=args.diff_method,
        interface=args.interface,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        noise_model=args.noise,
        noise_probability=args.noise_prob,
        seed=args.seed,
    )

    run_vqc_pipeline(config=cfg)


if __name__ == "__main__":
    main()
