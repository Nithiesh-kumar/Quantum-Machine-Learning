"""
Quantum Feature Maps for Variational Quantum Classifiers (VQC).

Implements configurable quantum state encoding routines including:
1. AngleEmbedding with RY rotation.
2. Havlicek-style ZZ entangling feature map with configurable entanglement
   topologies (linear, full, circular) and repetitions.
"""

from typing import Callable, List, Optional, Sequence, Union, Any
import numpy as np
import pennylane as qml


def angle_feature_map(
    features: Any,
    wires: Sequence[Union[int, str]],
    rotation: str = "Y",
) -> None:
    """
    Encodes classical features into qubit states via single-qubit rotations.

    Args:
        features: 1D or 2D array/tensor of features with last dim <= len(wires).
        wires: Sequence of wire identifiers to encode onto.
        rotation: Axis of rotation: 'X', 'Y', or 'Z'. Defaults to 'Y'.
    """
    # Check if features is torch Tensor or numpy array
    if hasattr(features, "ndim") and features.ndim == 2:
        num_features = features.shape[-1]
    elif hasattr(features, "shape") and len(features.shape) == 2:
        num_features = features.shape[-1]
    else:
        num_features = len(features)

    target_wires = wires[:num_features]
    qml.AngleEmbedding(features=features, wires=target_wires, rotation=rotation)


def zz_feature_map(
    features: Any,
    wires: Sequence[Union[int, str]],
    reps: int = 1,
    entanglement: str = "linear",
    data_scaling: float = 2.0,
) -> None:
    """
    ZZ-style entangling feature map inspired by Havlicek et al. (Nature 2019).

    For each repetition:
      1. Applies Hadamard gates to initialize a uniform superposition.
      2. Applies RZ single-qubit rotations with phase proportional to features.
      3. Applies pairwise ZZ interactions (via CNOT - RZ - CNOT) parameterized
         by non-linear product terms: data_scaling * (pi - x_i) * (pi - x_j).

    Args:
        features: 1D classical feature array/tensor.
        wires: Sequence of wires to act on.
        reps: Number of repetitions/layers for the feature map.
        entanglement: Entanglement topology: 'linear', 'full', or 'circular'.
        data_scaling: Scaling factor for the non-linear phase terms.
    """
    features_arr = np.asarray(features)
    is_batched = (features_arr.ndim == 2)
    n_features = features_arr.shape[-1] if is_batched else len(features_arr)
    n_wires = len(wires)
    active_len = min(n_features, n_wires)
    active_wires = list(wires[:active_len])

    # Determine pairwise interactions based on requested topology
    pairs: List[tuple[int, int]] = []
    if entanglement == "linear":
        pairs = [(i, i + 1) for i in range(active_len - 1)]
    elif entanglement == "full":
        pairs = [(i, j) for i in range(active_len) for j in range(i + 1, active_len)]
    elif entanglement == "circular":
        pairs = [(i, (i + 1) % active_len) for i in range(active_len)]
    else:
        raise ValueError(
            f"Unsupported entanglement '{entanglement}'. Choose 'linear', 'full', or 'circular'."
        )

    for _ in range(reps):
        # 1. Hadamard on each active wire
        for w in active_wires:
            qml.Hadamard(wires=w)

        # 2. Single-qubit RZ rotations
        for i, w in enumerate(active_wires):
            feat_i = features_arr[:, i] if is_batched else features_arr[i]
            qml.RZ(data_scaling * feat_i, wires=w)

        # 3. Pairwise ZZ interactions: CNOT -> RZ -> CNOT
        for i, j in pairs:
            w_i = active_wires[i]
            w_j = active_wires[j]
            feat_i = features_arr[:, i] if is_batched else features_arr[i]
            feat_j = features_arr[:, j] if is_batched else features_arr[j]
            # Havlicek non-linear kernel phase term
            phi = data_scaling * (np.pi - feat_i) * (np.pi - feat_j)
            qml.CNOT(wires=[w_i, w_j])
            qml.RZ(phi, wires=w_j)
            qml.CNOT(wires=[w_i, w_j])


def get_feature_map(
    name: str = "angle",
    rotation: str = "Y",
    reps: int = 1,
    entanglement: str = "linear",
    data_scaling: float = 2.0,
) -> Callable[[Any, Sequence[Union[int, str]]], None]:
    """
    Factory function returning a callable feature map function for PennyLane QNodes.

    Args:
        name: Feature map identifier ('angle', 'angle_ry', 'zz', 'zz_feature_map').
        rotation: Rotation axis for angle encoding ('Y', 'X', 'Z').
        reps: Number of repetitions for ZZ feature map.
        entanglement: Topology for ZZ feature map ('linear', 'full', 'circular').
        data_scaling: Scaling factor for non-linear phase.

    Returns:
        Callable taking (features, wires) to embed data into a quantum circuit.
    """
    clean_name = name.lower().strip()
    if clean_name in ("angle", "angle_ry", "angleembedding"):
        def _angle_fm(features: Any, wires: Sequence[Union[int, str]]) -> None:
            angle_feature_map(features=features, wires=wires, rotation=rotation)
        return _angle_fm

    elif clean_name in ("zz", "zz_feature_map", "zzfeaturemap"):
        def _zz_fm(features: Any, wires: Sequence[Union[int, str]]) -> None:
            zz_feature_map(
                features=features,
                wires=wires,
                reps=reps,
                entanglement=entanglement,
                data_scaling=data_scaling,
            )
        return _zz_fm

    else:
        raise ValueError(
            f"Unknown feature map '{name}'. Available: 'angle' (AngleEmbedding RY) or 'zz' (ZZ-style)."
        )
