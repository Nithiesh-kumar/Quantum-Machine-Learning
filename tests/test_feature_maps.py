"""Unit tests for Quantum Feature Maps module (src/quantum/feature_maps.py)."""

import numpy as np
import pennylane as qml
import pytest

from src.quantum.feature_maps import angle_feature_map, get_feature_map, zz_feature_map


def test_angle_feature_map_execution():
    """Verify AngleEmbedding runs without errors on default.qubit."""
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(x):
        angle_feature_map(x, wires=range(4), rotation="Y")
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    features = np.array([0.1, 0.5, 1.2, 2.0])
    res = circuit(features)
    assert len(res) == 4
    # With RY(theta) on |0>, expval(Z) = cos(theta)
    expected = np.cos(features)
    np.testing.assert_allclose(res, expected, atol=1e-5)


def test_zz_feature_map_topologies():
    """Verify ZZ feature map supports linear, circular, and full entanglement topologies."""
    dev = qml.device("default.qubit", wires=3)

    for topology in ["linear", "circular", "full"]:
        @qml.qnode(dev)
        def circuit(x):
            zz_feature_map(x, wires=range(3), reps=1, entanglement=topology)
            return qml.expval(qml.PauliZ(0))

        features = np.array([0.2, 0.4, 0.6])
        val = circuit(features)
        assert -1.0 <= val <= 1.0


def test_zz_feature_map_batched():
    """Verify ZZ feature map supports batched 2D feature inputs."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(x):
        zz_feature_map(x, wires=range(3), reps=1, entanglement="linear")
        return qml.expval(qml.PauliZ(0))

    features_batch = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    res = circuit(features_batch)
    assert len(res) == 2
    assert -1.0 <= res[0] <= 1.0
    assert -1.0 <= res[1] <= 1.0


def test_get_feature_map_factory():
    """Verify get_feature_map factory dispatches correct implementations."""
    fm_angle = get_feature_map("angle", rotation="Y")
    assert callable(fm_angle)

    fm_zz = get_feature_map("zz", reps=2, entanglement="linear")
    assert callable(fm_zz)

    with pytest.raises(ValueError, match="Unknown feature map"):
        get_feature_map("invalid_map_name")
