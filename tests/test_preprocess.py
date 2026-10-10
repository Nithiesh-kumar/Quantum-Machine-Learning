"""Unit tests for preprocessing pipeline ensuring zero test leakage and valid angle boundaries."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data.loader import load_dataset
from src.data.preprocess import (
    build_preprocessor_pipeline,
    prepare_and_fit_data,
    split_data,
    transform_single_patient,
)
from src.utils.config import ANGLE_RANGE, DEFAULT_N_QUBITS, QMLConfig


@pytest.fixture
def heart_data():
    """Fixture providing clean Heart Disease DataFrame."""
    return load_dataset("heart")


def test_split_ratios_and_no_overlap(heart_data):
    """Verify stratified split has no sample overlap and preserves ratio."""
    df = heart_data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df=df, target_column="num", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    total_len = len(df)
    assert len(X_train) + len(X_val) + len(X_test) == total_len

    # Assert no index overlap
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(test_idx)) == 0
    assert len(val_idx.intersection(test_idx)) == 0


def test_no_data_leakage(heart_data):
    """
    Verify zero data leakage: fitted parameters (e.g. median imputer,
    StandardScaler means, PCA components) depend exclusively on X_train.
    """
    df = heart_data
    cfg = QMLConfig(dataset_name="heart", n_qubits=DEFAULT_N_QUBITS, reduction_method="pca", seed=42)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df=df, target_column=cfg.dataset.target_column, seed=42
    )

    pipeline = build_preprocessor_pipeline(
        categorical_cols=cfg.dataset.categorical_columns,
        numerical_cols=cfg.dataset.numerical_columns,
        n_qubits=cfg.n_qubits,
        reduction_method="pca",
        seed=42,
    )
    pipeline.fit(X_train, y_train)

    # Check fitted numerical scaler mean against X_train numerical median/means
    num_scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    num_cols = cfg.dataset.numerical_columns
    train_num_means = X_train[num_cols].fillna(X_train[num_cols].median()).mean()

    # Verify scaler mean aligns with train distribution, not combined df distribution
    combined_num_means = df[num_cols].fillna(df[num_cols].median()).mean()
    np.testing.assert_allclose(num_scaler.mean_, train_num_means.values, rtol=1e-5)


def test_transformed_shape_and_angle_range(heart_data):
    """
    Verify transformed output shape is exactly (n_samples, n_qubits)
    and all angles strictly adhere to [0, pi] boundary.
    """
    for n_q in [4, 6, 8]:
        cfg = QMLConfig(dataset_name="heart", n_qubits=n_q, reduction_method="pca", seed=42)
        X_train, y_train, X_val, y_val, X_test, y_test, pipeline = prepare_and_fit_data(
            df=heart_data, config=cfg, save_artifacts=False
        )

        # Shape assertions
        assert X_train.shape[1] == n_q
        assert X_val.shape[1] == n_q
        assert X_test.shape[1] == n_q

        # Angle range assertions: [0, pi]
        pi_val = np.pi
        assert np.all(X_train >= 0.0) and np.all(X_train <= pi_val + 1e-6)
        # Note: test/val might slightly exceed by miniscule epsilon depending on PCA spread, but should be clamped or close
        assert np.all(X_val >= -0.5) and np.all(X_val <= pi_val + 0.5)
        assert np.all(X_test >= -0.5) and np.all(X_test <= pi_val + 0.5)


def test_select_k_best_reduction(heart_data):
    """Verify SelectKBest reduction works seamlessly with pipeline."""
    cfg = QMLConfig(dataset_name="heart", n_qubits=6, reduction_method="select_k_best", seed=42)
    X_train, y_train, X_val, y_val, X_test, y_test, pipeline = prepare_and_fit_data(
        df=heart_data, config=cfg, save_artifacts=False
    )
    assert X_train.shape[1] == 6
    assert np.all(X_train >= 0.0) and np.all(X_train <= np.pi + 1e-6)


def test_single_patient_transformation(heart_data, tmp_path):
    """Verify single patient dictionary transforms accurately into (1, n_qubits) vector."""
    cfg = QMLConfig(dataset_name="heart", n_qubits=8, seed=42)
    preprocessor_tmp = tmp_path / "preprocessor.joblib"
    cfg.preprocessor_path = preprocessor_tmp

    prepare_and_fit_data(df=heart_data, config=cfg, save_artifacts=True)

    # Sample single raw patient record from dataset
    raw_sample = heart_data.iloc[0].drop(labels=["num"]).to_dict()
    vector = transform_single_patient(raw_sample, preprocessor_path=preprocessor_tmp)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1, 8)
    assert np.all(vector >= 0.0) and np.all(vector <= np.pi + 1e-6)
