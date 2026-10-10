"""
Preprocessing pipeline for Quantum ML Health Risk System.
Ensures zero test leakage, performs imputation, scaling, feature reduction
to n_qubits features, and scales to [0, pi] for angle encoding.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.utils.config import (
    ANGLE_RANGE,
    DATASET_REGISTRY,
    DEFAULT_N_QUBITS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    QMLConfig,
    default_config,
)
from src.utils.logger import get_logger

logger = get_logger("preprocess")


def build_preprocessor_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
    n_qubits: int = DEFAULT_N_QUBITS,
    reduction_method: str = "pca",
    seed: int = SEED,
) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn Pipeline with zero test leakage.

    Steps:
      1. ColumnTransformer:
         - Numerical: Median Imputer -> StandardScaler
         - Categorical: Mode Imputer -> OneHotEncoder
      2. Feature Reduction: PCA or SelectKBest down to n_qubits
      3. Angle Scaler: MinMaxScaler scaling strictly to [0, pi]
    """
    transformers = []

    # Numerical transformer
    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    # Categorical transformer
    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    preprocessor_union = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # Reduction Step
    if reduction_method == "pca":
        reducer = PCA(n_components=n_qubits, random_state=seed)
    elif reduction_method == "select_k_best":
        reducer = SelectKBest(score_func=f_classif, k=n_qubits)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}. Choose 'pca' or 'select_k_best'.")

    # MinMax scaling strictly into [0, pi] for quantum angle encoding
    angle_scaler = MinMaxScaler(feature_range=ANGLE_RANGE)

    pipeline = Pipeline([
        ("preprocessor", preprocessor_union),
        ("reducer", reducer),
        ("angle_scaler", angle_scaler),
    ])

    return pipeline


def split_data(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits data into stratified train, val, and test partitions (default 70/15/15).
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=seed, stratify=y
    )

    # Second split: val vs test (equal 50/50 split of the remaining 30%)
    relative_test_ratio = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=seed, stratify=y_temp
    )

    logger.info(
        "Data split: Train=%d (%0.1f%%), Val=%d (%0.1f%%), Test=%d (%0.1f%%)",
        len(X_train), (len(X_train) / len(df)) * 100,
        len(X_val), (len(X_val) / len(df)) * 100,
        len(X_test), (len(X_test) / len(df)) * 100,
    )
    return X_train, X_val, X_test, y_train, y_test_val_helper(y_val), y_test


def y_test_val_helper(y_val: pd.Series) -> pd.Series:
    """Helper to return y_val cleanly."""
    return y_val


def prepare_and_fit_data(
    df: pd.DataFrame,
    config: Optional[QMLConfig] = None,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Fits the preprocessing pipeline exclusively on training data and transforms
    train, val, and test sets.
    """
    cfg = config or default_config
    meta = cfg.dataset

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df=df,
        target_column=meta.target_column,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    pipeline = build_preprocessor_pipeline(
        categorical_cols=meta.categorical_columns,
        numerical_cols=meta.numerical_columns,
        n_qubits=cfg.n_qubits,
        reduction_method=cfg.reduction_method,
        seed=cfg.seed,
    )

    logger.info("Fitting pipeline strictly on training data (shape: %s)...", X_train.shape)
    # Fit strictly on train
    X_train_trans = pipeline.fit_transform(X_train, y_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # Convert y to numpy arrays
    y_train_arr = y_train.to_numpy(dtype=int)
    y_val_arr = y_val.to_numpy(dtype=int)
    y_test_arr = y_test.to_numpy(dtype=int)

    logger.info(
        "Transformed feature shapes: Train=%s, Val=%s, Test=%s",
        X_train_trans.shape, X_val_trans.shape, X_test_trans.shape
    )
    logger.info(
        "Feature value ranges - Train: [%.4f, %.4f], Val: [%.4f, %.4f], Test: [%.4f, %.4f]",
        X_train_trans.min(), X_train_trans.max(),
        X_val_trans.min(), X_val_trans.max(),
        X_test_trans.min(), X_test_trans.max(),
    )

    if save_artifacts:
        # Save fitted pipeline to both .pkl and .joblib
        joblib.dump(pipeline, cfg.preprocessor_path)
        joblib.dump(pipeline, cfg.preprocessor_joblib_path)
        logger.info("Fitted preprocessor pipeline saved to: %s and %s", cfg.preprocessor_path, cfg.preprocessor_joblib_path)

        # Save processed splits
        processed_file = PROCESSED_DATA_DIR / f"{meta.name}_processed.npz"
        np.savez_compressed(
            processed_file,
            X_train=X_train_trans,
            y_train=y_train_arr,
            X_val=X_val_trans,
            y_val=y_val_arr,
            X_test=X_test_trans,
            y_test=y_test_arr,
        )
        logger.info("Processed data arrays saved to: %s", processed_file)

    return (
        X_train_trans,
        y_train_arr,
        X_val_trans,
        y_val_arr,
        X_test_trans,
        y_test_arr,
        pipeline,
    )


def transform_single_patient(
    patient_dict: Dict[str, Any],
    preprocessor_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Transforms a single raw patient dictionary into a model-ready quantum feature vector.

    Args:
        patient_dict: Dictionary with clinical feature names matching the dataset schema.
        preprocessor_path: Path to serialized preprocessor (defaults to models/preprocessor.joblib).

    Returns:
        np.ndarray of shape (1, n_qubits) with values in [0, pi].
    """
    path = preprocessor_path or default_config.preprocessor_path
    if not path.exists():
        raise FileNotFoundError(
            f"Fitted preprocessor not found at {path}. Run preprocessing first."
        )

    pipeline: Pipeline = joblib.load(path)
    df_patient = pd.DataFrame([patient_dict])
    vector = pipeline.transform(df_patient)
    return vector
