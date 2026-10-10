"""Dataset loader supporting UCI Heart Disease (Cleveland) and Pima Indians Diabetes."""
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.utils.config import DATASET_REGISTRY, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("data_loader")

# UCI Cleveland Heart Disease attribute names as documented by UCI
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Pima Indians Diabetes columns
DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

HEART_DIRECT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DIABETES_DIRECT_DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


def _fetch_heart_uci() -> pd.DataFrame:
    """Fetch UCI Cleveland Heart Disease dataset via ucimlrepo or direct HTTP fallbacks."""
    # Attempt 1: Try ucimlrepo package
    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Attempting to fetch Heart Disease dataset via ucimlrepo (id=45)...")
        heart_repo = fetch_ucirepo(id=45)
        X = heart_repo.data.features
        y = heart_repo.data.targets
        df = pd.concat([X, y], axis=1)
        logger.info("Successfully fetched heart disease dataset via ucimlrepo.")
        return df
    except Exception as e:
        logger.warning("ucimlrepo fetch failed (%s). Falling back to direct URL...", e)

    # Attempt 2: Direct processed.cleveland.data download
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(HEART_DIRECT_DATA_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_text = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw_text), names=HEART_COLUMNS, na_values=["?"])
        logger.info("Successfully fetched heart disease dataset from UCI direct repository.")
        return df
    except Exception as e:
        logger.warning("Direct cleveland.data fetch failed (%s). Trying UCI zip bundle...", e)

    # Attempt 3: UCI public zip
    zip_url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
    req_zip = urllib.request.Request(zip_url, headers=headers)
    with urllib.request.urlopen(req_zip, timeout=20) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with z.open("processed.cleveland.data") as f:
            df = pd.read_csv(f, names=HEART_COLUMNS, na_values=["?"])
    logger.info("Successfully extracted heart disease dataset from UCI zip archive.")
    return df


def _fetch_diabetes() -> pd.DataFrame:
    """Fetch Pima Indians Diabetes dataset."""
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(DIABETES_DIRECT_DATA_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw_text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw_text), names=DIABETES_COLUMNS)
    logger.info("Successfully fetched Pima Indians Diabetes dataset.")
    return df


def load_dataset(name: str = "heart", raw: bool = False) -> pd.DataFrame:
    """
    Loads, caches, and optionally binarizes the selected clinical dataset.

    Args:
        name: Name of the dataset ("heart" or "diabetes").
        raw: If True, returns raw unmodified DataFrame without binarization/cleanups.

    Returns:
        pd.DataFrame containing features and binary target column.
    """
    dataset_name = name.lower()
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset '{name}'. Supported datasets: {list(DATASET_REGISTRY.keys())}"
        )

    cache_file = RAW_DATA_DIR / f"{dataset_name}_raw.csv"

    if cache_file.exists():
        logger.info("Loading cached raw dataset from %s", cache_file)
        df = pd.read_csv(cache_file)
    else:
        logger.info("Cache not found. Downloading '%s' dataset...", dataset_name)
        if dataset_name == "heart":
            df = _fetch_heart_uci()
        elif dataset_name == "diabetes":
            df = _fetch_diabetes()
        else:
            raise NotImplementedError(f"Downloader for {dataset_name} not implemented.")

        # Cache raw CSV
        df.to_csv(cache_file, index=False)
        logger.info("Cached raw dataset to %s (shape: %s)", cache_file, df.shape)

    if raw:
        return df

    # Clinical cleanups & target binarization
    df_clean = df.copy()

    if dataset_name == "heart":
        # Target column: 'num' (0 = absence, 1, 2, 3, 4 = presence of heart disease)
        target_col = DATASET_REGISTRY["heart"].target_column
        if target_col in df_clean.columns:
            # Binarize: 0 -> 0 (low risk/no disease), >=1 -> 1 (high risk/disease present)
            df_clean[target_col] = (df_clean[target_col].astype(float) > 0).astype(int)
            logger.info(
                "Heart disease target '%s' binarized: %s",
                target_col,
                df_clean[target_col].value_counts().to_dict(),
            )
        # Ensure numerical casting where missing markers might exist
        for col in ["ca", "thal"]:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    elif dataset_name == "diabetes":
        target_col = DATASET_REGISTRY["diabetes"].target_column
        logger.info(
            "Diabetes target '%s' balance: %s",
            target_col,
            df_clean[target_col].value_counts().to_dict(),
        )

    logger.info("Dataset '%s' loaded successfully with shape: %s", dataset_name, df_clean.shape)
    return df_clean


if __name__ == "__main__":
    print("-" * 50)
    print("Testing Heart Disease dataset loader:")
    df_heart = load_dataset("heart")
    print(df_heart.head())
    print("\nTarget balance:")
    print(df_heart["num"].value_counts())
    print("-" * 50)
