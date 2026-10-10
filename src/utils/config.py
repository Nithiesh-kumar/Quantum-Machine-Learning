"""Central configuration module for Quantum ML Health Risk System."""
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Literal

# Root directory: directory containing the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Core subdirectories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_RESULTS_DIR = RESULTS_DIR / "eda"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, EDA_RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# System constants
SEED: int = 42
MAX_QUBITS: int = 8
DEFAULT_N_QUBITS: int = 8
ANGLE_RANGE = (0.0, 3.141592653589793)  # [0, pi] for angle encoding

# Mandatory Medical Disclaimer
MEDICAL_DISCLAIMER: str = (
    "DISCLAIMER: This system is an educational and experimental Quantum Machine "
    "Learning prototype. It is NOT certified for medical diagnostics or clinical "
    "decision-making. Always consult qualified healthcare professionals."
)


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata and schema specification for supported clinical datasets."""
    name: str
    target_column: str
    categorical_columns: List[str]
    numerical_columns: List[str]
    uci_id: int | None = None
    direct_url: str | None = None
    description: str = ""


# Dataset Registry
DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "heart": DatasetMetadata(
        name="heart",
        target_column="num",
        uci_id=45,
        direct_url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        description="UCI Heart Disease (Cleveland Clinic) tabular dataset for binary heart disease risk.",
        categorical_columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
        numerical_columns=["age", "trestbps", "chol", "thalach", "oldpeak", "ca"],
    ),
    "diabetes": DatasetMetadata(
        name="diabetes",
        target_column="Outcome",
        uci_id=None,
        direct_url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        description="Pima Indians Diabetes dataset for binary diabetes onset risk.",
        categorical_columns=[],
        numerical_columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    ),
}


@dataclass
class QMLConfig:
    """Global configuration settings for pipeline execution."""
    dataset_name: Literal["heart", "diabetes"] = "heart"
    n_qubits: int = DEFAULT_N_QUBITS
    seed: int = SEED
    reduction_method: Literal["pca", "select_k_best"] = "pca"
    
    # Train / Val / Test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Artifact paths
    preprocessor_path: Path = MODELS_DIR / "preprocessor.pkl"
    preprocessor_joblib_path: Path = MODELS_DIR / "preprocessor.joblib"
    vqc_model_path: Path = MODELS_DIR / "vqc_model.joblib"
    qsvc_model_path: Path = MODELS_DIR / "qsvc_model.joblib"
    classical_model_path: Path = MODELS_DIR / "classical_best.joblib"
    metrics_path: Path = RESULTS_DIR / "metrics.json"
    classical_metrics_path: Path = RESULTS_DIR / "classical_metrics.json"


    def __post_init__(self):
        if self.n_qubits > MAX_QUBITS:
            raise ValueError(f"n_qubits ({self.n_qubits}) exceeds MAX_QUBITS limit ({MAX_QUBITS}).")
        if abs((self.train_ratio + self.val_ratio + self.test_ratio) - 1.0) > 1e-5:
            raise ValueError("Split ratios (train, val, test) must sum to 1.0.")

    @property
    def dataset(self) -> DatasetMetadata:
        if self.dataset_name not in DATASET_REGISTRY:
            raise KeyError(
                f"Unknown dataset '{self.dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}"
            )
        return DATASET_REGISTRY[self.dataset_name]


# Active configuration instance (respects DATASET env var if set)
ACTIVE_DATASET: str = os.getenv("DATASET", "heart").lower()
default_config = QMLConfig(dataset_name="heart" if ACTIVE_DATASET not in DATASET_REGISTRY else ACTIVE_DATASET)
