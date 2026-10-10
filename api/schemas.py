"""Pydantic schemas and data contracts for FastAPI service."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Patient Input Schemas
# ---------------------------------------------------------------------------
class PatientData(BaseModel):
    """
    Standardized clinical input fields for cardiac risk prediction.
    Validated with realistic clinical ranges based on UCI Cleveland criteria.
    """
    age: float = Field(
        ...,
        ge=18.0,
        le=120.0,
        description="Patient age in years",
        json_schema_extra={"example": 60.0},
    )
    sex: int = Field(
        ...,
        ge=0,
        le=1,
        description="Biological sex (0 = Female, 1 = Male)",
        json_schema_extra={"example": 1},
    )
    cp: int = Field(
        ...,
        ge=1,
        le=4,
        description="Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
        json_schema_extra={"example": 4},
    )
    trestbps: float = Field(
        ...,
        ge=60.0,
        le=260.0,
        description="Resting blood pressure (mm Hg)",
        json_schema_extra={"example": 145.0},
    )
    chol: float = Field(
        ...,
        ge=80.0,
        le=650.0,
        description="Serum cholestoral (mg/dl)",
        json_schema_extra={"example": 260.0},
    )
    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)",
        json_schema_extra={"example": 0},
    )
    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)",
        json_schema_extra={"example": 2},
    )
    thalach: float = Field(
        ...,
        ge=40.0,
        le=250.0,
        description="Maximum heart rate achieved (bpm)",
        json_schema_extra={"example": 142.0},
    )
    exang: int = Field(
        ...,
        ge=0,
        le=1,
        description="Exercise induced angina (0 = No, 1 = Yes)",
        json_schema_extra={"example": 1},
    )
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=8.0,
        description="ST depression induced by exercise relative to rest (mm)",
        json_schema_extra={"example": 2.5},
    )
    slope: int = Field(
        ...,
        ge=1,
        le=3,
        description="Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)",
        json_schema_extra={"example": 2},
    )
    ca: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Number of major vessels colored by fluoroscopy (0 to 3)",
        json_schema_extra={"example": 2.0},
    )
    thal: float = Field(
        ...,
        ge=0.0,
        le=7.0,
        description="Thalassemia status (3: normal, 6: fixed defect, 7: reversible defect)",
        json_schema_extra={"example": 7.0},
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 60.0,
                "sex": 1,
                "cp": 4,
                "trestbps": 145.0,
                "chol": 260.0,
                "fbs": 0,
                "restecg": 2,
                "thalach": 142.0,
                "exang": 1,
                "oldpeak": 2.5,
                "slope": 2,
                "ca": 2.0,
                "thal": 7.0,
            }
        },
    )


class PredictRequest(BaseModel):
    """Prediction request payload with optional model selection and compare flag."""
    patient: PatientData
    model_name: Optional[str] = Field(
        default="production",
        description="Target model for prediction ('production', 'MLP', 'VQC', 'QSVC', 'Random Forest', 'Logistic Regression', 'SVM-RBF')",
        json_schema_extra={"example": "production"},
    )
    compare: bool = Field(
        default=False,
        description="When True, evaluates patient across all supported classical and quantum models side-by-side",
        json_schema_extra={"example": False},
    )


# ---------------------------------------------------------------------------
# Prediction Response Schemas
# ---------------------------------------------------------------------------
class ConfidenceInterval(BaseModel):
    """Uncertainty quantification bounds for clinical prediction."""
    lower: float = Field(..., description="Lower bound of probability confidence interval")
    upper: float = Field(..., description="Upper bound of probability confidence interval")
    confidence_level: float = Field(default=0.95, description="Confidence interval coverage level")
    uncertainty_std: float = Field(..., description="Standard error / uncertainty standard deviation")
    range_display: str = Field(..., description="Formatted string for user display, e.g. '88% – 96% (95% CI)'")
    method: str = Field(..., description="Uncertainty quantification methodology")


class FeatureContribution(BaseModel):
    """Attribution metric for a single clinical biomarker."""
    feature: str = Field(..., description="Biomarker name")
    patient_value: float = Field(..., description="Input value from patient record")
    baseline_mean: float = Field(..., description="Population training mean")
    importance: float = Field(..., description="Relative contribution magnitude")
    directional_impact: str = Field(..., description="'Elevates Risk' or 'Reduces Risk'")


class SingleModelPrediction(BaseModel):
    """Detailed prediction output for an individual model."""
    model_name: str
    model_type: Literal["Classical", "Quantum"]
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_label: int = Field(..., ge=0, le=1)
    risk_level: Literal["Low", "Moderate", "High"]
    confidence_percentage: float
    confidence_interval: Optional[ConfidenceInterval] = None
    brier_score: Optional[float] = None


class PredictResponse(BaseModel):
    """Structured response contract for POST /predict."""
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of cardiac disease")
    risk_label: int = Field(..., ge=0, le=1, description="Binary classification (0 = Low Risk, 1 = High Risk)")
    risk_level: Literal["Low", "Moderate", "High"] = Field(..., description="Categorical risk stratification")
    model_used: str = Field(..., description="Name of the algorithm producing the primary prediction")
    model_type: str = Field(..., description="'Classical' or 'Quantum'")
    confidence_interval: ConfidenceInterval = Field(
        ..., description="Statistical confidence interval and uncertainty estimation"
    )
    brier_score: Optional[float] = Field(
        None, description="Brier score indicating model calibration quality (lower is better, <0.25)"
    )
    quantum_representation: Optional[List[float]] = Field(
        None, description="Scaled [0, pi] quantum angle feature vector fed into the circuit"
    )
    top_contributing_features: List[FeatureContribution] = Field(
        ..., description="Top biomarkers driving the clinical risk assessment"
    )
    comparison: Optional[Dict[str, SingleModelPrediction]] = Field(
        None, description="Side-by-side predictions across all models if compare=True"
    )
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    medical_disclaimer: str = Field(..., description="Mandatory medical advisory notice")


# ---------------------------------------------------------------------------
# Dataset and Field Info Schemas
# ---------------------------------------------------------------------------
class FieldDefinition(BaseModel):
    name: str
    type: str
    range: str
    units: Optional[str] = None
    description: str
    options: Optional[Dict[str, str]] = None


class DatasetInfo(BaseModel):
    condition_name: str
    dataset_key: str
    sample_count: int
    target_variable: str
    target_classes: Dict[str, str]
    quantum_reduced_features: int
    description: str
    fields: List[FieldDefinition]


class DatasetsResponse(BaseModel):
    supported_conditions: List[DatasetInfo]
    count: int


# ---------------------------------------------------------------------------
# Metrics and Model Info Schemas
# ---------------------------------------------------------------------------
class ModelBenchmarkEntry(BaseModel):
    model_name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    parameters: int
    training_time_seconds: float
    brier_score: Optional[float] = None
    std_metrics: Optional[Dict[str, float]] = None


class MetricsResponse(BaseModel):
    timestamp: str
    evaluation_dataset: str
    production_model_winner: str
    comparison_table: List[ModelBenchmarkEntry]
    brier_scores: Optional[Dict[str, float]] = None
    raw_csv_url: str
    summary_markdown_url: str


class QuantumCircuitDetails(BaseModel):
    qubits: int
    circuit_depth: int
    ansatz: str
    feature_map: str
    measurement: str
    loss_function: str
    optimizer: str
    training_epochs: int
    batch_size: int
    learning_rate: float
    noisy_simulation_available: bool


class QuantumKernelDetails(BaseModel):
    qubits: int
    reps: int
    entanglement: str
    feature_map: str
    kernel_class: str
    regularizer_C: float
    kernel_caching_enabled: bool


class ModelInfoResponse(BaseModel):
    production_model: Dict[str, Any]
    quantum_vqc: QuantumCircuitDetails
    quantum_qsvc: QuantumKernelDetails
    circuit_diagram_url: str
    calibration_figure_url: str = "/static/figures/calibration_curves.png"
    brier_scores: Optional[Dict[str, float]] = None
    global_feature_importances: Optional[Dict[str, float]] = None
    figures: Dict[str, str]
    system_device: str
    medical_disclaimer: str


# ---------------------------------------------------------------------------
# Health Response Schema
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str
    version: str
    models_loaded: bool
    loaded_models_count: int
    active_dataset: str
    production_model: str
