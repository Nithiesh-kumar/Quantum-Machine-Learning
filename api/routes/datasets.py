"""Datasets metadata endpoint listing supported clinical conditions and schema fields."""
from fastapi import APIRouter
from api.schemas import DatasetsResponse, DatasetInfo, FieldDefinition

router = APIRouter(tags=["Datasets & Metadata"])


HEART_FIELDS = [
    FieldDefinition(
        name="age",
        type="float",
        range="18.0 - 120.0",
        units="years",
        description="Patient age at clinical presentation.",
    ),
    FieldDefinition(
        name="sex",
        type="binary",
        range="0 or 1",
        units="categorical",
        description="Biological sex.",
        options={"0": "Female", "1": "Male"},
    ),
    FieldDefinition(
        name="cp",
        type="categorical",
        range="1 - 4",
        units="class",
        description="Chest pain severity classification.",
        options={
            "1": "Typical angina",
            "2": "Atypical angina",
            "3": "Non-anginal pain",
            "4": "Asymptomatic ischemia",
        },
    ),
    FieldDefinition(
        name="trestbps",
        type="float",
        range="60.0 - 260.0",
        units="mm Hg",
        description="Resting blood pressure upon hospital admission.",
    ),
    FieldDefinition(
        name="chol",
        type="float",
        range="80.0 - 650.0",
        units="mg/dl",
        description="Serum cholestoral measurement.",
    ),
    FieldDefinition(
        name="fbs",
        type="binary",
        range="0 or 1",
        units="binary",
        description="Fasting blood sugar > 120 mg/dl.",
        options={"0": "Normal (<120 mg/dl)", "1": "Elevated (>120 mg/dl)"},
    ),
    FieldDefinition(
        name="restecg",
        type="categorical",
        range="0 - 2",
        units="class",
        description="Resting electrocardiographic results.",
        options={
            "0": "Normal",
            "1": "ST-T wave abnormality",
            "2": "Definite left ventricular hypertrophy",
        },
    ),
    FieldDefinition(
        name="thalach",
        type="float",
        range="40.0 - 250.0",
        units="bpm",
        description="Maximum heart rate achieved during exercise stress testing.",
    ),
    FieldDefinition(
        name="exang",
        type="binary",
        range="0 or 1",
        units="binary",
        description="Exercise-induced angina symptom.",
        options={"0": "No", "1": "Yes"},
    ),
    FieldDefinition(
        name="oldpeak",
        type="float",
        range="0.0 - 8.0",
        units="mm",
        description="ST depression induced by exercise relative to rest.",
    ),
    FieldDefinition(
        name="slope",
        type="categorical",
        range="1 - 3",
        units="class",
        description="Slope of the peak exercise ST segment.",
        options={"1": "Upsloping", "2": "Flat", "3": "Downsloping"},
    ),
    FieldDefinition(
        name="ca",
        type="float",
        range="0.0 - 3.0",
        units="vessels",
        description="Number of major cardiac vessels colored by fluoroscopy.",
    ),
    FieldDefinition(
        name="thal",
        type="float",
        range="0.0 - 7.0",
        units="class",
        description="Thallium scintigraphy stress defect status.",
        options={"3": "Normal", "6": "Fixed defect", "7": "Reversible defect"},
    ),
]

DIABETES_FIELDS = [
    FieldDefinition(name="Pregnancies", type="integer", range="0 - 20", units="count", description="Number of times pregnant."),
    FieldDefinition(name="Glucose", type="float", range="0.0 - 300.0", units="mg/dl", description="Plasma glucose concentration in oral glucose tolerance test."),
    FieldDefinition(name="BloodPressure", type="float", range="0.0 - 180.0", units="mm Hg", description="Diastolic blood pressure."),
    FieldDefinition(name="SkinThickness", type="float", range="0.0 - 100.0", units="mm", description="Triceps skin fold thickness."),
    FieldDefinition(name="Insulin", type="float", range="0.0 - 900.0", units="mu U/ml", description="Two-hour serum insulin."),
    FieldDefinition(name="BMI", type="float", range="0.0 - 70.0", units="kg/m^2", description="Body mass index."),
    FieldDefinition(name="DiabetesPedigreeFunction", type="float", range="0.0 - 3.0", units="score", description="Diabetes pedigree function genetic score."),
    FieldDefinition(name="Age", type="integer", range="18 - 120", units="years", description="Age in years."),
]


@router.get(
    "/datasets",
    response_model=DatasetsResponse,
    summary="List Supported Clinical Datasets & Fields",
    description="Enumerates supported healthcare conditions, clinical biomarker definitions, valid ranges, and measurement units.",
)
async def get_datasets() -> DatasetsResponse:
    """Returns dataset schemas for client-side dynamic form generation and validation."""
    heart_condition = DatasetInfo(
        condition_name="Coronary Heart Disease Risk",
        dataset_key="heart",
        sample_count=303,
        target_variable="num",
        target_classes={"0": "Low Risk (<50% vessel diameter narrowing)", "1": "High Risk (>=50% diameter narrowing)"},
        quantum_reduced_features=4,
        description="UCI Cleveland Clinic Heart Disease benchmark. Primary dataset actively trained in the quantum circuit pipeline.",
        fields=HEART_FIELDS,
    )

    diabetes_condition = DatasetInfo(
        condition_name="Type 2 Diabetes Onset Risk",
        dataset_key="diabetes",
        sample_count=768,
        target_variable="Outcome",
        target_classes={"0": "Negative (Non-diabetic)", "1": "Positive (Diabetic)"},
        quantum_reduced_features=4,
        description="National Institute of Diabetes and Digestive and Kidney Diseases Pima Indian female cohort.",
        fields=DIABETES_FIELDS,
    )

    return DatasetsResponse(
        supported_conditions=[heart_condition, diabetes_condition],
        count=2,
    )
