"""Inference endpoint for clinical health risk prediction."""
from typing import Optional, Union
from fastapi import APIRouter, Body, Query, HTTPException
from api.model_service import ModelService
from api.schemas import PatientData, PredictRequest, PredictResponse

router = APIRouter(tags=["Inference"])


class FlatPredictRequest(PatientData):
    """Allows patient fields directly at top-level with optional model_name and compare."""
    model_name: Optional[str] = "production"
    compare: bool = False


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Patient Cardiac Disease Risk",
    description=(
        "Performs real-time clinical risk inference on patient biomarkers using the production "
        "trained model (or a selectable quantum/classical model). Provides probability, risk stratification "
        "(Low/Moderate/High), top contributing biomarker attributions, and optional side-by-side comparison across all models."
    ),
)
async def predict_risk(
    payload: Union[PredictRequest, FlatPredictRequest] = Body(
        ...,
        description="Patient clinical fields with optional model_name and compare flag.",
        openapi_examples={
            "high_risk_patient": {
                "summary": "High Risk Patient (Asymptomatic, Elevated Oldpeak & CA)",
                "value": {
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
                    "compare": True,
                },
            },
            "low_risk_patient": {
                "summary": "Low Risk Patient (Normal ECG, High Reserve, No Defect)",
                "value": {
                    "age": 42.0,
                    "sex": 0,
                    "cp": 2,
                    "trestbps": 120.0,
                    "chol": 195.0,
                    "fbs": 0,
                    "restecg": 0,
                    "thalach": 175.0,
                    "exang": 0,
                    "oldpeak": 0.0,
                    "slope": 1,
                    "ca": 0.0,
                    "thal": 3.0,
                    "compare": False,
                },
            },
        },
    ),
    model_name: Optional[str] = Query(
        None,
        description="Override model to execute ('production', 'MLP', 'VQC', 'QSVC', 'Random Forest', 'Logistic Regression', 'SVM-RBF')",
    ),
    compare: Optional[bool] = Query(
        None,
        description="Override compare flag to evaluate all models side-by-side",
    ),
) -> PredictResponse:
    """Evaluates clinical features and returns risk predictions and attributions in < 2 seconds."""
    service = ModelService.get_instance()

    # Extract patient and options depending on payload format
    if isinstance(payload, PredictRequest):
        patient = payload.patient
        req_model = payload.model_name
        req_compare = payload.compare
    else:
        # Flat format: extract patient fields
        patient_data_dict = {
            k: v for k, v in payload.model_dump().items() if k not in ["model_name", "compare"]
        }
        patient = PatientData(**patient_data_dict)
        req_model = payload.model_name
        req_compare = payload.compare

    # Query params take precedence if explicitly provided
    chosen_model = model_name if model_name is not None else (req_model or "production")
    chosen_compare = compare if compare is not None else req_compare

    try:
        response = service.predict(
            patient=patient,
            model_name=chosen_model,
            compare=chosen_compare,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference execution failed: {str(e)}",
        ) from e
