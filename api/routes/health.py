"""Health check and service status endpoint."""
from datetime import datetime, timezone
from fastapi import APIRouter
from api.model_service import ModelService
from api.schemas import HealthResponse

router = APIRouter(tags=["Health & Status"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns the operating status of the QML Health Risk API, including model load status and active dataset.",
)
async def get_health() -> HealthResponse:
    """Verifies that the service and machine learning model instances are initialized and responsive."""
    service = ModelService.get_instance()
    loaded_count = len(service.models_registry)
    status_str = "healthy" if service.is_loaded and service.production_model is not None else "degraded"

    return HealthResponse(
        status=status_str,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        models_loaded=service.is_loaded,
        loaded_models_count=loaded_count,
        active_dataset="UCI Cleveland Heart Disease",
        production_model=service.production_model_name,
    )
