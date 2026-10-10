"""FastAPI main application entrypoint for the Quantum ML Health Risk System."""
from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.model_service import ModelService
from api.routes import datasets, health, metrics, model_info, predict
from src.utils.config import MEDICAL_DISCLAIMER, RESULTS_DIR, PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager ensuring that all models and preprocessors
    are loaded into memory exactly once at startup.
    """
    logger.info("Starting up QML Health Risk API...")
    try:
        service = ModelService.get_instance()
        service.initialize()
        logger.info(
            "Startup complete. Production model '%s' loaded and ready for inference.",
            service.production_model_name,
        )
    except Exception as e:
        logger.error("Error during model initialization at startup: %s", e, exc_info=True)
    yield
    logger.info("Shutting down QML Health Risk API.")


# FastAPI Application Instance
app = FastAPI(
    title="Quantum Machine Learning Health Risk API",
    description=(
        "Production-ready clinical risk prediction API integrating Variational Quantum Classifiers (PennyLane), "
        "Quantum Support Vector Classifiers (Qiskit), and optimized classical ML baselines on clinical patient data.\n\n"
        f"**Medical Advisory**: {MEDICAL_DISCLAIMER}"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


import os

# Read allowed origins from env var (comma-separated)
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
if not allowed_origins:
    allowed_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=False if "*" in allowed_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Logs incoming HTTP requests, response status codes, and execution latency."""
    start_time = time.time()
    path = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        logger.info(
            "[%s] %s -> Status: %d | Latency: %.2f ms",
            method, path, response.status_code, duration_ms
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        logger.error(
            "[%s] %s -> Exception: %s | Latency: %.2f ms",
            method, path, str(exc), duration_ms, exc_info=True
        )
        raise exc


# ---------------------------------------------------------------------------
# Structured Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors into standardized structured JSON."""
    formatted_errors = []
    for err in exc.errors():
        loc = " -> ".join([str(item) for item in err.get("loc", [])])
        formatted_errors.append({
            "field": loc,
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        })

    logger.warning("Validation error on %s: %s", request.url.path, formatted_errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Input clinical biomarker payload failed validation criteria.",
            "details": formatted_errors,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Standardizes HTTP error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions and returns clean 500 responses."""
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "medical_disclaimer": MEDICAL_DISCLAIMER,
        },
    )


# ---------------------------------------------------------------------------
# Static Files Mount & Root Route
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=RESULTS_DIR), name="static")

FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend_app")


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects the root URL to /app if frontend exists, or to /docs."""
    if FRONTEND_DIST.exists():
        return RedirectResponse(url="/app")
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Router Registration
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(model_info.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
