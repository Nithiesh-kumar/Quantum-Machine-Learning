"""Metrics endpoint serving benchmark comparison data from /results."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from api.model_service import ModelService
from api.schemas import MetricsResponse, ModelBenchmarkEntry
from src.utils.config import RESULTS_DIR

router = APIRouter(tags=["Metrics & Benchmarks"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get Model Comparison Benchmark Results",
    description="Returns cross-algorithm performance benchmarks across all 6 classical and quantum models (mean ± std over 5 seeds).",
)
async def get_metrics() -> MetricsResponse:
    """Reads and parses benchmark results and comparison tables from /results."""
    service = ModelService.get_instance()
    csv_path = RESULTS_DIR / "model_comparison.csv"
    meta_path = RESULTS_DIR / "production_model_metadata.json"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark results not found at {csv_path}. Run src.evaluation.compare first.",
        )

    production_winner = "MLP"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
                production_winner = meta_json.get("production_model_name", "MLP")
        except Exception:
            pass

    table_entries: List[ModelBenchmarkEntry] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_name = row["model"]
                brier = service.brier_scores.get(m_name, None)
                entry = ModelBenchmarkEntry(
                    model_name=m_name,
                    model_type=row.get("model_type", "classical").capitalize(),
                    accuracy=round(float(row["accuracy_mean"]), 4),
                    precision=round(float(row["precision_mean"]), 4),
                    recall=round(float(row["recall_mean"]), 4),
                    specificity=round(float(row["specificity_mean"]), 4),
                    f1_score=round(float(row["f1_mean"]), 4),
                    roc_auc=round(float(row["roc_auc_mean"]), 4),
                    parameters=int(float(row.get("param_count", 0))),
                    training_time_seconds=round(float(row["train_time_mean_seconds"]), 4),
                    brier_score=brier,
                    std_metrics={
                        "accuracy_std": round(float(row.get("accuracy_std", 0.0)), 4),
                        "recall_std": round(float(row.get("recall_std", 0.0)), 4),
                        "f1_std": round(float(row.get("f1_std", 0.0)), 4),
                        "roc_auc_std": round(float(row.get("roc_auc_std", 0.0)), 4),
                    },
                )
                table_entries.append(entry)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse comparison CSV: {str(e)}",
        ) from e

    return MetricsResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        evaluation_dataset="UCI Cleveland Heart Disease (Binary Cardiac Risk)",
        production_model_winner=production_winner,
        comparison_table=table_entries,
        brier_scores=service.brier_scores,
        raw_csv_url="/static/model_comparison.csv",
        summary_markdown_url="/static/results_summary.md",
    )
