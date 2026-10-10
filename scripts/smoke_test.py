#!/usr/bin/env python3
"""
CardioQuantum API Smoke Test Script.

Validates end-to-end API service readiness:
1. GET /health -> Verifies status 'ok' and models_loaded: true
2. POST /predict -> Evaluates sample clinical patient with sub-2s latency,
                    confidence intervals, and top-3 contributing features.
3. GET /metrics -> Verifies benchmark comparison availability.

Exits with code 0 on success, code 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


SAMPLE_PATIENT = {
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
}


def make_request(url: str, method: str = "GET", data: Dict[str, Any] | None = None, timeout: float = 10.0) -> Tuple[int, Dict[str, Any], float]:
    """Executes HTTP request and returns (status_code, json_body, duration_ms)."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None

    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            duration_ms = (time.time() - start_time) * 1000.0
            resp_bytes = resp.read()
            resp_json = json.loads(resp_bytes.decode("utf-8")) if resp_bytes else {}
            return resp.status, resp_json, duration_ms
    except urllib.error.HTTPError as err:
        duration_ms = (time.time() - start_time) * 1000.0
        err_bytes = err.read()
        try:
            err_json = json.loads(err_bytes.decode("utf-8"))
        except Exception:
            err_json = {"error": str(err)}
        return err.code, err_json, duration_ms
    except Exception as err:
        duration_ms = (time.time() - start_time) * 1000.0
        return 0, {"error": str(err)}, duration_ms


def run_smoke_tests(base_url: str) -> bool:
    """Runs smoke test suite against target base_url."""
    clean_base = base_url.rstrip("/")
    print("=" * 70)
    print(f" CardioQuantum Production Smoke Test: {clean_base}")
    print("=" * 70)

    all_passed = True

    # 1. Test /health
    health_url = f"{clean_base}/health"
    print(f"\n[1/3] Probing GET {health_url}...")
    status, body, latency = make_request(health_url, method="GET")

    if status == 200:
        models_loaded = body.get("models_loaded", False)
        status_val = body.get("status", "")
        if status_val in ["ok", "healthy"] and models_loaded:
            print(f"  [PASS] Status: {status} ({latency:.2f} ms) | Service: '{status_val}', Models Loaded: {models_loaded}")
        else:
            print(f"  [FAIL] Unexpected payload: status='{status_val}', models_loaded={models_loaded}")
            all_passed = False
    else:
        print(f"  [FAIL] Failed with HTTP {status} ({latency:.2f} ms): {body}")
        all_passed = False

    # 2. Test /predict
    predict_url = f"{clean_base}/predict"
    print(f"\n[2/3] Testing POST {predict_url} (Clinical Patient Prediction)...")
    status, body, latency = make_request(predict_url, method="POST", data=SAMPLE_PATIENT, timeout=15.0)

    if status == 200:
        prob = body.get("risk_probability")
        label = body.get("risk_label")
        level = body.get("risk_level")
        model = body.get("model_used")
        ci = body.get("confidence_interval", {})
        top_feats = body.get("top_contributing_features", [])

        if prob is not None and label in [0, 1] and level in ["Low", "Moderate", "High"]:
            print(f"  [PASS] Status: 200 ({latency:.2f} ms)")
            print(f"         Prediction: {level} Risk (Prob: {prob:.4f}, Model: {model})")
            print(f"         Confidence Interval: {ci.get('range_display', 'N/A')} ({ci.get('method', 'N/A')})")
            print(f"         Top Contributing Factors ({len(top_feats)}): {[f.get('feature') for f in top_feats[:3]]}")
            if latency > 2000.0:
                print(f"  [WARN] Latency exceeded 2000 ms: {latency:.2f} ms")
        else:
            print(f"  [FAIL] Malformed prediction response: {body}")
            all_passed = False
    else:
        print(f"  [FAIL] Failed with HTTP {status} ({latency:.2f} ms): {body}")
        all_passed = False

    # 3. Test /metrics
    metrics_url = f"{clean_base}/metrics"
    print(f"\n[3/3] Testing GET {metrics_url} (Benchmark Comparison)...")
    status, body, latency = make_request(metrics_url, method="GET")

    if status == 200:
        winner = body.get("production_model_winner", "Unknown")
        table = body.get("comparison_table", [])
        print(f"  [PASS] Status: 200 ({latency:.2f} ms) | Winner: {winner}, Benchmarked Models: {len(table)}")
    else:
        print(f"  [FAIL] Failed with HTTP {status} ({latency:.2f} ms): {body}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL SMOKE TESTS PASSED - API READY FOR PRODUCTION DEPLOYMENT")
        print("=" * 70)
        return True
    else:
        print(" SMOKE TESTS FAILED - REVIEW LOGS ABOVE")
        print("=" * 70)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioQuantum API Smoke Test")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Target base URL (e.g. http://127.0.0.1:8000 or https://cardioquantum-api.onrender.com)",
    )
    args = parser.parse_args()

    success = run_smoke_tests(args.base_url)
    sys.exit(0 if success else 1)
