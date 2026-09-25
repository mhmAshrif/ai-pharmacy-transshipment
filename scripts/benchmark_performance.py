#!/usr/bin/env python3
"""Benchmark end-to-end pipeline timing for NFR02.

Runs: Data Fusion -> Prophet Forecasts -> PuLP Optimization -> (optional) DB persistence.
Prints per-phase latencies and exits with code 0 when total <= 60s, otherwise exits 2.
"""
import time
import os
import sys

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_pipeline import fuse_healthcare_data
from src.forecast_engine import generate_demand_forecasts
from src.optimization_engine import run_transshipment_optimization


def human(ms):
    return f"{ms:.3f}s"


def main():
    print("Starting benchmark: Data Fusion -> Forecasting -> Optimization")
    start_total = time.perf_counter()

    t0 = time.perf_counter()
    try:
        fuse_healthcare_data()
    except Exception as e:
        print(f"Data fusion failed: {e}")
        sys.exit(3)
    t1 = time.perf_counter()
    print(f"Phase 1 (Data Fusion): {human(t1-t0)}")

    try:
        generate_demand_forecasts()
    except Exception as e:
        print(f"Forecast generation failed: {e}")
        sys.exit(4)
    t2 = time.perf_counter()
    print(f"Phase 2 (Forecasting): {human(t2-t1)}")

    try:
        result = run_transshipment_optimization()
    except Exception as e:
        print(f"Optimization failed: {e}")
        sys.exit(5)
    t3 = time.perf_counter()
    print(f"Phase 3 (Optimization): {human(t3-t2)}")

    # Optional persistence timing (reading/writing CSVs already done by components)
    total = time.perf_counter() - start_total
    print(f"Total E2E time: {human(total)}")

    out = {
        "phase_times": {
            "data_fusion": round(t1-t0, 3),
            "forecasting": round(t2-t1, 3),
            "optimization": round(t3-t2, 3),
        },
        "total_seconds": round(total, 3),
        "manifest_routes": len(result.get("manifest", [])) if isinstance(result, dict) else 0,
    }

    import json
    print(json.dumps(out, indent=2))

    NFR_LIMIT = float(os.getenv("NFR02_LIMIT_S", "60"))
    if total <= NFR_LIMIT:
        print(f"SUCCESS: Completed within NFR02 limit ({NFR_LIMIT}s)")
        sys.exit(0)
    else:
        print(f"FAIL: Exceeded NFR02 limit ({NFR_LIMIT}s). Total: {total:.2f}s")
        sys.exit(2)


if __name__ == "__main__":
    main()
