# src/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys
import math

# Append the project root to path to ensure crisp absolute internal source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your working functional script methods
from src.data_pipeline import fuse_healthcare_data
from src.forecast_engine import generate_demand_forecasts
from src.optimization_engine import run_transshipment_optimization

app = FastAPI(
    title="Climate-Responsive Lateral Pharmaceutical Transshipment API Backend",
    description="A centralized Object-Oriented Software Engineering framework linking predictive AI with Operations Research",
    version="1.0.0",
    docs_url="/",
    redoc_url=None,
    openapi_url="/openapi.json"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

@app.get("/health", tags=["Root"])
def read_root():
    """Simple health endpoint to verify server startup and provide quick links."""
    return {
        "message": "MedTransship API is running.",
        "endpoints": [
            "/api/pipeline/run-all",
            "/api/dashboard/manifest",
            "/api/dashboard/forecast-summary",
        ],
    }

# Enable Cross-Origin Resource Sharing (CORS) so your Next.js dashboard can connect without block policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/api/pipeline/run-all", methods=["GET", "POST"], tags=["Core Pipeline Orchestration"])
def execute_system_pipeline():
    """
    Executes Phase 1, Phase 2, and Phase 3 of the framework consecutively.
    Fuses environmental weather data, generates Facebook Prophet forecasts, and runs PuLP optimization.
    """
    try:
        print("\n--- Triggering Full API-Driven Orchestration Chain ---")
        # 1. Run Data Preprocessing and Fusion
        fuse_healthcare_data()
        
        # 2. Trigger Facebook Prophet Forecasting Brain
        generate_demand_forecasts()
        
        # 3. Compute cost-optimal routing manifests via Linear Programming
        optimized_manifest = run_transshipment_optimization()
        
        return {
            "status": "Success",
            "message": "Entire automated forecasting and redistribution optimization cycle completed.",
            "total_routes_generated": len(optimized_manifest)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Orchestration Failed: {str(e)}")

@app.get("/api/dashboard/manifest", tags=["Data Delivery Endpoints"])
def get_optimized_manifest():
    """
    Delivers the compiled operations research transshipment records to the Next.js frontend UI tables.
    """
    path = os.path.join(DATA_DIR, "optimized_transshipment_manifest.csv")
    if not os.path.exists(path):
        return {
            "message": "No active manifest dataset calculated yet. Execute /run-all first.", 
            "data": []
        }
    
    try:
        manifest_df = pd.read_csv(path)
        records = manifest_df.to_dict(orient="records")
        return {
            "count": len(records),
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data matrix: {str(e)}")

@app.get("/api/dashboard/metrics", tags=["Data Delivery Endpoints"])
def get_dashboard_metrics():
    """Returns the summary KPI numbers used on the overview dashboard."""
    manifest_path = os.path.join(DATA_DIR, "optimized_transshipment_manifest.csv")
    if not os.path.exists(manifest_path):
        return {"total_savings": 0, "stock_saved": 0, "active_manifests": 0}

    try:
        manifest_df = pd.read_csv(manifest_path)
        total_savings = float(manifest_df["financial_value_saved_lkr"].sum()) if "financial_value_saved_lkr" in manifest_df.columns else 0.0
        stock_saved = int(manifest_df["quantity_to_move"].sum()) if "quantity_to_move" in manifest_df.columns else 0
        active_manifests = int(len(manifest_df))

        return {
            "total_savings": round(total_savings, 2),
            "stock_saved": stock_saved,
            "active_manifests": active_manifests,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute dashboard metrics: {str(e)}")

@app.get("/api/dashboard/alerts", tags=["Data Delivery Endpoints"])
def get_dashboard_alerts():
    """Returns inventory alerts for the dashboard warning feed."""
    alerts_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    if not os.path.exists(alerts_path):
        return []

    try:
        alerts_df = pd.read_csv(alerts_path)
        warnings = alerts_df[(alerts_df["expiry_days_remaining"] < 60) | (alerts_df["stock_level"] < 500)].copy()
        if warnings.empty:
            warnings = alerts_df.head(8).copy()

        warnings = warnings.sort_values(["expiry_days_remaining", "stock_level"]).head(8)
        records = []
        for idx, row in warnings.iterrows():
            records.append({
                "id": idx + 1,
                "district": str(row.get("district", "Unknown")),
                "medicine": str(row.get("medicine", "Unknown")),
                "stock_level": int(row.get("stock_level", 0)),
                "expiry_days_remaining": int(row.get("expiry_days_remaining", 0)),
            })
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate alerts: {str(e)}")

@app.get("/api/dashboard/chart", tags=["Data Delivery Endpoints"])
def get_dashboard_chart():
    """Returns time-series values for the climate correlation chart widget."""
    chart_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    if not os.path.exists(chart_path):
        return []

    try:
        chart_df = pd.read_csv(chart_path)
        chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
        chart_df = chart_df.dropna(subset=["date"])

        summary = (
            chart_df.groupby(chart_df["date"].dt.date)
            .agg(units_sold=("units_sold", "sum"), precipitation_sum=("precipitation_sum", "mean"))
            .reset_index()
        )
        summary = summary.sort_values("date").head(20)
        summary["time"] = summary["date"].astype(str)
        summary["units_sold"] = summary["units_sold"].astype(float).round(2)
        summary["precipitation_sum"] = summary["precipitation_sum"].astype(float).round(2)

        return summary[["time", "units_sold", "precipitation_sum"]].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build chart data: {str(e)}")

@app.get("/api/inventory", tags=["Data Delivery Endpoints"])
def get_inventory_records():
    """Returns the current inventory ledger rows used by the inventory page."""
    inventory_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    if not os.path.exists(inventory_path):
        return []

    try:
        inventory_df = pd.read_csv(inventory_path)
        inventory_df = inventory_df[["district", "medicine", "category", "stock_level", "expiry_days_remaining", "unit_price"]].copy()
        inventory_df = inventory_df.drop_duplicates().reset_index(drop=True)
        inventory_df["stock_level"] = inventory_df["stock_level"].astype(int)
        inventory_df["expiry_days_remaining"] = inventory_df["expiry_days_remaining"].astype(int)
        inventory_df["unit_price"] = inventory_df["unit_price"].astype(float)
        return inventory_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read inventory data: {str(e)}")

@app.get("/api/forecast/options", tags=["Forecasting Endpoints"])
def get_forecast_options():
    """Returns the available medicines and districts for the forecast form controls."""
    forecast_path = os.path.join(DATA_DIR, "upcoming_demand_forecasts.csv")
    historical_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")

    medicines = []
    districts = []

    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path)
        medicines = sorted([str(item) for item in forecast_df["medicine"].dropna().astype(str).unique() if str(item).strip()])
        districts = sorted([str(item) for item in forecast_df["district"].dropna().astype(str).unique() if str(item).strip()])

    if not medicines and os.path.exists(historical_path):
        historical_df = pd.read_csv(historical_path)
        medicines = sorted([str(item) for item in historical_df["medicine"].dropna().astype(str).unique() if str(item).strip()])
        districts = sorted([str(item) for item in historical_df["district"].dropna().astype(str).unique() if str(item).strip()])

    return {
        "medicines": medicines,
        "districts": districts,
    }

@app.get("/api/forecast", tags=["Forecasting Endpoints"])
def get_forecast_series(medicine: str, district: str):
    """Returns historical actuals and future prophet forecasts for the selected medicine and district."""
    historical_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    forecast_path = os.path.join(DATA_DIR, "upcoming_demand_forecasts.csv")

    if not os.path.exists(historical_path) or not os.path.exists(forecast_path):
        return []

    try:
        historical_df = pd.read_csv(historical_path)
        forecast_df = pd.read_csv(forecast_path)

        historical_subset = historical_df[
            (historical_df["medicine"].astype(str).str.lower() == medicine.lower()) &
            (historical_df["district"].astype(str).str.lower() == district.lower())
        ].copy()

        forecast_subset = forecast_df[
            (forecast_df["medicine"].astype(str).str.lower() == medicine.lower()) &
            (forecast_df["district"].astype(str).str.lower() == district.lower())
        ].copy()

        if historical_subset.empty and forecast_subset.empty:
            return []

        historical_subset = historical_subset[["date", "units_sold", "precipitation_sum"]].copy()
        historical_subset["time"] = historical_subset["date"].astype(str)
        historical_subset["units_sold"] = pd.to_numeric(historical_subset["units_sold"], errors="coerce")
        historical_subset["predicted_demand"] = None
        historical_subset["precipitation_sum"] = pd.to_numeric(historical_subset["precipitation_sum"], errors="coerce")

        forecast_subset = forecast_subset[["date", "predicted_demand", "district", "medicine"]].copy()
        forecast_subset = forecast_subset.rename(columns={"date": "time"})
        forecast_subset["units_sold"] = None
        forecast_subset["precipitation_sum"] = None
        forecast_subset["predicted_demand"] = pd.to_numeric(forecast_subset["predicted_demand"], errors="coerce")

        combined = pd.concat([historical_subset, forecast_subset], ignore_index=True)
        combined["time"] = combined["time"].astype(str)
        combined = combined.sort_values("time").reset_index(drop=True)

        def to_json_safe_value(value):
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if pd.isna(value):
                return None
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(numeric_value):
                return None
            return numeric_value

        records = []
        for _, row in combined.iterrows():
            records.append({
                "time": str(row["time"]),
                "units_sold": to_json_safe_value(row["units_sold"]),
                "predicted_demand": to_json_safe_value(row["predicted_demand"]),
                "precipitation_sum": to_json_safe_value(row["precipitation_sum"]),
            })

        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build forecast series: {str(e)}")

@app.get("/api/dashboard/forecast-summary", tags=["Data Delivery Endpoints"])
def get_forecast_summary():
    """
    Exposes aggregated target climate-aware demand predictions grouped per district node for high-level graph charts.
    """
    path = os.path.join(DATA_DIR, "upcoming_demand_forecasts.csv")
    if not os.path.exists(path):
        return {"message": "No predictive targets found.", "data": []}
    
    try:
        forecast_df = pd.read_csv(path)
        summary = forecast_df.groupby(['district', 'medicine'])['predicted_demand'].sum().reset_index()
        return {
            "data": summary.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate graph objects: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Launch uvicorn hot-reloads locally on localhost port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)