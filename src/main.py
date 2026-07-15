# src/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import sys

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
    path = "data/processed/optimized_transshipment_manifest.csv"
    if not os.path.exists(path):
        return {
            "message": "No active manifest dataset calculated yet. Execute /run-all first.", 
            "data": []
        }
    
    try:
        manifest_df = pd.read_csv(path)
        # Convert the dataframe to a clean JSON array structure for web rendering
        records = manifest_df.to_dict(orient="records")
        return {
            "count": len(records),
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read data matrix: {str(e)}")

@app.get("/api/dashboard/forecast-summary", tags=["Data Delivery Endpoints"])
def get_forecast_summary():
    """
    Exposes aggregated target climate-aware demand predictions grouped per district node for high-level graph charts.
    """
    path = "data/processed/upcoming_demand_forecasts.csv"
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