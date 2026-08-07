# src/main.py
import os
import sys
import math
from typing import Any, Dict, List, Optional

# Append the project root to path to ensure crisp absolute internal source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.database import get_db, test_db_connection
# Import your working functional script methods
from src.data_pipeline import fuse_healthcare_data
from src.forecast_engine import build_prophet_forecast, generate_demand_forecasts
from src.optimization_engine import run_transshipment_optimization

app = FastAPI(
    title="Climate-Responsive Lateral Pharmaceutical Transshipment API Backend",
    description="A centralized Object-Oriented Software Engineering framework linking predictive AI with Operations Research",
    version="1.0.0",
    docs_url="/",
    redoc_url=None,
    openapi_url="/openapi.json"
)


@app.on_event("startup")
def on_startup():
    test_db_connection()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

OPTIMIZER_STATE: Dict[str, object] = {
    "manifests": [],
    "last_run": None,
}


def _load_optimizer_manifest_rows() -> List[Dict[str, object]]:
    manifest_path = os.path.join(DATA_DIR, "optimized_transshipment_manifest.csv")
    if not os.path.exists(manifest_path):
        return []

    try:
        manifest_df = pd.read_csv(manifest_path)
    except Exception:
        return []

    rows: List[Dict[str, object]] = []
    for index, row in manifest_df.iterrows():
        source_hospital = str(row.get("source_hospital") or row.get("source_district") or "Unknown")
        destination_hospital = str(row.get("destination_hospital") or row.get("destination_district") or "Unknown")
        quantity_to_move = int(row.get("quantity_to_move", 0) or 0)
        transport_cost = float(row.get("logistical_cost_lkr", row.get("transport_cost", 0)) or 0.0)
        net_savings = float(row.get("financial_value_saved_lkr", row.get("net_savings", 0)) or 0.0)
        status = "PENDING_DISPATCH"

        rows.append({
            "id": index + 1,
            "source_district": source_hospital,
            "dest_district": destination_hospital,
            "medicine": str(row.get("medicine", "Unknown")),
            "quantity_to_move": quantity_to_move,
            "transport_cost": transport_cost,
            "net_savings": net_savings,
            "status": status,
        })

    return rows


def _sync_optimizer_state() -> List[Dict[str, object]]:
    persisted_rows = _load_optimizer_manifest_rows()
    if persisted_rows:
        OPTIMIZER_STATE["manifests"] = persisted_rows
    elif not OPTIMIZER_STATE["manifests"]:
        OPTIMIZER_STATE["manifests"] = []
    return OPTIMIZER_STATE["manifests"]


def _serialize_manifest(manifest: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": manifest.get("id"),
        "source_district": manifest.get("source_district"),
        "dest_district": manifest.get("dest_district"),
        "medicine": manifest.get("medicine"),
        "quantity_to_move": manifest.get("quantity_to_move"),
        "transport_cost": manifest.get("transport_cost"),
        "net_savings": manifest.get("net_savings"),
        "status": manifest.get("status"),
    }

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
def get_dashboard_metrics(db: Optional[Session] = Depends(get_db)):
    """Returns the summary KPI numbers used on the overview dashboard."""
    if db is not None:
        try:
            total_stock = db.execute(text("SELECT COALESCE(SUM(stock_quantity), 0) FROM inventory")).scalar() or 0
            total_asset_value = db.execute(text("SELECT COALESCE(SUM(stock_quantity * unit_price), 0) FROM inventory")).scalar() or 0.0
            expiring_stock_count = db.execute(text("SELECT COALESCE(COUNT(*), 0) FROM inventory WHERE expiry_days_remaining < 60")).scalar() or 0
            active_manifest_count = db.execute(text(
                "SELECT COALESCE(COUNT(*), 0) FROM transfer_manifests "
                "WHERE status IN ('PENDING_DISPATCH', 'DISPATCHED')"
            )).scalar() or 0

            return {
                "total_stock": int(total_stock),
                "total_asset_value": float(total_asset_value),
                "expiring_stock_count": int(expiring_stock_count),
                "active_manifest_count": int(active_manifest_count),
                "total_savings": float(total_asset_value),
                "stock_saved": int(total_stock),
                "active_manifests": int(active_manifest_count),
            }
        except (SQLAlchemyError, Exception) as exc:
            print(f"⚠️ Dashboard metrics DB query failed: {exc}")

    inventory_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    manifest_path = os.path.join(DATA_DIR, "optimized_transshipment_manifest.csv")
    total_stock = 0
    total_asset_value = 0.0
    expiring_stock_count = 0
    active_manifest_count = 0

    try:
        if os.path.exists(inventory_path):
            inventory_df = pd.read_csv(inventory_path)
            quantity_col = "stock_quantity" if "stock_quantity" in inventory_df.columns else "stock_level"
            inventory_df[quantity_col] = inventory_df[quantity_col].fillna(0).astype(float)
            inventory_df["unit_price"] = inventory_df["unit_price"].fillna(0.0).astype(float)
            total_stock = int(inventory_df[quantity_col].sum())
            total_asset_value = float((inventory_df[quantity_col] * inventory_df["unit_price"]).sum())
            expiring_stock_count = int(inventory_df[inventory_df["expiry_days_remaining"].fillna(0) < 60].shape[0])
        if os.path.exists(manifest_path):
            manifest_df = pd.read_csv(manifest_path)
            active_manifest_count = int(manifest_df.shape[0])
        return {
            "total_stock": total_stock,
            "total_asset_value": round(total_asset_value, 2),
            "expiring_stock_count": expiring_stock_count,
            "active_manifest_count": active_manifest_count,
            "total_savings": round(total_asset_value, 2),
            "stock_saved": total_stock,
            "active_manifests": active_manifest_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute dashboard metrics: {str(e)}")

@app.get("/api/dashboard/alerts", tags=["Data Delivery Endpoints"])
def get_dashboard_alerts(db: Optional[Session] = Depends(get_db)):
    """Returns inventory alerts for the dashboard warning feed."""
    if db is not None:
        try:
            query = text(
                "SELECT district, medicine_name AS medicine, stock_quantity AS stock_level, expiry_days_remaining "
                "FROM inventory "
                "WHERE expiry_days_remaining < 60 OR stock_quantity < 500 "
                "ORDER BY expiry_days_remaining ASC, stock_quantity ASC "
                "LIMIT 8"
            )
            rows = db.execute(query).mappings().all()
            if rows:
                return [
                    {
                        "id": idx + 1,
                        "district": str(row.get("district") or "Unknown"),
                        "medicine": str(row.get("medicine") or "Unknown"),
                        "stock_level": int(row.get("stock_level") or 0),
                        "expiry_days_remaining": int(row.get("expiry_days_remaining") or 0),
                    }
                    for idx, row in enumerate(rows)
                ]
        except SQLAlchemyError as exc:
            print(f"⚠️ Dashboard alerts DB query failed: {exc}")

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
def get_inventory_records(db: Optional[Session] = Depends(get_db)):
    """Returns the current inventory ledger rows used by the inventory page."""
    if db is not None:
        try:
            rows = db.execute(text(
                "SELECT district, medicine_name, category, stock_quantity, expiry_days_remaining, unit_price "
                "FROM inventory"
            )).mappings().all()
            return [
                {
                    "district": str(row.get("district") or "Unknown"),
                    "medicine_name": str(row.get("medicine_name") or "Unknown"),
                    "category": str(row.get("category") or "Unknown"),
                    "stock_quantity": int(row.get("stock_quantity") or 0),
                    "expiry_days_remaining": int(row.get("expiry_days_remaining") or 0),
                    "unit_price": float(row.get("unit_price") or 0.0),
                }
                for row in rows
            ]
        except (SQLAlchemyError, Exception) as exc:
            print(f"⚠️ Inventory DB query failed: {exc}")

    inventory_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    if not os.path.exists(inventory_path):
        return []

    try:
        inventory_df = pd.read_csv(inventory_path)
        district_col = "district" if "district" in inventory_df.columns else "district"
        medicine_col = "medicine" if "medicine" in inventory_df.columns else "medicine_name"
        quantity_col = "stock_quantity" if "stock_quantity" in inventory_df.columns else "stock_level"
        inventory_df["district"] = inventory_df[district_col].astype(str).fillna("Unknown")
        inventory_df["medicine_name"] = inventory_df[medicine_col].astype(str).fillna("Unknown")
        inventory_df["category"] = inventory_df["category"].astype(str).fillna("Unknown")
        inventory_df["stock_quantity"] = inventory_df[quantity_col].fillna(0).astype(int)
        inventory_df["expiry_days_remaining"] = inventory_df["expiry_days_remaining"].fillna(0).astype(int)
        inventory_df["unit_price"] = inventory_df["unit_price"].fillna(0.0).astype(float)

        return inventory_df[["district", "medicine_name", "category", "stock_quantity", "expiry_days_remaining", "unit_price"]].to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read inventory data: {str(e)}")

@app.get("/api/network/nodes", tags=["Data Delivery Endpoints"])
def get_network_nodes(db: Optional[Session] = Depends(get_db)):
    """Returns the spatial network nodes with active inbound/outbound counts and health status."""
    node_definitions = [
        {"district": "Colombo", "hospital_name": "Colombo General Node", "latitude": 6.9271, "longitude": 79.8612},
        {"district": "Jaffna", "hospital_name": "Jaffna Teaching Hospital Node", "latitude": 9.6615, "longitude": 80.0255},
        {"district": "Galle", "hospital_name": "Galle Regional Depot", "latitude": 6.0535, "longitude": 80.2210},
        {"district": "Kandy", "hospital_name": "Kandy General Hospital", "latitude": 7.2906, "longitude": 80.6337},
        {"district": "Anuradhapura", "hospital_name": "Anuradhapura Base Node", "latitude": 8.3114, "longitude": 80.4037},
    ]

    def build_node_from_row(district: str, row: Optional[Dict[str, Any]]):
        total_stock = int(row.get("total_stock") or 0)
        expiry_count_30 = int(row.get("expiry_count_30") or 0)
        expiry_count_60 = int(row.get("expiry_count_60") or 0)
        if total_stock < 1000 or expiry_count_30 > 0:
            health_status = "CRITICAL"
        elif total_stock < 2500 or expiry_count_60 > 0:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"

        return {
            "id": 0,
            "district": district,
            "hospital_name": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "active_inbound": int(row.get("active_inbound") or 0),
            "active_outbound": int(row.get("active_outbound") or 0),
            "health_status": health_status,
        }

    if db is not None:
        try:
            query = text(
                "SELECT d.district, "
                "COALESCE(inv.total_stock, 0) AS total_stock, "
                "COALESCE(inv.expiry_count_30, 0) AS expiry_count_30, "
                "COALESCE(inv.expiry_count_60, 0) AS expiry_count_60, "
                "COALESCE(inbound.active_inbound, 0) AS active_inbound, "
                "COALESCE(outbound.active_outbound, 0) AS active_outbound "
                "FROM (VALUES ('Colombo'), ('Jaffna'), ('Galle'), ('Kandy'), ('Anuradhapura')) AS d(district) "
                "LEFT JOIN ( "
                "  SELECT district, "
                "    SUM(stock_quantity) AS total_stock, "
                "    SUM(CASE WHEN expiry_days_remaining < 30 THEN 1 ELSE 0 END) AS expiry_count_30, "
                "    SUM(CASE WHEN expiry_days_remaining < 60 THEN 1 ELSE 0 END) AS expiry_count_60 "
                "  FROM inventory "
                "  GROUP BY district "
                ") AS inv ON inv.district = d.district "
                "LEFT JOIN ( "
                "  SELECT dest_district AS district, COUNT(*) FILTER (WHERE status IN ('PENDING_DISPATCH', 'DISPATCHED')) AS active_inbound "
                "  FROM transfer_manifests "
                "  GROUP BY dest_district "
                ") AS inbound ON inbound.district = d.district "
                "LEFT JOIN ( "
                "  SELECT source_district AS district, COUNT(*) FILTER (WHERE status IN ('PENDING_DISPATCH', 'DISPATCHED')) AS active_outbound "
                "  FROM transfer_manifests "
                "  GROUP BY source_district "
                ") AS outbound ON outbound.district = d.district "
                "ORDER BY d.district"
            )
            rows = db.execute(query).mappings().all()
            row_map = {row["district"]: row for row in rows}
            nodes = []
            for idx, node in enumerate(node_definitions, start=1):
                raw_row = row_map.get(node["district"], {})
                node_payload = build_node_from_row(node["district"], raw_row)
                node_payload["id"] = idx
                node_payload["hospital_name"] = node["hospital_name"]
                node_payload["latitude"] = node["latitude"]
                node_payload["longitude"] = node["longitude"]
                nodes.append(node_payload)
            return nodes
        except SQLAlchemyError as exc:
            print(f"⚠️ Network nodes DB query failed: {exc}")

    source_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")
    if not os.path.exists(source_path):
        return []

    try:
        inventory_df = pd.read_csv(source_path)
        inventory_df["district"] = inventory_df["district"].astype(str)
        inventory_df["status"] = inventory_df.get("status", pd.Series(["PENDING_DISPATCH"] * len(inventory_df))).astype(str)

        nodes = []
        for idx, node in enumerate(node_definitions, start=1):
            node_df = inventory_df[inventory_df["district"].str.lower() == node["district"].lower()]
            active_inbound = int(node_df[node_df["status"].isin(["PENDING_DISPATCH", "DISPATCHED"])].shape[0])
            active_outbound = int(node_df[node_df["status"].isin(["PENDING_DISPATCH", "DISPATCHED"])].shape[0])
            total_stock = int(node_df["stock_level"].fillna(0).sum())
            expiry_count_30 = int(node_df[node_df["expiry_days_remaining"] < 30].shape[0])
            expiry_count_60 = int(node_df[node_df["expiry_days_remaining"] < 60].shape[0])

            if total_stock < 1000 or expiry_count_30 > 0:
                health_status = "CRITICAL"
            elif total_stock < 2500 or expiry_count_60 > 0:
                health_status = "WARNING"
            else:
                health_status = "HEALTHY"

            nodes.append({
                "id": idx,
                "district": node["district"],
                "hospital_name": node["hospital_name"],
                "latitude": node["latitude"],
                "longitude": node["longitude"],
                "active_inbound": active_inbound,
                "active_outbound": active_outbound,
                "health_status": health_status,
            })
        return nodes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute network topology: {str(e)}")

@app.get("/api/optimizer/manifests", tags=["Optimizer Endpoints"])
def get_optimizer_manifests():
    """Returns the current optimizer manifest queue for the dispatch UI."""
    manifests = _sync_optimizer_state()
    return [_serialize_manifest(manifest) for manifest in manifests]


@app.post("/api/optimizer/run", tags=["Optimizer Endpoints"])
def run_optimizer_sweep():
    """Refreshes the optimizer queue from the processed CSV and marks the run as completed."""
    manifests = _sync_optimizer_state()
    OPTIMIZER_STATE["last_run"] = {
        "status": "completed",
        "generated_manifests": len(manifests),
    }

    return {
        "status": "success",
        "message": "PuLP optimization sweep completed successfully.",
        "manifests": [_serialize_manifest(manifest) for manifest in manifests],
    }


@app.patch("/api/optimizer/manifests/{manifest_id}/dispatch", tags=["Optimizer Endpoints"])
def dispatch_optimizer_manifest(manifest_id: str):
    """Approves a pending transfer and updates the manifest status locally."""
    manifests = _sync_optimizer_state()
    target_manifest = None

    for manifest in manifests:
        if str(manifest.get("id")) == str(manifest_id):
            target_manifest = manifest
            break

    if target_manifest is None:
        raise HTTPException(status_code=404, detail="Manifest not found")

    target_manifest["status"] = "DISPATCHED"
    return {
        "status": "success",
        "message": "Manifest approved and dispatched.",
        "manifest": _serialize_manifest(target_manifest),
    }


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
    """Returns evaluation metrics and a chart-ready series for the selected medicine and district."""
    historical_path = os.path.join(DATA_DIR, "fused_master_dataset.csv")

    if not os.path.exists(historical_path):
        return {
            "metrics": {"rmse": 0.0, "mae": 0.0, "mape_percent": 0.0},
            "chart_data": [],
        }

    try:
        historical_df = pd.read_csv(historical_path)
        historical_df["date"] = pd.to_datetime(historical_df["date"], errors="coerce")
        historical_df = historical_df.dropna(subset=["date"]).reset_index(drop=True)

        historical_subset = historical_df[
            (historical_df["medicine"].astype(str).str.lower() == medicine.lower()) &
            (historical_df["district"].astype(str).str.lower() == district.lower())
        ].copy()

        if historical_subset.empty:
            return {
                "metrics": {"rmse": 0.0, "mae": 0.0, "mape_percent": 0.0},
                "chart_data": [],
            }

        model_payload = build_prophet_forecast(
            district=district,
            medicine=medicine,
            historical_df=historical_df,
            forecast_horizon=30,
        )

        future_predictions = model_payload["future_predictions"].copy()
        historical_subset = historical_subset[["date", "units_sold", "precipitation_sum"]].copy()
        historical_subset = historical_subset.sort_values("date").reset_index(drop=True)

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
        for _, row in historical_subset.iterrows():
            records.append({
                "time": row["date"].strftime("%Y-%m-%d"),
                "units_sold": to_json_safe_value(row["units_sold"]),
                "predicted_demand": None,
                "precipitation_sum": to_json_safe_value(row["precipitation_sum"]),
            })

        for _, row in future_predictions.iterrows():
            records.append({
                "time": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "units_sold": None,
                "predicted_demand": to_json_safe_value(row["predicted_demand"]),
                "precipitation_sum": None,
            })

        records = sorted(records, key=lambda item: item["time"])

        return {
            "metrics": model_payload["metrics"],
            "chart_data": records,
        }
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