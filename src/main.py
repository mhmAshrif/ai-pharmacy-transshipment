# src/main.py
import os
import sys
import math
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Append the project root to path to ensure crisp absolute internal source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, func, or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.database import get_db, test_db_connection, init_db, seed_initial_inventory, ensure_reference_lookup_tables, cleanup_duplicate_transfer_manifests, backfill_inventory_foreign_keys, SessionLocal
from src.models import Inventory, TransferManifest, ForecastMetric, District, Medicine, AuditLog
from sqlalchemy import text
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
    ok = test_db_connection()
    if ok:
        # Ensure DB schema exists and seed inventory from CSV if empty
        init_db()
        try:
            if SessionLocal is not None:
                db = SessionLocal()
                try:
                    inserted = seed_initial_inventory(db)
                    if inserted:
                        print(f"Seeded inventory with {inserted} rows from CSV.")
                finally:
                    db.close()
        except Exception as exc:
            print(f"Warning: seeding initial inventory failed: {exc}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

PIPELINE_LOCK = threading.Lock()
OPTIMIZER_STATE: Dict[str, object] = {
    "manifests": [],
    "last_run": None,
    "running": False,
}


def _load_optimizer_manifest_rows(db: Optional[Session] = None) -> List[Dict[str, object]]:
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
        expiring_asset_value = float(row.get("financial_value_saved_lkr", row.get("expiring_asset_value", 0)) or 0.0)
        net_savings = float(row.get("net_savings", 0) or 0.0)
        status = "PENDING_DISPATCH"
        database_id = None

        if db is not None:
            persisted = db.execute(
                text("""
                    SELECT tm.id, tm.status
                    FROM public.transfer_manifests tm
                    JOIN public.districts sd ON sd.id = tm.source_district_id
                    JOIN public.districts dd ON dd.id = tm.dest_district_id
                    JOIN public.medicines m ON m.id = tm.medicine_id
                    WHERE LOWER(sd.name) = LOWER(:source)
                      AND LOWER(dd.name) = LOWER(:destination)
                      AND LOWER(m.name) = LOWER(:medicine)
                      AND tm.quantity_to_move = :quantity
                    ORDER BY tm.id DESC
                    LIMIT 1
                """),
                {
                    "source": source_hospital,
                    "destination": destination_hospital,
                    "medicine": str(row.get("medicine", "Unknown")),
                    "quantity": quantity_to_move,
                },
            ).first()
            if persisted:
                database_id, persisted_status = persisted
                status = str(persisted_status)
            else:
                database_id = None

        rows.append({
            "id": index + 1,
            "database_id": database_id,
            "source_district": source_hospital,
            "dest_district": destination_hospital,
            "medicine": str(row.get("medicine", "Unknown")),
            "quantity_to_move": quantity_to_move,
            "transport_cost": transport_cost,
            "expiring_asset_value": expiring_asset_value,
            "net_savings": net_savings,
            "status": status,
        })

    return rows


def _sync_optimizer_state(db: Optional[Session] = None) -> List[Dict[str, object]]:
    persisted_rows = _load_optimizer_manifest_rows(db)
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
        "expiring_asset_value": manifest.get("expiring_asset_value"),
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
def execute_system_pipeline(db: Optional[Session] = Depends(get_db)):
    """
    Executes Phase 1, Phase 2, and Phase 3 of the framework consecutively.
    Fuses environmental weather data, generates Facebook Prophet forecasts, and runs PuLP optimization.
    """
    if OPTIMIZER_STATE.get("running"):
        return {
            "status": "Busy",
            "message": "Pipeline already running. Please wait for the current execution to finish.",
            "total_routes_generated": len(OPTIMIZER_STATE.get("manifests", [])),
            "total_net_savings_lkr": 0.0,
            "rejected_unprofitable_route_count": 0,
        }

    with PIPELINE_LOCK:
        if OPTIMIZER_STATE.get("running"):
            return {
                "status": "Busy",
                "message": "Pipeline already running. Please wait for the current execution to finish.",
                "total_routes_generated": len(OPTIMIZER_STATE.get("manifests", [])),
                "total_net_savings_lkr": 0.0,
                "rejected_unprofitable_route_count": 0,
            }

        OPTIMIZER_STATE["running"] = True
        try:
            print("\n--- Triggering Full API-Driven Orchestration Chain ---")
            # 1. Run Data Preprocessing and Fusion
            fuse_healthcare_data()

            # 2. Trigger Facebook Prophet Forecasting Brain
            generate_demand_forecasts()

            # 2.b Persist forecast evaluation metrics into DB if available
            try:
                metrics_path = os.path.join(DATA_DIR, "forecast_evaluation_metrics.csv")
                if db is not None and os.path.exists(metrics_path):
                    import pandas as pd

                    metrics_df = pd.read_csv(metrics_path)
                    metric_objs = []
                    for _, row in metrics_df.iterrows():
                        try:
                            metric_objs.append(
                                ForecastMetric(
                                    district=str(row.get("district") or "").strip(),
                                    medicine_name=str(row.get("medicine") or "").strip(),
                                    rmse=float(row.get("rmse") or 0.0),
                                    mae=float(row.get("mae") or 0.0),
                                    mape=float(row.get("mape") or 0.0),
                                    best_changepoint_prior=float(row.get("best_changepoint_prior") or 0.0),
                                    best_seasonality_prior=float(row.get("best_seasonality_prior") or 0.0),
                                    evaluated_at=datetime.utcnow(),
                                )
                            )
                        except Exception:
                            continue
                    if metric_objs:
                        try:
                            db.bulk_save_objects(metric_objs)
                            db.commit()
                            print(f"Persisted {len(metric_objs)} forecast metric rows to DB.")
                        except Exception as exc:
                            db.rollback()
                            print(f"Warning: failed to persist forecast metrics: {exc}")
            except Exception as exc:
                print(f"Warning: forecast metrics persistence step failed: {exc}")

            # 3. Compute cost-optimal routing manifests via Linear Programming
            optimization_result = run_transshipment_optimization()
            optimized_manifest = optimization_result.get("manifest", [])
            rejected_candidates = optimization_result.get("rejected_candidates", [])

            audit_entries = []
            approvals = []
            rejections = []

            if db is not None:
                try:
                    ensure_reference_lookup_tables(db)
                    backfill_inventory_foreign_keys(db)
                    cleanup_duplicate_transfer_manifests(db)
                    # Clear existing PENDING_DISPATCH manifests to avoid duplication and ID jumping across runs
                    deleted_count = db.execute(text("DELETE FROM public.transfer_manifests WHERE status = 'PENDING_DISPATCH'")).rowcount
                    if deleted_count > 0:
                        print(f"Cleared {deleted_count} stale PENDING_DISPATCH manifests before inserting new routes.")
                    db.commit()
                    db.execute(text("DELETE FROM public.transfer_manifests WHERE source_district_id IS NULL OR dest_district_id IS NULL OR medicine_id IS NULL"))
                    db.commit()

                    # Build case-insensitive lookup maps for districts and medicines using the
                    # fully backfilled reference tables and the inventory-sourced FK map.
                    district_rows = db.query(District).all()
                    district_map = {}
                    for d in district_rows:
                        name_val = getattr(d, "name", None) or getattr(d, "district", "")
                        if name_val:
                            district_map[name_val.strip().lower()] = d.id

                    medicine_rows = db.query(Medicine).all()
                    medicine_map = {}
                    for m in medicine_rows:
                        name_val = getattr(m, "name", None) or getattr(m, "medicine", "")
                        if name_val:
                            medicine_map[name_val.strip().lower()] = m.id

                    inventory_id_map = db.execute(text(
                        "SELECT DISTINCT district, district_id, medicine, medicine_id FROM public.inventory WHERE district IS NOT NULL AND medicine IS NOT NULL"
                    )).mappings().all()
                    for row in inventory_id_map:
                        district_name = (row.get("district") or "").strip().lower()
                        medicine_name = (row.get("medicine") or "").strip().lower()
                        if district_name and row.get("district_id") is not None:
                            district_map.setdefault(district_name, row["district_id"])
                        if medicine_name and row.get("medicine_id") is not None:
                            medicine_map.setdefault(medicine_name, row["medicine_id"])

                    existing_manifest_keys = set(db.execute(text(
                        "SELECT source_district_id, dest_district_id, medicine_id, quantity_to_move FROM public.transfer_manifests WHERE source_district_id IS NOT NULL AND dest_district_id IS NOT NULL AND medicine_id IS NOT NULL"
                    )).fetchall())

                    # Prepare ORM objects for a single atomic insert, and skip duplicates/unresolved rows.
                    seen_keys = set(existing_manifest_keys)
                    manifest_objs = []
                    for r in optimized_manifest:
                        med_name = (r.get("medicine") or r.get("medicine_name") or "").strip()
                        src_name = (r.get("source_district") or r.get("source_hospital") or "").strip()
                        dst_name = (r.get("dest_district") or r.get("destination_hospital") or "").strip()
                        qty = int(r.get("quantity_to_move", 0) or 0)
                        transport_cost = float(r.get("transport_cost", r.get("logistical_cost_lkr", 0)) or 0.0)
                        net_savings = float(r.get("net_savings", 0) or 0.0)
                        unit_price = float(r.get("unit_price_lkr", 0) or 0.0)

                        src_id = district_map.get(src_name.lower())
                        dst_id = district_map.get(dst_name.lower())
                        med_id = medicine_map.get(med_name.lower())

                        if src_id is None or dst_id is None or med_id is None:
                            print(f"Skipping manifest row due to unresolved FK: medicine={med_name!r}, source={src_name!r}, destination={dst_name!r}")
                            continue

                        key = (src_id, dst_id, med_id, qty)
                        if key in seen_keys:
                            print(f"Skipping duplicate manifest key: {key}")
                            continue
                        seen_keys.add(key)

                        manifest_objs.append(
                            TransferManifest(
                                source_district_id=src_id,
                                dest_district_id=dst_id,
                                medicine_id=med_id,
                                quantity_to_move=qty,
                                transport_cost=transport_cost,
                                expiring_asset_value=qty * unit_price,
                                net_savings=net_savings,
                                status=r.get("status", "PENDING_DISPATCH"),
                            )
                        )

                    # Atomic bulk insert
                    try:
                        if manifest_objs:
                            db.add_all(manifest_objs)
                            db.commit()
                            # Build approvals list with persisted IDs
                            for obj in manifest_objs:
                                approvals.append(
                                    {
                                        "manifest_id": getattr(obj, "id", None),
                                        "source_district_id": getattr(obj, "source_district_id", None),
                                        "dest_district_id": getattr(obj, "dest_district_id", None),
                                        "medicine_id": getattr(obj, "medicine_id", None),
                                        "source_district": None,
                                        "destination_district": None,
                                        "medicine_name": None,
                                        "quantity": getattr(obj, "quantity_to_move", 0),
                                        "transport_cost": getattr(obj, "transport_cost", 0.0),
                                        "net_savings": getattr(obj, "net_savings", 0.0),
                                    }
                                )
                    except Exception as e:
                        db.rollback()
                        print(f"Manifest persistence error: {e}")

                    # Persist rejected candidates as audit logs
                    # Persist rejected candidates as audit logs in a single batch to avoid per-row commits
                    try:
                        audit_objs = []
                        for rc in rejected_candidates:
                            audit_objs.append(
                                AuditLog(
                                    action="REJECT_ROUTE",
                                    entity_type="transfer_candidate",
                                    entity_id=None,
                                    user_id=None,
                                    details=rc,
                                )
                            )
                            rejections.append(rc)

                        if audit_objs:
                            db.bulk_save_objects(audit_objs)
                            db.commit()
                    except Exception:
                        db.rollback()
                except Exception as exc:
                    print(f"Warning: manifest persistence/audit failed: {exc}")

            OPTIMIZER_STATE["manifests"] = optimized_manifest
            OPTIMIZER_STATE["last_run"] = time.time()
            total_net_savings = optimization_result.get("total_net_savings_lkr", 0.0)
            rejected_routes = optimization_result.get("rejected_unprofitable_route_count", 0)

            return {
                "status": "Success",
                "message": "Entire automated forecasting and redistribution optimization cycle completed.",
                "total_routes_generated": len(optimized_manifest),
                "total_net_savings_lkr": total_net_savings,
                "rejected_unprofitable_route_count": rejected_routes,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline Orchestration Failed: {str(e)}")
        finally:
            OPTIMIZER_STATE["running"] = False

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
        records = []
        for _, row in manifest_df.iterrows():
            record = row.to_dict()
            record["expiring_asset_value"] = float(
                row.get("financial_value_saved_lkr", row.get("expiring_asset_value", 0)) or 0.0
            )
            record["net_savings"] = float(row.get("net_savings", 0) or 0.0)
            records.append(record)
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
            total_stock = db.query(func.coalesce(func.sum(Inventory.stock_quantity), 0)).scalar() or 0
            total_asset_value = db.query(func.coalesce(func.sum(Inventory.stock_quantity * Inventory.unit_price), 0)).scalar() or 0.0
            expiring_stock_count = db.query(func.count()).filter(Inventory.expiry_days_remaining < 60).scalar() or 0
            active_manifest_count = db.query(func.count()).select_from(TransferManifest).filter(TransferManifest.status.in_(("PENDING_DISPATCH", "DISPATCHED"))).scalar() or 0

            return {
                "total_stock": int(total_stock),
                "total_asset_value": float(total_asset_value),
                "expiring_stock_count": int(expiring_stock_count),
                "active_manifest_count": int(active_manifest_count),
                "total_savings": float(
                    db.query(func.coalesce(func.sum(TransferManifest.net_savings), 0)).scalar() or 0.0
                ),
                "stock_saved": int(total_stock),
                "active_manifests": int(active_manifest_count),
            }
        except (SQLAlchemyError, Exception) as exc:
            print(f"⚠️ Dashboard metrics DB query failed, falling back to CSV: {exc}")

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
            total_net_savings = float(manifest_df.get("net_savings", pd.Series(dtype=float)).fillna(0).sum())
        else:
            total_net_savings = 0.0
        return {
            "total_stock": total_stock,
            "total_asset_value": round(total_asset_value, 2),
            "expiring_stock_count": expiring_stock_count,
            "active_manifest_count": active_manifest_count,
            "total_savings": round(total_net_savings, 2),
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
            rows = (
                db.query(Inventory)
                .filter(or_(Inventory.expiry_days_remaining < 60, Inventory.stock_quantity < 500))
                .order_by(Inventory.expiry_days_remaining.asc(), Inventory.stock_quantity.asc())
                .limit(8)
                .all()
            )
            if rows:
                return [
                    {
                        "id": idx + 1,
                        "district": getattr(r, "district", "Unknown"),
                        "medicine": getattr(r, "medicine_name", "Unknown"),
                        "stock_level": int(getattr(r, "stock_quantity", 0) or 0),
                        "expiry_days_remaining": int(getattr(r, "expiry_days_remaining", 0) or 0),
                    }
                    for idx, r in enumerate(rows)
                ]
        except SQLAlchemyError as exc:
            print(f"⚠️ Dashboard alerts DB query failed, falling back to CSV: {exc}")

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
def get_optimizer_manifests(db: Optional[Session] = Depends(get_db)):
    """Returns the current optimizer manifest queue for the dispatch UI."""
    manifests = _sync_optimizer_state(db)
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
def dispatch_optimizer_manifest(manifest_id: str, db: Optional[Session] = Depends(get_db)):
    """Approve a transfer, persist its status, and record an immutable audit event."""
    manifests = _sync_optimizer_state(db)
    target_manifest = None

    for manifest in manifests:
        if str(manifest.get("id")) == str(manifest_id):
            target_manifest = manifest
            break

    if target_manifest is None:
        raise HTTPException(status_code=404, detail="Manifest not found")

    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable; dispatch cannot be persisted")

    try:
        persisted_id = target_manifest.get("database_id")
        if persisted_id is None:
            persisted_id = db.execute(
                text("""
                    SELECT tm.id
                    FROM public.transfer_manifests tm
                    JOIN public.districts sd ON sd.id = tm.source_district_id
                    JOIN public.districts dd ON dd.id = tm.dest_district_id
                    JOIN public.medicines m ON m.id = tm.medicine_id
                    WHERE LOWER(sd.name) = LOWER(:source)
                      AND LOWER(dd.name) = LOWER(:destination)
                      AND LOWER(m.name) = LOWER(:medicine)
                      AND tm.quantity_to_move = :quantity
                    ORDER BY tm.id DESC
                    LIMIT 1
                """),
                {
                    "source": target_manifest.get("source_district"),
                    "destination": target_manifest.get("dest_district"),
                    "medicine": target_manifest.get("medicine"),
                    "quantity": int(target_manifest.get("quantity_to_move") or 0),
                },
            ).scalar()
        if persisted_id is None:
            raise HTTPException(status_code=404, detail="Persisted manifest not found")

        db.execute(
            text("UPDATE public.transfer_manifests SET status = 'DISPATCHED' WHERE id = :manifest_id"),
            {"manifest_id": persisted_id},
        )
        db.add(
            AuditLog(
                action="DISPATCH_APPROVED",
                entity_type="transfer_manifest",
                entity_id=int(persisted_id),
                user_id="Hospital Admin",
                details={
                    "manifest_id": int(persisted_id),
                    "performed_by": "Hospital Admin",
                    "timestamp": datetime.utcnow().isoformat(),
                },
                timestamp=datetime.utcnow(),
            )
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to persist dispatch approval: {exc}")

    target_manifest["status"] = "DISPATCHED"
    return {
        "status": "success",
        "message": "Manifest approved and dispatched.",
        "manifest": _serialize_manifest(target_manifest),
    }


@app.patch("/api/optimizer/manifests/{manifest_id}/undo", tags=["Optimizer Endpoints"])
def undo_optimizer_manifest(manifest_id: str, db: Optional[Session] = Depends(get_db)):
    """Revert a dispatched transfer to pending and record the undo event."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable; dispatch cannot be reverted")

    manifests = _sync_optimizer_state(db)
    target_manifest = next(
        (manifest for manifest in manifests if str(manifest.get("id")) == str(manifest_id)),
        None,
    )
    if target_manifest is None:
        raise HTTPException(status_code=404, detail="Manifest not found")

    try:
        persisted_id = target_manifest.get("database_id")
        if persisted_id is None:
            persisted_id = db.execute(
                text("""
                    SELECT tm.id
                    FROM public.transfer_manifests tm
                    JOIN public.districts sd ON sd.id = tm.source_district_id
                    JOIN public.districts dd ON dd.id = tm.dest_district_id
                    JOIN public.medicines m ON m.id = tm.medicine_id
                    WHERE LOWER(sd.name) = LOWER(:source)
                      AND LOWER(dd.name) = LOWER(:destination)
                      AND LOWER(m.name) = LOWER(:medicine)
                      AND tm.quantity_to_move = :quantity
                    ORDER BY tm.id DESC
                    LIMIT 1
                """),
                {
                    "source": target_manifest.get("source_district"),
                    "destination": target_manifest.get("dest_district"),
                    "medicine": target_manifest.get("medicine"),
                    "quantity": int(target_manifest.get("quantity_to_move") or 0),
                },
            ).scalar()
        if persisted_id is None:
            raise HTTPException(status_code=404, detail="Persisted manifest not found")

        database_manifest_id = persisted_id
        current_status = db.execute(
            text("SELECT status FROM public.transfer_manifests WHERE id = :manifest_id"),
            {"manifest_id": database_manifest_id},
        ).scalar()
        if current_status != "DISPATCHED":
            raise HTTPException(status_code=400, detail="Only dispatched manifests can be reverted")

        db.execute(
            text("UPDATE public.transfer_manifests SET status = 'PENDING_DISPATCH' WHERE id = :manifest_id"),
            {"manifest_id": database_manifest_id},
        )
        audit_timestamp = datetime.utcnow()
        db.add(
            AuditLog(
                action="DISPATCH_REVERTED_UNDO",
                entity_type="transfer_manifest",
                entity_id=int(database_manifest_id),
                user_id="Hospital Admin",
                details={
                    "manifest_id": int(database_manifest_id),
                    "performed_by": "Hospital Admin",
                    "timestamp": audit_timestamp.isoformat(),
                },
                timestamp=audit_timestamp,
            )
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to revert dispatch: {exc}")

    target_manifest["status"] = "PENDING_DISPATCH"
    return {
        "status": "success",
        "message": "Transfer reverted to Pending",
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

        forecast_path = os.path.join(DATA_DIR, "upcoming_demand_forecasts.csv")
        metrics_path = os.path.join(DATA_DIR, "forecast_evaluation_metrics.csv")
        future_predictions = pd.DataFrame(columns=["date", "predicted_demand"])
        model_metrics = {"rmse": 0.0, "mae": 0.0, "mape": 0.0}

        if os.path.exists(forecast_path):
            cached_forecasts = pd.read_csv(forecast_path)
            cached_forecasts["date"] = pd.to_datetime(cached_forecasts["date"], errors="coerce")
            future_predictions = cached_forecasts[
                (cached_forecasts["district"].astype(str).str.lower() == district.lower()) &
                (cached_forecasts["medicine"].astype(str).str.lower() == medicine.lower())
            ][["date", "predicted_demand"]].dropna(subset=["date"])

        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            metric_rows = metrics_df[
                (metrics_df["district"].astype(str).str.lower() == district.lower()) &
                (metrics_df["medicine"].astype(str).str.lower() == medicine.lower())
            ]
            if not metric_rows.empty:
                metric_row = metric_rows.iloc[-1]
                model_metrics = {
                    "rmse": float(metric_row.get("rmse", 0.0) or 0.0),
                    "mae": float(metric_row.get("mae", 0.0) or 0.0),
                    "mape": float(metric_row.get("mape", 0.0) or 0.0),
                }

        if future_predictions.empty:
            model_payload = build_prophet_forecast(
                district=district,
                medicine=medicine,
                historical_df=historical_df,
                forecast_horizon=30,
            )
            future_predictions = model_payload["future_predictions"].copy()
            model_metrics = model_payload["metrics"]

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
            "metrics": {
                "rmse": model_metrics.get("rmse"),
                "mae": model_metrics.get("mae"),
                "mape_percent": model_metrics.get("mape"),
            },
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