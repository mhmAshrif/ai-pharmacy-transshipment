# src/optimization_engine.py
import os
import warnings
from typing import Dict, List

import pandas as pd
import pulp

warnings.filterwarnings("ignore")


def run_transshipment_optimization():
    sales_path = "data/processed/fused_master_dataset.csv"
    forecast_path = "data/processed/upcoming_demand_forecasts.csv"
    output_dir = "data/processed"

    if not os.path.exists(sales_path) or not os.path.exists(forecast_path):
        raise FileNotFoundError("Pipeline broken. Ensure data_pipeline.py and forecast_engine.py have been executed.")

    print("Ingesting current inventory states and AI demand predictions...")
    sales_df = pd.read_csv(sales_path)
    forecast_df = pd.read_csv(forecast_path)

    sales_df["medicine"] = sales_df["medicine"].astype(str).str.strip()
    forecast_df["medicine"] = forecast_df["medicine"].astype(str).str.strip()

    print("Compiling inventory baseline tracking tables (using most recent stock snapshot per node)...")
    # Use the most recent observed stock level per (district, medicine) to avoid mean-inflation
    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    sales_df = sales_df.sort_values("date")
    # Keep last record per group (most recent snapshot)
    current_inventory = (
        sales_df.groupby(["district", "medicine"], as_index=False)
        .last()[["district", "medicine", "stock_level", "unit_price", "expiry_days_remaining"]]
        .rename(columns={"stock_level": "stock_level", "unit_price": "unit_price", "expiry_days_remaining": "expiry_days_remaining"})
        .reset_index(drop=True)
    )

    demand_summary = (
        forecast_df.groupby(["district", "medicine"], as_index=False)["predicted_demand"]
        .sum()
    )

    cost_matrix = {
        "Colombo": {"Colombo": 0, "Kandy": 120, "Galle": 100, "Anuradhapura": 200, "Jaffna": 350},
        "Kandy": {"Colombo": 120, "Kandy": 0, "Galle": 220, "Anuradhapura": 140, "Jaffna": 300},
        "Galle": {"Colombo": 100, "Kandy": 220, "Galle": 0, "Anuradhapura": 280, "Jaffna": 420},
        "Anuradhapura": {"Colombo": 200, "Kandy": 140, "Galle": 280, "Anuradhapura": 0, "Jaffna": 180},
        "Jaffna": {"Colombo": 350, "Kandy": 300, "Galle": 420, "Anuradhapura": 180, "Jaffna": 0},
    }

    nodes = list(cost_matrix.keys())
    unique_medicines = sorted(current_inventory["medicine"].dropna().astype(str).unique())

    transfer_manifest_records: List[Dict[str, object]] = []
    rejected_candidates: List[Dict[str, object]] = []

    # Bulk logistics discount multiplier (per-unit scaling factor against distance cost)
    # Using a mid-range factor to favor bulk crate economics while staying conservative
    # Allow runtime overrides via environment variables for safer experimentation
    try:
        batch_discount = float(os.getenv("BATCH_DISCOUNT", "0.07"))
    except Exception:
        batch_discount = 0.07

    print(f"Initializing optimization constraints across {len(unique_medicines)} active pharmaceutical lines...")

    rejected_unprofitable_route_count = 0

    # Parameters (configurable via environment variables)
    try:
        safety_buffer = float(os.getenv("SAFETY_BUFFER", "500"))
    except Exception:
        safety_buffer = 500.0
    try:
        expiry_cutoff_days = float(os.getenv("EXPIRY_CUTOFF_DAYS", "90"))
    except Exception:
        expiry_cutoff_days = 90.0
    try:
        profit_eps = float(os.getenv("PROFIT_EPS", "0.01"))
    except Exception:
        profit_eps = 0.01

    for medicine in unique_medicines:
        # Build per-medicine supply/demand matrices
        inventory_lookup: Dict[str, Dict[str, float]] = {}
        deficit: Dict[str, float] = {}
        surplus: Dict[str, float] = {}

        for node in nodes:
            stock_row = current_inventory[(current_inventory["district"] == node) & (current_inventory["medicine"] == medicine)]
            demand_row = demand_summary[(demand_summary["district"] == node) & (demand_summary["medicine"] == medicine)]

            stock_level = float(stock_row["stock_level"].iloc[0]) if not stock_row.empty else 0.0
            unit_price = float(stock_row["unit_price"].iloc[0]) if not stock_row.empty else 0.0
            expiry_days = float(stock_row["expiry_days_remaining"].iloc[0]) if not stock_row.empty else 9999.0
            predicted_demand = float(demand_row["predicted_demand"].iloc[0]) if not demand_row.empty else 0.0

            # Compute deficit per proposal: deficit = max(0, predicted_demand - (stock_level - safety_buffer))
            node_deficit = max(0.0, predicted_demand - max(stock_level - safety_buffer, 0.0))

            # Compute transferable surplus: surplus = max(0, stock_level - safety_buffer) only if expiry_days <= expiry_cutoff_days
            node_surplus = 0.0
            if expiry_days <= expiry_cutoff_days:
                node_surplus = max(0.0, stock_level - safety_buffer)

            inventory_lookup[node] = {
                "stock_level": stock_level,
                "unit_price": unit_price,
                "expiry_days": expiry_days,
            }
            deficit[node] = node_deficit
            surplus[node] = node_surplus

        # Skip medicine if no transferable surplus or no deficits exist
        total_surplus = sum(surplus.values())
        total_deficit = sum(deficit.values())
        if total_surplus <= 0 or total_deficit <= 0:
            continue

        # Create LP: maximize net savings
        prob = pulp.LpProblem(f"Lateral_Transshipment_{medicine.replace(' ', '_')}", pulp.LpMaximize)

        # Decision variables for quantities and binaries to enforce gatekeeping
        x = {}
        y = {}
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                # only consider pairs where source has surplus and dest has deficit
                if surplus[i] <= 0 or deficit[j] <= 0:
                    continue
                var_name = f"x_{i}_{j}"
                x[(i, j)] = pulp.LpVariable(var_name, lowBound=0, cat="Integer")
                y_name = f"y_{i}_{j}"
                y[(i, j)] = pulp.LpVariable(y_name, cat="Binary")

        # Supply constraints: sum_j x[i,j] <= surplus[i]
        for i in nodes:
            outgoing_vars = [x[(i, j)] for j in nodes if (i, j) in x]
            if outgoing_vars:
                prob += pulp.lpSum(outgoing_vars) <= surplus[i]

        # Demand constraints: sum_i x[i,j] <= deficit[j]
        for j in nodes:
            incoming_vars = [x[(i, j)] for i in nodes if (i, j) in x]
            if incoming_vars:
                prob += pulp.lpSum(incoming_vars) <= deficit[j]

        # Pairwise coupling constraints: x <= M*y and profit constraint per used pair
        for (i, j), xvar in list(x.items()):
            unit_price = inventory_lookup[i]["unit_price"]
            transport_cost_per_unit = cost_matrix[i][j] * batch_discount
            M_ij = min(surplus[i], deficit[j]) if min(surplus[i], deficit[j]) > 0 else 0
            # Upper bound linking
            prob += xvar <= M_ij * y[(i, j)]
            # Net savings must be positive when route used: (unit_price - transport_cost_per_unit) * x >= eps * y
            profit_per_unit = unit_price - transport_cost_per_unit
            # If profit_per_unit <= 0 then force y=0 (route never profitable)
            if profit_per_unit <= 0:
                prob += y[(i, j)] <= 0
            else:
                prob += profit_per_unit * xvar >= 0.01 * y[(i, j)]

        # Objective: maximize total net savings = sum_x (unit_price - transport_cost_per_unit) * x
        objective_terms = []
        for (i, j), xvar in x.items():
            unit_price = inventory_lookup[i]["unit_price"]
            transport_cost_per_unit = cost_matrix[i][j] * batch_discount
            objective_terms.append((unit_price - transport_cost_per_unit) * xvar)

        if not objective_terms:
            continue

        prob += pulp.lpSum(objective_terms)

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract positive, profitable routes
        for (i, j), xvar in x.items():
            qty = int(getattr(xvar, "varValue", 0) or 0)
            if qty <= 0:
                continue
            unit_price = inventory_lookup[i]["unit_price"]
            transport_cost = qty * cost_matrix[i][j] * batch_discount
            net_savings = (qty * unit_price) - transport_cost
            if net_savings <= 0:
                rejected_unprofitable_route_count += 1
                rejected_candidates.append({
                    "medicine": medicine,
                    "source_hospital": i,
                    "destination_hospital": j,
                    "quantity_to_move": qty,
                    "unit_price_lkr": round(unit_price, 2),
                    "financial_value_saved_lkr": round(qty * unit_price, 2),
                    "logistical_cost_lkr": round(transport_cost, 2),
                    "net_savings": round(net_savings, 2),
                    "reason": "non_positive_net_savings_postsolve",
                })
                continue

            transfer_manifest_records.append({
                "medicine": medicine,
                "source_hospital": i,
                "destination_hospital": j,
                "quantity_to_move": qty,
                "unit_price_lkr": round(unit_price, 2),
                "financial_value_saved_lkr": round(qty * unit_price, 2),
                "logistical_cost_lkr": round(transport_cost, 2),
                "net_savings": round(net_savings, 2),
            })

    if transfer_manifest_records:
        manifest_df = pd.DataFrame(transfer_manifest_records)
    else:
        manifest_df = pd.DataFrame(
            columns=[
                "medicine",
                "source_hospital",
                "destination_hospital",
                "quantity_to_move",
                "unit_price_lkr",
                "financial_value_saved_lkr",
                "logistical_cost_lkr",
                "net_savings",
            ]
        )

    output_path = f"{output_dir}/optimized_transshipment_manifest.csv"
    manifest_df.to_csv(output_path, index=False)

    structured_manifest = []
    for index, row in manifest_df.iterrows():
        structured_manifest.append(
            {
                "id": index + 1,
                "source_district": str(row.get("source_hospital") or "Unknown"),
                "dest_district": str(row.get("destination_hospital") or "Unknown"),
                "medicine": str(row.get("medicine") or "Unknown"),
                "quantity_to_move": int(row.get("quantity_to_move", 0) or 0),
                "unit_price_lkr": float(row.get("unit_price_lkr", 0) or 0.0),
                "transport_cost": float(row.get("logistical_cost_lkr", 0) or 0.0),
                "net_savings": float(row.get("net_savings", 0) or 0.0),
                "status": "PENDING_DISPATCH",
            }
        )

    total_net_savings_lkr = round(manifest_df["net_savings"].sum(), 2) if not manifest_df.empty else 0.0
    print(f"\n✅ Step 5 Complete: Operations Research strategy compiled in LKR. Manifest saved to {output_path}")
    print(f"Total redistribution routes generated: {len(structured_manifest)}")
    print(f"Total net savings enforced: LKR {total_net_savings_lkr}")
    print(f"Rejected unprofitable routes: {rejected_unprofitable_route_count}")

    return {
        "manifest": structured_manifest,
        "total_net_savings_lkr": total_net_savings_lkr,
        "rejected_unprofitable_route_count": rejected_unprofitable_route_count,
        "rejected_candidates": rejected_candidates,
    }


if __name__ == "__main__":
    run_transshipment_optimization()


    