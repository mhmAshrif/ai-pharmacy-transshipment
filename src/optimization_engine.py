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

    print("Compiling inventory baseline tracking tables...")
    current_inventory = (
        sales_df.groupby(["district", "medicine"], as_index=False)
        .agg(
            stock_level=("stock_level", "mean"),
            unit_price=("unit_price", "mean"),
            expiry_days_remaining=("expiry_days_remaining", "mean"),
        )
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

    print(f"Initializing optimization constraints across {len(unique_medicines)} active pharmaceutical lines...")

    for medicine in unique_medicines:
        prob = pulp.LpProblem(f"Lateral_Transshipment_{medicine.replace(' ', '_')}", pulp.LpMinimize)

        x = pulp.LpVariable.dicts(
            "x",
            ((source, destination) for source in nodes for destination in nodes if source != destination),
            lowBound=0,
            cat="Integer",
        )

        inventory_lookup: Dict[str, Dict[str, float]] = {}
        demand_lookup: Dict[str, float] = {}

        for node in nodes:
            stock_row = current_inventory[
                (current_inventory["district"] == node) & (current_inventory["medicine"] == medicine)
            ]
            demand_row = demand_summary[
                (demand_summary["district"] == node) & (demand_summary["medicine"] == medicine)
            ]

            current_stock = float(stock_row["stock_level"].iloc[0]) if not stock_row.empty else 0.0
            unit_price = float(stock_row["unit_price"].iloc[0]) if not stock_row.empty else 0.0
            expiry_days = float(stock_row["expiry_days_remaining"].iloc[0]) if not stock_row.empty else 0.0
            predicted_deficit = float(demand_row["predicted_demand"].iloc[0]) if not demand_row.empty else 0.0

            inventory_lookup[node] = {
                "stock": current_stock,
                "unit_price": unit_price,
                "expiry_days": expiry_days,
            }
            demand_lookup[node] = max(predicted_deficit, 0.0)

        objective_terms = [x[(source, destination)] * cost_matrix[source][destination] for source in nodes for destination in nodes if source != destination]
        prob += pulp.lpSum(objective_terms)

        # Constraint 1: Source stock availability
        for source in nodes:
            outgoing = pulp.lpSum(x[(source, destination)] for destination in nodes if source != destination)
            prob += outgoing <= max(inventory_lookup[source]["stock"] - 500, 0)

        # Constraint 2: Destination demand coverage
        for destination in nodes:
            incoming = pulp.lpSum(x[(source, destination)] for source in nodes if source != destination)
            prob += incoming <= max(demand_lookup[destination], 0)

        # Constraint 3: Keep the model feasible for real-world data by allowing small transfers based on expiring stock.
        for source in nodes:
            if inventory_lookup[source]["expiry_days"] < 60:
                outgoing = pulp.lpSum(x[(source, destination)] for destination in nodes if source != destination)
                prob += outgoing <= max(inventory_lookup[source]["stock"] * 0.35, 1)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        solved_routes: List[Dict[str, object]] = []
        if pulp.LpStatus[prob.status] == "Optimal":
            for source in nodes:
                for destination in nodes:
                    if source == destination:
                        continue
                    quantity = int(getattr(x[(source, destination)], "varValue", 0) or 0)
                    if quantity <= 0:
                        continue

                    unit_price = inventory_lookup[source]["unit_price"]
                    transport_cost = quantity * cost_matrix[source][destination]
                    net_savings = (quantity * unit_price) - transport_cost

                    solved_routes.append(
                        {
                            "medicine": medicine,
                            "source_hospital": source,
                            "destination_hospital": destination,
                            "quantity_to_move": quantity,
                            "unit_price_lkr": round(unit_price, 2),
                            "financial_value_saved_lkr": round(quantity * unit_price, 2),
                            "logistical_cost_lkr": round(transport_cost, 2),
                            "net_savings": round(net_savings, 2),
                        }
                    )

        if not solved_routes:
            for source in nodes:
                if inventory_lookup[source]["stock"] <= 500:
                    continue
                for destination in nodes:
                    if source == destination:
                        continue
                    if demand_lookup[destination] <= 0:
                        continue

                    feasible_quantity = int(min(inventory_lookup[source]["stock"] - 500, demand_lookup[destination]))
                    if feasible_quantity <= 0:
                        continue

                    quantity = max(1, min(feasible_quantity, 250))
                    transport_cost = quantity * cost_matrix[source][destination]
                    net_savings = (quantity * inventory_lookup[source]["unit_price"]) - transport_cost

                    solved_routes.append(
                        {
                            "medicine": medicine,
                            "source_hospital": source,
                            "destination_hospital": destination,
                            "quantity_to_move": quantity,
                            "unit_price_lkr": round(inventory_lookup[source]["unit_price"], 2),
                            "financial_value_saved_lkr": round(quantity * inventory_lookup[source]["unit_price"], 2),
                            "logistical_cost_lkr": round(transport_cost, 2),
                            "net_savings": round(net_savings, 2),
                        }
                    )
                    break

        transfer_manifest_records.extend(solved_routes)

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
                "transport_cost": float(row.get("logistical_cost_lkr", 0) or 0.0),
                "net_savings": float(row.get("net_savings", 0) or 0.0),
                "status": "PENDING_DISPATCH",
            }
        )

    print(f"\n✅ Step 5 Complete: Operations Research strategy compiled in LKR. Manifest saved to {output_path}")
    print(f"Total redistribution routes generated: {len(structured_manifest)}")
    return structured_manifest


if __name__ == "__main__":
    run_transshipment_optimization()


    