# src/forecast_engine.py
import os
import warnings
from typing import Dict, Optional, List, Tuple
import multiprocessing

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

# Try to import joblib for parallel execution; fall back to sequential when unavailable
try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    _JOBLIB_AVAILABLE = False


def _prepare_prophet_frame(sub_df: pd.DataFrame) -> pd.DataFrame:
    prophet_df = sub_df[["date", "units_sold", "precipitation_sum"]].copy()
    prophet_df = prophet_df.rename(columns={"date": "ds", "units_sold": "y"})
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
    prophet_df = prophet_df.dropna(subset=["ds", "y", "precipitation_sum"]).sort_values("ds").reset_index(drop=True)
    return prophet_df


def _compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    return {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "mape": round(mape, 2),
    }


def _rolling_holdout_validation(
    prophet_df: pd.DataFrame,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    n_folds: int = 3,
) -> Dict[str, float]:
    """Evaluate a model across a small chronological walk-forward validation schedule."""
    total = len(prophet_df)
    if total < 45:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}

    fold_size = max(7, total // (n_folds + 2))
    metrics_list: List[Dict[str, float]] = []

    for fold_index in range(n_folds):
        validation_start = total - (n_folds - fold_index) * fold_size
        validation_end = total - (n_folds - fold_index - 1) * fold_size if fold_index < n_folds - 1 else total

        if validation_start <= 0:
            continue

        train_df = prophet_df.iloc[:validation_start].copy()
        val_df = prophet_df.iloc[validation_start:validation_end].copy()
        if len(train_df) < 20 or len(val_df) < 5:
            continue

        candidate = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        candidate.add_regressor("precipitation_sum")
        candidate.fit(train_df)

        val_future = val_df[["ds", "precipitation_sum"]].copy()
        val_pred = candidate.predict(val_future)
        metrics = _compute_error_metrics(
            y_true=val_df["y"].astype(float).to_numpy(),
            y_pred=val_pred["yhat"].astype(float).to_numpy(),
        )
        metrics_list.append(metrics)

    if not metrics_list:
        return {"rmse": 0.0, "mae": 0.0, "mape": 0.0}

    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in metrics_list])),
        "mae": float(np.mean([m["mae"] for m in metrics_list])),
        "mape": float(np.mean([m["mape"] for m in metrics_list])),
    }
    return {
        "rmse": round(avg_metrics["rmse"], 2),
        "mae": round(avg_metrics["mae"], 2),
        "mape": round(avg_metrics["mape"], 2),
    }


def build_prophet_forecast(
    district: str,
    medicine: str,
    historical_df: Optional[pd.DataFrame] = None,
    forecast_horizon: int = 30,
) -> Dict[str, object]:
    input_path = "data/processed/fused_master_dataset.csv"

    if historical_df is None:
        if not os.path.exists(input_path):
            raise FileNotFoundError("Error: 'fused_master_dataset.csv' not found. Run data_pipeline.py first!")
        historical_df = pd.read_csv(input_path)

    historical_df = historical_df.copy()
    historical_df["date"] = pd.to_datetime(historical_df["date"], errors="coerce")
    historical_df = historical_df.dropna(subset=["date"]).reset_index(drop=True)

    sub_df = historical_df[
        (historical_df["district"].astype(str).str.lower() == district.lower()) &
        (historical_df["medicine"].astype(str).str.lower() == medicine.lower())
    ].copy()

    if sub_df.empty:
        raise ValueError(f"No historical data found for {medicine} in {district}")

    sub_df = sub_df.sort_values("date").reset_index(drop=True)

    if len(sub_df) < 30:
        raise ValueError(f"Need at least 30 historical rows for Prophet training for {medicine} in {district}")

    prophet_df = _prepare_prophet_frame(sub_df)
    if len(prophet_df) < 30:
        raise ValueError(f"Need at least 30 usable rows for Prophet training for {medicine} in {district}")

    # Use a small rolling-origin holdout schedule to better estimate generalization risk.
    changepoint_grid = [0.01, 0.05, 0.1, 0.2]
    seasonality_grid = [0.1, 1.0, 5.0]

    best_mae = float("inf")
    best_params = {"changepoint_prior_scale": None, "seasonality_prior_scale": None}
    best_metrics = {"rmse": 0.0, "mae": 0.0, "mape": 0.0}

    for cps in changepoint_grid:
        for sps in seasonality_grid:
            try:
                metrics_candidate = _rolling_holdout_validation(
                    prophet_df=prophet_df,
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    n_folds=3,
                )

                if metrics_candidate["mae"] < best_mae:
                    best_mae = metrics_candidate["mae"]
                    best_params["changepoint_prior_scale"] = cps
                    best_params["seasonality_prior_scale"] = sps
                    best_metrics = metrics_candidate
            except Exception:
                continue

    # If no candidate improved (edge cases), fall back to defaults
    if best_params["changepoint_prior_scale"] is None:
        best_params["changepoint_prior_scale"] = 0.05
    if best_params["seasonality_prior_scale"] is None:
        best_params["seasonality_prior_scale"] = 1.0

    # Final model fitted on the full dataset using best hyperparameters
    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
    )
    final_model.add_regressor("precipitation_sum")
    final_model.fit(prophet_df)

    # Prepare future frame and predict
    future = final_model.make_future_dataframe(periods=forecast_horizon, freq="D")
    future["precipitation_sum"] = sub_df["precipitation_sum"].mean()
    forecast = final_model.predict(future)

    upcoming_predictions = forecast.tail(forecast_horizon)[["ds", "yhat"]].copy()
    upcoming_predictions = upcoming_predictions.rename(columns={"ds": "date", "yhat": "predicted_demand"})
    upcoming_predictions["predicted_demand"] = upcoming_predictions["predicted_demand"].clip(lower=0)
    upcoming_predictions["district"] = district
    upcoming_predictions["medicine"] = medicine

    tuned_parameters = {
        "best_changepoint_prior": float(best_params["changepoint_prior_scale"]),
        "best_seasonality_prior": float(best_params["seasonality_prior_scale"]),
    }

    return {
        "metrics": best_metrics,
        "future_predictions": upcoming_predictions,
        "tuned_parameters": tuned_parameters,
    }


def generate_demand_forecasts():
    input_path = "data/processed/fused_master_dataset.csv"
    output_dir = "data/processed"

    if not os.path.exists(input_path):
        raise FileNotFoundError("Error: 'fused_master_dataset.csv' not found. Run data_pipeline.py first!")

    print("Loading fused master healthcare dataset...")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # If a recent forecast exists, prefer to reuse it only if evaluation metrics also exist.
    output_path = f"{output_dir}/upcoming_demand_forecasts.csv"
    metrics_out = f"{output_dir}/forecast_evaluation_metrics.csv"
    try:
        if os.path.exists(output_path):
            mtime = os.path.getmtime(output_path)
            import time

            if (time.time() - mtime) < 24 * 3600:
                # If metrics file exists alongside the cached forecasts, reuse both and skip retrain.
                if os.path.exists(metrics_out):
                    print("Using cached demand forecasts and existing evaluation metrics (generated <24 hours ago). Skipping retrain.")
                    return pd.read_csv(output_path)
                else:
                    print("Cached forecasts found but evaluation metrics missing — recomputing to generate metrics.")
    except Exception:
        pass

    districts = df["district"].unique()

    # Pre-compute counts per (district, medicine) to avoid starting models for sparse series
    counts = df.groupby(["district", "medicine"]).size().reset_index(name="count")

    all_forecasts = []
    all_metrics = []

    print(f"Beginning localized Prophet training across {len(districts)} hospital nodes (filtered)...")

    # Build task list of (district, medicine) pairs with at least 30 observations
    tasks: List[Tuple[str, str]] = []
    for district in districts:
        valid_meds = counts[(counts["district"] == district) & (counts["count"] >= 30)]["medicine"].unique()
        for medicine in valid_meds:
            tasks.append((str(district), str(medicine)))

    def _fit_one(task: Tuple[str, str]):
        district, medicine = task
        try:
            payload = build_prophet_forecast(district=district, medicine=medicine, historical_df=df, forecast_horizon=30)
            # attach identifying info and metrics
            future = payload.get("future_predictions")
            metrics = payload.get("metrics") or {}
            tuned = payload.get("tuned_parameters") or {}
            if future is not None:
                future = future.copy()
                future["district"] = district
                future["medicine"] = medicine

            metric_row = {
                "district": district,
                "medicine": medicine,
                "rmse": float(metrics.get("rmse", 0.0)),
                "mae": float(metrics.get("mae", 0.0)),
                "mape": float(metrics.get("mape", 0.0)),
                "best_changepoint_prior": float(tuned.get("best_changepoint_prior", 0.0)),
                "best_seasonality_prior": float(tuned.get("best_seasonality_prior", 0.0)),
            }

            return {"future": future, "metrics": metric_row}
        except Exception as e:
            print(f"     Skipping {medicine} in {district} due to fitting variance: {str(e)}")
            return None

    if not tasks:
        raise ValueError("AI engine found no (district, medicine) series with sufficient history for forecasting.")

    # Choose number of parallel workers: leave one core free when possible
    try:
        cpu_cnt = max(1, multiprocessing.cpu_count() - 1)
    except Exception:
        cpu_cnt = 1

    if _JOBLIB_AVAILABLE and len(tasks) > 1 and cpu_cnt > 1:
        n_jobs = min(len(tasks), cpu_cnt)
        print(f"Running Prophet fits in parallel using joblib with n_jobs={n_jobs}")
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_one)(t) for t in tasks)
        except Exception as e:
            print(f"Parallel execution failed, falling back to sequential: {e}")
            results = [ _fit_one(t) for t in tasks ]
    else:
        print("Running Prophet fits sequentially (joblib unavailable or single-worker)")
        results = [ _fit_one(t) for t in tasks ]

    # Collect non-empty results and separate metrics
    for r in results:
        if r is None:
            continue
        future = r.get("future") if isinstance(r, dict) else None
        metrics = r.get("metrics") if isinstance(r, dict) else None
        if future is not None:
            all_forecasts.append(future)
        if metrics is not None:
            all_metrics.append(metrics)

    if not all_forecasts:
        raise ValueError("AI engine failed to produce valid future matrices. Review underlying transaction densities.")

    master_forecast_df = pd.concat(all_forecasts, ignore_index=True)

    output_path = f"{output_dir}/upcoming_demand_forecasts.csv"
    master_forecast_df.to_csv(output_path, index=False)

    # Persist evaluation metrics summary
    try:
        metrics_df = pd.DataFrame(all_metrics)
        if not metrics_df.empty:
            metrics_df["evaluated_at"] = pd.Timestamp.now()
            metrics_out = f"{output_dir}/forecast_evaluation_metrics.csv"
            metrics_df.to_csv(metrics_out, index=False)
            print(f"✅ Forecast evaluation metrics exported to {metrics_out}")
    except Exception as e:
        print(f"Warning: failed to export forecast evaluation metrics: {e}")

    print(f"✅ Step 4 Complete: 30-Day Predictive Future Model matrices exported to {output_path}")
    return master_forecast_df


if __name__ == "__main__":
    generate_demand_forecasts()