# src/forecast_engine.py
import os
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")


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
        "mape_percent": round(mape, 2),
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

    split_index = int(len(prophet_df) * 0.8)
    split_index = max(2, min(split_index, len(prophet_df) - 1))
    train_df = prophet_df.iloc[:split_index].copy()
    test_df = prophet_df.iloc[split_index:].copy()

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_regressor("precipitation_sum")
    model.fit(train_df)

    if not test_df.empty:
        test_future = test_df[["ds", "precipitation_sum"]].copy()
        test_forecast = model.predict(test_future)
        metrics = _compute_error_metrics(
            y_true=test_df["y"].astype(float).to_numpy(),
            y_pred=test_forecast["yhat"].astype(float).to_numpy(),
        )
    else:
        metrics = {"rmse": 0.0, "mae": 0.0, "mape_percent": 0.0}

    full_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    full_model.add_regressor("precipitation_sum")
    full_model.fit(prophet_df)

    future = full_model.make_future_dataframe(periods=forecast_horizon, freq="D")
    future["precipitation_sum"] = sub_df["precipitation_sum"].mean()
    forecast = full_model.predict(future)

    upcoming_predictions = forecast.tail(forecast_horizon)[["ds", "yhat"]].copy()
    upcoming_predictions = upcoming_predictions.rename(columns={"ds": "date", "yhat": "predicted_demand"})
    upcoming_predictions["predicted_demand"] = upcoming_predictions["predicted_demand"].clip(lower=0)
    upcoming_predictions["district"] = district
    upcoming_predictions["medicine"] = medicine

    return {
        "metrics": metrics,
        "future_predictions": upcoming_predictions,
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

    districts = df["district"].unique()
    medicines = df["medicine"].unique()

    all_forecasts = []

    print(f"Beginning localized Prophet training across {len(districts)} hospital nodes...")

    for district in districts:
        print(f" -> Training AI models for Hospital Node: {district}")
        for medicine in medicines:
            try:
                model_payload = build_prophet_forecast(
                    district=str(district),
                    medicine=str(medicine),
                    historical_df=df,
                    forecast_horizon=30,
                )
                all_forecasts.append(model_payload["future_predictions"])
            except Exception as e:
                print(f"     Skipping {medicine} in {district} due to fitting variance: {str(e)}")
                continue

    if not all_forecasts:
        raise ValueError("AI engine failed to produce valid future matrices. Review underlying transaction densities.")

    master_forecast_df = pd.concat(all_forecasts, ignore_index=True)

    output_path = f"{output_dir}/upcoming_demand_forecasts.csv"
    master_forecast_df.to_csv(output_path, index=False)

    print(f"✅ Step 4 Complete: 30-Day Predictive Future Model matrices exported to {output_path}")
    return master_forecast_df


if __name__ == "__main__":
    generate_demand_forecasts()