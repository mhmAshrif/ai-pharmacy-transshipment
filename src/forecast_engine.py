# src/forecast_engine.py
import pandas as pd
from prophet import Prophet
import os
import warnings
warnings.filterwarnings('ignore')

def generate_demand_forecasts():
    input_path = "data/processed/fused_master_dataset.csv"
    output_dir = "data/processed"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError("Error: 'fused_master_dataset.csv' not found. Run data_pipeline.py first!")
        
    print("Loading fused master healthcare dataset...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    districts = df['district'].unique()
    medicines = df['medicine'].unique()
    
    all_forecasts = []
    
    print(f"Beginning localized Prophet training across {len(districts)} hospital nodes...")
    
    for district in districts:
        print(f" -> Training AI models for Hospital Node: {district}")
        for medicine in medicines:
            # Isolate data for this specific district and drug combination
            sub_df = df[(df['district'] == district) & (df['medicine'] == medicine)].copy()
            
            # Prophet requires at least 30 non-null data points to establish standard seasonality parameters
            if len(sub_df) < 30:
                continue
                
            # Restructure data schema into standard Facebook Prophet columns (ds and y)
            prophet_df = sub_df[['date', 'units_sold', 'precipitation_sum']].rename(
                columns={'date': 'ds', 'y': 'units_sold', 'units_sold': 'y'}
            )
            
            # Sort chronological timeline rows
            prophet_df = prophet_df.sort_values(by='ds').reset_index(drop=True)
            
            try:
                # Instantiate Prophet with annual seasonality hooks for monsoonal iterations
                model = Prophet(
                    yearly_seasonality=True, 
                    weekly_seasonality=True, 
                    daily_seasonality=False
                )
                
                # CRITICAL RESILIENCE RULE: Add tropical rainfall patterns as an exogenous context regressor
                model.add_regressor('precipitation_sum')
                
                # Fit the machine learning model
                model.fit(prophet_df)
                
                # Construct future projection skeleton data frame for the upcoming 30 days
                future = model.make_future_dataframe(periods=30, freq='D')
                
                # Create a baseline proxy for future rainfall using historical average behavior
                mean_rain = sub_df['precipitation_sum'].mean()
                future['precipitation_sum'] = mean_rain
                
                # Run the trained model to extract predictive target outputs
                forecast = model.predict(future)
                
                # Capture only the newly predicted subsequent 30 days tail window array
                upcoming_predictions = forecast.tail(30)[['ds', 'yhat']].copy()
                upcoming_predictions.rename(columns={'ds': 'date', 'yhat': 'predicted_demand'}, inplace=True)
                
                # Force negative forecast artifacts to 0 (since you can't sell negative drug units)
                upcoming_predictions['predicted_demand'] = upcoming_predictions['predicted_demand'].clip(lower=0)
                
                # Append organizational metadata tags
                upcoming_predictions['district'] = district
                upcoming_predictions['medicine'] = medicine
                
                all_forecasts.append(upcoming_predictions)
                
            except Exception as e:
                print(f"     Skipping {medicine} in {district} due to fitting variance: {str(e)}")
                continue

    if not all_forecasts:
        raise ValueError("AI engine failed to produce valid future matrices. Review underlying transaction densities.")
        
    # Combine individual matrix chunks into a macro system table
    master_forecast_df = pd.concat(all_forecasts, ignore_index=True)
    
    # Save the prediction data artifact
    output_path = f"{output_dir}/upcoming_demand_forecasts.csv"
    master_forecast_df.to_csv(output_path, index=False)
    
    print(f"✅ Step 4 Complete: 30-Day Predictive Future Model matrices exported to {output_path}")
    return master_forecast_df

if __name__ == "__main__":
    generate_demand_forecasts()