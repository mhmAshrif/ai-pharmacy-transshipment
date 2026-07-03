# src/data_pipeline.py
import pandas as pd
import os
import re

def fuse_healthcare_data():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    # Ensure our destination output folder exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Ingest filtered 5-node Sri Lanka public health records
    pharmacy_file = f"{raw_dir}/srilanka_5node_pharmacy_sales.csv"
    if not os.path.exists(pharmacy_file):
        raise FileNotFoundError(f"Error: Missing {pharmacy_file} inside data/raw/")
        
    print("Ingesting pharmacy sales data...")
    pharmacy_df = pd.read_csv(pharmacy_file)
    pharmacy_df['date'] = pd.to_datetime(pharmacy_df['date'])
    
    # 2. Map regional public health nodes to explicit weather records
    weather_mapping = {
        "Colombo": "colombo_weather_dataset.csv",
        "Kandy": "kandy_weather_dataset.csv",
        "Anuradhapura": "anuradhapura_weather_dataset.csv",
        "Jaffna": "jafna_weather_dataset.csv",
        "Galle": "galle_weather_dataset.csv"
    }
    
    weather_frames = []
    
    print("Processing and cleaning weather datasets...")
    for district, filename in weather_mapping.items():
        file_path = f"{raw_dir}/{filename}"
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: Weather file missing for {district} ({filename}). Skipping...")
            continue
            
        # Open-Meteo files skip the first 3 lines of API metadata headers
        df = pd.read_csv(file_path, skiprows=3)
        
        # Clean column names to remove bracketed units (e.g., '(mm)', '(°C)') and white spaces
        df.columns = [re.sub(r'\(.*\)', '', col).strip() for col in df.columns]
        
        # Standardize date keys
        df.rename(columns={'time': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        
        # Add a district key so we can perform a proper network join
        df['district'] = district
        
        # Retain features critical for exogenous context flags in Facebook Prophet
        df = df[['date', 'district', 'temperature_2m_mean', 'precipitation_sum']]
        weather_frames.append(df)
        
    if not weather_frames:
        raise ValueError("No weather datasets were loaded. Check your data/raw/ folder contents.")
        
    master_weather_df = pd.concat(weather_frames, ignore_index=True)
    
    # 3. Perform exact left-join on composite structural keys (date + district)
    print("Aligning and fusing datasets...")
    fused_df = pd.merge(pharmacy_df, master_weather_df, on=['date', 'district'], how='left')
    
    # Clean any unaligned empty spaces at dataset boundaries
    fused_df.dropna(subset=['units_sold', 'precipitation_sum'], inplace=True)
    
    # 4. Save clean processed artifact
    output_path = f"{processed_dir}/fused_master_dataset.csv"
    fused_df.to_csv(output_path, index=False)
    
    print(f"✅ Step 3 Complete: Consolidated data saved to {output_path}")
    print(f"Total processed rows: {len(fused_df)}")
    return fused_df

if __name__ == "__main__":
    fuse_healthcare_data()