import os
import pandas as pd

paths = [
    'data/processed/optimized_transshipment_manifest.csv',
    'data/processed/upcoming_demand_forecasts.csv',
    'data/processed/fused_master_dataset.csv',
]

for path in paths:
    print('\nFILE', path, 'exists=', os.path.exists(path))
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(df.head(3).to_string())
        print('COLUMNS', list(df.columns))
