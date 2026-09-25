import pandas as pd

sales_path = 'data/processed/fused_master_dataset.csv'
forecast_path = 'data/processed/upcoming_demand_forecasts.csv'

sales = pd.read_csv(sales_path)
forecast = pd.read_csv(forecast_path)

sales['medicine'] = sales['medicine'].astype(str).str.strip()
forecast['medicine'] = forecast['medicine'].astype(str).str.strip()

cur = (
    sales.groupby(['district','medicine'],as_index=False)
    .agg(stock_level=('stock_level','mean'), unit_price=('unit_price','mean'), expiry_days_remaining=('expiry_days_remaining','mean'))
)

demand = forecast.groupby(['district','medicine'],as_index=False)['predicted_demand'].sum()

nodes = ['Colombo','Kandy','Galle','Anuradhapura','Jaffna']

safety_buffer = 500
expiry_cutoff_days = 90

medicines = sorted(cur['medicine'].unique())[:10]

for med in medicines:
    print('\n===', med)
    total_surplus = 0
    total_deficit = 0
    for node in nodes:
        stock_row = cur[(cur['district']==node)&(cur['medicine']==med)]
        demand_row = demand[(demand['district']==node)&(demand['medicine']==med)]
        stock = float(stock_row['stock_level'].iloc[0]) if not stock_row.empty else 0.0
        unit_price = float(stock_row['unit_price'].iloc[0]) if not stock_row.empty else 0.0
        expiry = float(stock_row['expiry_days_remaining'].iloc[0]) if not stock_row.empty else 9999.0
        predicted = float(demand_row['predicted_demand'].iloc[0]) if not demand_row.empty else 0.0
        node_deficit = max(0.0, predicted - max(stock - safety_buffer, 0.0))
        node_surplus = max(0.0, stock - safety_buffer) if expiry <= expiry_cutoff_days else 0.0
        print(f"{node}: stock={stock:.1f}, expiry={expiry:.1f}, pred={predicted:.1f}, surplus={node_surplus:.1f}, deficit={node_deficit:.1f}, unit_price={unit_price:.2f}")
        total_surplus += node_surplus
        total_deficit += node_deficit
    print('Total surplus:', total_surplus, 'Total deficit:', total_deficit)
