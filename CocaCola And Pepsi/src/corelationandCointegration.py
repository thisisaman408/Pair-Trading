import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

processed_data = pd.read_csv('ProcessedData/Processed_Pairs_Trading_Data.csv')

processed_data['Date'] = pd.to_datetime(processed_data['Date'])

correlation = processed_data['Close/Last_KO'].corr(processed_data['Close/Last_PEP'])


coint_stat, p_value, _ = coint(processed_data['Close/Last_KO'], processed_data['Close/Last_PEP'])

visualization_folder = 'visualisation'
os.makedirs(visualization_folder, exist_ok=True)

price_plot_path = os.path.join(visualization_folder, 'stock_prices_with_stats.png')
plt.figure(figsize=(12, 6))
plt.plot(processed_data['Date'], processed_data['Close/Last_KO'], label='Coca-Cola (KO)', color='blue')
plt.plot(processed_data['Date'], processed_data['Close/Last_PEP'], label='Pepsi (PEP)', color='orange')
plt.title(f'Stock Prices of Coca-Cola and Pepsi Over Time\nCorrelation: {correlation:.2f}, Cointegration Test Stat: {coint_stat:.2f}, p-value: {p_value:.2e}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.savefig(price_plot_path)
plt.close()

print(f"Price plot saved at: {price_plot_path}")
