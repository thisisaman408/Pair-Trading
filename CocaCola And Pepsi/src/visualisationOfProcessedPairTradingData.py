import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the processed data
processed_data = pd.read_csv('ProcessedData/Processed_Pairs_Trading_Data.csv')

# Convert 'Date' column to datetime for proper plotting
processed_data['Date'] = pd.to_datetime(processed_data['Date'])

# Create a folder for visualizations if it doesn't exist
visualization_folder = 'visualisation'
os.makedirs(visualization_folder, exist_ok=True)

# Plot and save the Spread over time
spread_plot_path = os.path.join(visualization_folder, 'spread_over_time.png')
plt.figure(figsize=(12, 6))
plt.plot(processed_data['Date'], processed_data['Spread'], label='Spread', color='blue')
plt.axhline(y=processed_data['Spread'].mean(), color='red', linestyle='--', label='Mean Spread')
plt.title('Spread Between Coca-Cola and Pepsi Over Time')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.savefig(spread_plot_path)
plt.close()

# Plot and save the Z-Scores over time
zscore_plot_path = os.path.join(visualization_folder, 'zscore_over_time.png')
plt.figure(figsize=(12, 6))
plt.plot(processed_data['Date'], processed_data['Z-Score'], label='Z-Score', color='green')
plt.axhline(y=2, color='red', linestyle='--', label='Upper Threshold (Z=2)')
plt.axhline(y=-2, color='red', linestyle='--', label='Lower Threshold (Z=-2)')
plt.title('Z-Scores of Spread Over Time')
plt.xlabel('Date')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True)
plt.savefig(zscore_plot_path)
plt.close()

print(f"Spread plot saved at: {spread_plot_path}")
print(f"Z-Score plot saved at: {zscore_plot_path}")
