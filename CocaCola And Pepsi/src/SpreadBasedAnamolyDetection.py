import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from momentfm import MOMENTPipeline

processed_data = pd.read_csv('ProcessedData/Processed_Pairs_Trading_Data.csv')

processed_data['Date'] = pd.to_datetime(processed_data['Date'])


visualization_folder = 'visualisation'
os.makedirs(visualization_folder, exist_ok=True)


model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"}  
)
model.init()


spread_series = processed_data[['Date', 'Spread']].set_index('Date')
spread_values = spread_series['Spread'].values


patch_size = 512
context_length = (len(spread_values) // patch_size) * patch_size  
spread_values = spread_values[:context_length]

spread_tensor = torch.tensor(spread_values, dtype=torch.float32).view(1, 1, -1)


with torch.no_grad():
    try:
        output = model(x_enc=spread_tensor) 
        reconstructed_values = output.reconstruction.squeeze().numpy()
        anomaly_scores = (spread_values - reconstructed_values) ** 2 
    except RuntimeError as e:
        print(f"Error during MOMENT processing: {e}")
        raise


spread_series = spread_series.iloc[:context_length] 
spread_series['Anomaly_Score'] = anomaly_scores


anomaly_threshold = spread_series['Anomaly_Score'].quantile(0.95)
spread_series['Anomaly'] = spread_series['Anomaly_Score'] > anomaly_threshold


anomalies = spread_series[spread_series['Anomaly']]
anomalies.to_csv(os.path.join(visualization_folder, 'detected_anomalies.csv'))


spread_anomaly_plot_path = os.path.join(visualization_folder, 'spread_with_anomalies.png')
plt.figure(figsize=(12, 6))
plt.plot(spread_series.index, spread_series['Spread'], label='Spread', color='blue')
plt.scatter(anomalies.index, anomalies['Spread'], color='red', label='Anomalies', zorder=5)
plt.axhline(y=spread_series['Spread'].mean(), color='green', linestyle='--', label='Mean Spread')
plt.title('Spread Between Coca-Cola and Pepsi with Anomalies')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.savefig(spread_anomaly_plot_path)
plt.close()

print(f"Spread anomaly plot saved at: {spread_anomaly_plot_path}")
print(f"Anomalies saved at: {os.path.join(visualization_folder, 'detected_anomalies.csv')}")
