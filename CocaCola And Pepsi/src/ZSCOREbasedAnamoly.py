import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from momentfm import MOMENTPipeline

# Load the processed data
processed_data = pd.read_csv('ProcessedData/Processed_Pairs_Trading_Data.csv')

# Convert 'Date' column to datetime for proper plotting
processed_data['Date'] = pd.to_datetime(processed_data['Date'])

# Create a folder for visualizations if it doesn't exist
visualization_folder = 'visualisation'
os.makedirs(visualization_folder, exist_ok=True)

# Initialize MOMENT for anomaly detection
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"}  # Reconstruction mode for anomaly detection
)
model.init()

# Prepare Z-Score data for anomaly detection
zscore_series = processed_data[['Date', 'Z-Score']].set_index('Date')
zscore_values = zscore_series['Z-Score'].values

# Ensure data length aligns with MOMENT's patch size
patch_size = 512
context_length = (len(zscore_values) // patch_size) * patch_size  # Truncate to nearest multiple of patch_size
zscore_values = zscore_values[:context_length]

# Convert to tensor [batch_size, n_channels, context_length]
zscore_tensor = torch.tensor(zscore_values, dtype=torch.float32).view(1, 1, -1)

# Debugging: Check tensor dimensions
print(f"Z-Score tensor shape: {zscore_tensor.shape}")  # Expected: [1, 1, context_length]

# Perform anomaly detection using MOMENT
with torch.no_grad():
    try:
        # Note: No `input_mask` is explicitly passed (consistent with your spread-based code)
        output = model(x_enc=zscore_tensor)
        reconstructed_values = output.reconstruction.squeeze().numpy()
        anomaly_scores = (zscore_values - reconstructed_values) ** 2  # Mean Squared Error as anomaly score
    except RuntimeError as e:
        print(f"Error during MOMENT processing: {e}")
        raise

# Add anomaly scores to the Z-Score data
zscore_series = zscore_series.iloc[:context_length]  # Align the series with the truncated/padded data
zscore_series['Anomaly_Score'] = anomaly_scores

# Define an anomaly threshold (e.g., 95th percentile of scores)
anomaly_threshold = zscore_series['Anomaly_Score'].quantile(0.95)
zscore_series['Anomaly'] = zscore_series['Anomaly_Score'] > anomaly_threshold

# Save anomalies to a CSV file
anomalies = zscore_series[zscore_series['Anomaly']]
anomalies.to_csv(os.path.join(visualization_folder, 'detected_anomalies_zscore.csv'))

# Plot and save the Z-Score with anomalies highlighted
zscore_anomaly_plot_path = os.path.join(visualization_folder, 'zscore_with_anomalies.png')
plt.figure(figsize=(12, 6))
plt.plot(zscore_series.index, zscore_series['Z-Score'], label='Z-Score', color='blue')
plt.scatter(anomalies.index, anomalies['Z-Score'], color='red', label='Anomalies', zorder=5)
plt.axhline(y=zscore_series['Z-Score'].mean(), color='green', linestyle='--', label='Mean Z-Score')
plt.axhline(y=2.5, color='orange', linestyle='--', label='Upper Threshold')
plt.axhline(y=-2.5, color='orange', linestyle='--', label='Lower Threshold')
plt.title('Z-Score Between Coca-Cola and Pepsi with Anomalies')
plt.xlabel('Date')
plt.ylabel('Z-Score')
plt.legend()
plt.grid(True)
plt.savefig(zscore_anomaly_plot_path)
plt.close()

print(f"Z-Score anomaly plot saved at: {zscore_anomaly_plot_path}")
print(f"Anomalies saved at: {os.path.join(visualization_folder, 'detected_anomalies_zscore.csv')}")
