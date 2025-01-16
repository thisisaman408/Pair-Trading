import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from momentfm import MOMENTPipeline

processed_data = pd.read_csv('ProcessedData/Processed_Pairs_Trading_Data.csv')


processed_data['Date'] = pd.to_datetime(processed_data['Date'])


visualization_folder = 'visualisation'
os.makedirs(visualization_folder, exist_ok=True)

# Initialize MOMENT for anomaly detection
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"}  # Reconstruction mode for anomaly detection
)
model.init()

# Prepare Spread data for anomaly detection
spread_series = processed_data[['Date', 'Spread']].set_index('Date')
spread_values = spread_series['Spread'].values

# Ensure data length aligns with MOMENT's patch size
patch_size = 512
context_length = (len(spread_values) // patch_size) * patch_size  # Truncate to nearest multiple of patch_size
spread_values = spread_values[:context_length]

# Convert to tensor [batch_size, n_channels, context_length]
spread_tensor = torch.tensor(spread_values, dtype=torch.float32).view(1, 1, -1)

# Create a DataLoader for fine-tuning
dataset = TensorDataset(spread_tensor.squeeze().unfold(dimension=-1, size=patch_size, step=patch_size))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Fine-Tuning Setup
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-Tuning Loop
n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
        batch_x = batch[0].to(device).unsqueeze(1)  # [batch_size, 1, patch_len]

        # Forward Pass
        output = model(x_enc=batch_x)
        loss = criterion(output.reconstruction, batch_x)
        total_loss += loss.item()

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(dataloader)}")

# Save the fine-tuned model
model.save_pretrained(os.path.join(visualization_folder, 'fine_tuned_model'))

# Perform anomaly detection using the fine-tuned model
model.eval()
with torch.no_grad():
    output = model(x_enc=spread_tensor.to(device))
    reconstructed_values = output.reconstruction.squeeze().cpu().numpy()
    anomaly_scores = (spread_values - reconstructed_values) ** 2  # MSE as anomaly score

# Add anomaly scores to the spread data
spread_series = spread_series.iloc[:context_length]  # Align the series with the truncated/padded data
spread_series['Anomaly_Score'] = anomaly_scores

# Define an anomaly threshold (e.g., 95th percentile of scores)
anomaly_threshold = spread_series['Anomaly_Score'].quantile(0.95)
spread_series['Anomaly'] = spread_series['Anomaly_Score'] > anomaly_threshold

# Save anomalies to a CSV file
anomalies = spread_series[spread_series['Anomaly']]
anomalies.to_csv(os.path.join(visualization_folder, 'fine_tuned_detected_anomalies.csv'))

# Plot and save the Spread with anomalies highlighted
spread_anomaly_plot_path = os.path.join(visualization_folder, 'fine_tuned_spread_with_anomalies.png')
plt.figure(figsize=(12, 6))
plt.plot(spread_series.index, spread_series['Spread'], label='Spread', color='blue')
plt.scatter(anomalies.index, anomalies['Spread'], color='red', label='Anomalies', zorder=5)
plt.axhline(y=spread_series['Spread'].mean(), color='green', linestyle='--', label='Mean Spread')
plt.title('Spread Between Coca-Cola and Pepsi with Fine-Tuned Anomalies')
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.savefig(spread_anomaly_plot_path)
plt.close()

print(f"Fine-Tuned Spread anomaly plot saved at: {spread_anomaly_plot_path}")
print(f"Fine-Tuned anomalies saved at: {os.path.join(visualization_folder, 'fine_tuned_detected_anomalies.csv')}")
