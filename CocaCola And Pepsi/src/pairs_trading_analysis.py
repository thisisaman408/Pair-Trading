import pandas as pd

coca_cola_data = pd.read_csv('Data/CocaColaData.csv')
pepsi_data = pd.read_csv('Data/PepsiData.csv')


coca_cola_data['Date'] = pd.to_datetime(coca_cola_data['Date'])
coca_cola_data['Close/Last'] = coca_cola_data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)
coca_cola_data = coca_cola_data.sort_values('Date')


pepsi_data['Date'] = pd.to_datetime(pepsi_data['Date'])
pepsi_data['Close/Last'] = pepsi_data['Close/Last'].replace({'\$': ''}, regex=True).astype(float)
pepsi_data = pepsi_data.sort_values('Date')


merged_data = pd.merge(coca_cola_data[['Date', 'Close/Last']],
                       pepsi_data[['Date', 'Close/Last']],
                       on='Date', suffixes=('_KO', '_PEP'))

merged_data['Spread'] = merged_data['Close/Last_KO'] - merged_data['Close/Last_PEP']



spread_mean = merged_data['Spread'].mean()
spread_std = merged_data['Spread'].std()


merged_data['Z-Score'] = (merged_data['Spread'] - spread_mean) / spread_std

merged_data.to_csv('ProcessedData/Processed_Pairs_Trading_Data.csv', index=False)
