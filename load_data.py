import pandas as pd
import numpy as np
import os
import glob

def load_5g_kaggle_data(folder_path='archive/5G_Traffic_Datasets', sample_file=None):
    """
    Loads the 5G Traffic Kaggle dataset (packet-level).
    Combines multiple traffic sources for better model training.
    """
    print("Loading REAL 5G Kaggle Dataset (Multi-Source)...")
    
    # Define files to load (one from each category for diversity)
    target_files = [
        os.path.join(folder_path, 'Video_Conferencing/Zoom/Zoom_1.csv'),
        os.path.join(folder_path, 'Stored_Streaming/Netflix/Netflix_1.csv'),
        os.path.join(folder_path, 'Stored_Streaming/YouTube/YouTube_1.csv'),
        os.path.join(folder_path, 'Live_Streaming/YouTube_Live/YouTube_Live_1.csv'),
    ]
    
    # Filter to existing files only
    csv_files = [f for f in target_files if os.path.exists(f)]
    
    if not csv_files:
        print("No CSV files found. Using mock data.")
        return generate_mock_6g_data()
    
    print(f"Loading {len(csv_files)} traffic sources...")
    
    all_data = []
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
        try:
            df = pd.read_csv(csv_file, nrows=1000000)  # 1M rows per file
            df['Source'] = os.path.basename(csv_file).replace('.csv', '')
            all_data.append(df)
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    if not all_data:
        return generate_mock_6g_data()
    
    # Combine all files
    df = pd.concat(all_data, ignore_index=True)
    print(f"Total packets loaded: {len(df):,}")
    
    # Parse Time column
    df['Timestamp'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    # Length is packet size in bytes
    df['Bytes'] = pd.to_numeric(df['Length'], errors='coerce').fillna(0)
    
    # Aggregate to 1-minute intervals
    df.set_index('Timestamp', inplace=True)
    df_agg = df.resample('1min').agg({
        'Bytes': 'sum',
        'No.': 'count'
    }).reset_index()
    
    # Convert Bytes to Mbps
    df_agg['DL_Throughput_Mbps'] = (df_agg['Bytes'] * 8 / 60 / 1e6).round(2)
    df_agg['UL_Throughput_Mbps'] = (df_agg['DL_Throughput_Mbps'] * 0.3).round(2)
    df_agg['Packet_Count'] = df_agg['No.']
    
    # Simulate Latency/Jitter
    df_agg['Latency_ms'] = 5 + (df_agg['Packet_Count'] / df_agg['Packet_Count'].max()) * 15 + np.random.normal(0, 1, len(df_agg))
    df_agg['Latency_ms'] = df_agg['Latency_ms'].clip(1, 50)
    df_agg['Jitter_ms'] = np.abs(np.random.normal(1, 0.5, len(df_agg))).round(3)
    
    result = df_agg[['Timestamp', 'DL_Throughput_Mbps', 'UL_Throughput_Mbps', 'Latency_ms', 'Jitter_ms']].copy()
    
    # Filter out zero-traffic intervals (gaps between sessions)
    result = result[result['DL_Throughput_Mbps'] > 0.01]
    result = result.reset_index(drop=True)
    
    print(f"Loaded {len(result)} non-zero time intervals from combined 5G data.")
    return result

def generate_mock_6g_data(filename='network_traffic_6g.csv', days=14):
    """
    Generates synthetic 6G Network Performance Data.
    (Fallback if no real data available)
    """
    print(f"Generating Mock 6G Data: {filename}...")
    
    intervals = pd.date_range(start=pd.Timestamp.now().floor('D') - pd.Timedelta(days=days), 
                              periods=1440*days, freq='1min')
    
    n = len(intervals)
    time_val = (intervals.hour * 60 + intervals.minute)
    daily_pattern = np.sin(time_val * 2 * np.pi / 1440 - np.pi/2) + 1
    
    base_speed = 2000
    variable_speed = daily_pattern * 3000
    noise = np.random.normal(0, 500, n)
    bursts = np.random.choice([0, 5000], size=n, p=[0.99, 0.01])
    
    dl_throughput = base_speed + variable_speed + noise + bursts
    dl_throughput = np.maximum(dl_throughput, 100)
    
    latency_base = 5 
    congestion_factor = (dl_throughput / 10000) * 10
    latency_noise = np.random.exponential(2, n)
    latency = latency_base + congestion_factor + latency_noise
    
    jitter = np.abs(np.random.normal(1, 0.5, n))
    
    df = pd.DataFrame({
        'Timestamp': intervals,
        'DL_Throughput_Mbps': dl_throughput.round(2),
        'UL_Throughput_Mbps': (dl_throughput * 0.4).round(2),
        'Latency_ms': latency.round(2),
        'Jitter_ms': jitter.round(3)
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")
    return df

def load_data(filepath='network_traffic_6g.csv', use_kaggle=True):
    """
    Standard loader for the 6G project.
    Set use_kaggle=True to load real Kaggle data.
    """
    # 1. Try loading consolidated real data
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df

    # 2. Fallback to raw data if CSV missing (e.g. first run)
    if use_kaggle and os.path.exists('archive/5G_Traffic_Datasets'):
        return load_5g_kaggle_data()
    
    # 3. Generate mock data if nothing exists
    generate_mock_6g_data(filepath)
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

if __name__ == "__main__":
    df = load_data(use_kaggle=True)
    print(df.head())
    print(f"\nData Schema:\n{df.dtypes}")
    print(f"\nTotal rows: {len(df)}")
