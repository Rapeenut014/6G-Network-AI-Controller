from load_data import load_5g_kaggle_data
import os

print("Starting consolidation...")
if os.path.exists('archive/5G_Traffic_Datasets'):
    # Load the real large dataset
    df = load_5g_kaggle_data()
    
    # Save to the single compact CSV
    # preserving the format expected by load_data.py
    output_file = 'network_traffic_6g.csv'
    df.to_csv(output_file, index=True) # Index is Timestamp
    print(f"Successfully saved consolidated data to {output_file}")
    print(f"Rows: {len(df)}")
else:
    print("Error: archive folder not found. Cannot consolidate.")
