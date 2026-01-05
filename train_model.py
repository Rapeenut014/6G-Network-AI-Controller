import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from load_data import load_data

def preprocess_data(df):
    """
    Feature Engineering for 6G Network Traffic.
    """
    df = df.sort_values('Timestamp')
    
    # Time Features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    
    # Target: DL_Throughput_Mbps
    # Features from History
    target = 'DL_Throughput_Mbps'
    
    # Create Lags (Autoregressive features)
    # Data is 1-minute interval
    df['Lag_1m'] = df[target].shift(1)
    df['Lag_5m'] = df[target].shift(5)
    df['Lag_60m'] = df[target].shift(60) # 1 hour ago
    
    # Rolling Statistics (Trends)
    df['Rolling_Mean_10m'] = df[target].shift(1).rolling(window=10).mean()
    df['Rolling_Std_10m'] = df[target].shift(1).rolling(window=10).std()
    
    # Context Features (Latency influences Throughput perception/TCP window)
    df['Lag_Latency'] = df['Latency_ms'].shift(1)
    
    df = df.dropna()
    return df

def train_model(df):
    """Trains the 6G Throughput Predictor."""
    # Split chronologically
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Features to use
    features = ['Hour', 'Minute', 'DayOfWeek', 
                'Lag_1m', 'Lag_5m', 'Lag_60m', 
                'Rolling_Mean_10m', 'Rolling_Std_10m', 'Lag_Latency']
    
    target = 'DL_Throughput_Mbps'
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    print(f"Training on {len(train)} samples...")
    model = HistGradientBoostingRegressor(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    return model, test, features, X_test, y_test

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f} Mbps")
    print(f"R2 Score: {r2:.4f}")
    
    return preds

if __name__ == "__main__":
    print("Starting 6G Model Training...")
    
    # 1. Load
    raw_df = load_data()
    
    # 2. Preprocess
    df = preprocess_data(raw_df)
    
    # 3. Train
    model, test_df, features, X_test, y_test = train_model(df)
    
    # 4. Evaluate
    preds = evaluate_model(model, X_test, y_test)
    
    # 5. Save
    with open('model_6g.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    # Save test results for Dashboard
    results = test_df.copy()
    results['Predicted_Throughput'] = preds
    results.to_csv('test_results_6g.csv', index=False)
    print("Model saved to 'model_6g.pkl'")
