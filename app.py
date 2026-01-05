import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="6G Network AI", layout="wide", page_icon="📡")

# --- Custom Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📡 AI-Driven 6G Network Controller & QoS Intelligence")
st.caption("Next-Gen Artificial Intelligence for Cellular Network Management")

# --- Sidebar ---
st.sidebar.header("Network Controller")
page = st.sidebar.radio("Module Selection", ["📊 Network Health Monitor", "🚀 QoS Future Forecast"])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **System Status:** ONLINE 🟢
    **Network:** 6G-URLLC Slice
    **Target Latency:** < 5ms
    **Target Throughput:** > 1 Gbps
    """
)

# --- Data Loading ---
@st.cache_data
def load_results():
    try:
        df = pd.read_csv('test_results_6g.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    try:
        with open('model_6g.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

df = load_results()

if df is None:
    st.error("Data not found. Please run 'train_model.py' first.")
    st.stop()

# --- Page 1: Network Health Monitor ---
if page == "📊 Network Health Monitor":
    st.subheader("Real-time Network Telemetry")
    
    # Latest Metadata
    latest = df.iloc[-1]
    last_1h = df.tail(60).copy()
    
    # --- ANOMALY DETECTION ENGINE ---
    # Logic: Detect Latency Spikes (> 15ms)
    last_1h['Is_Anomaly'] = last_1h['Latency_ms'] > 15
    anomalies = last_1h[last_1h['Is_Anomaly']]
    
    # Update 'latest' to include the new column
    is_anomaly = last_1h.iloc[-1]['Is_Anomaly']
    
    # Use mean of recent values for stable display (avoid showing 0 from gaps)
    recent_dl = last_1h['DL_Throughput_Mbps'].tail(10)
    recent_ul = last_1h['UL_Throughput_Mbps'].tail(10)
    avg_dl = recent_dl[recent_dl > 0].mean() if (recent_dl > 0).any() else recent_dl.mean()
    avg_ul = recent_ul[recent_ul > 0].mean() if (recent_ul > 0).any() else recent_ul.mean()
    
    # Gauges / Metrics (using stable averages)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("DL Throughput", f"{avg_dl:.2f} Mbps", delta=f"{avg_dl - recent_dl.iloc[-2] if len(recent_dl) > 1 else 0:.2f}")
    with col2:
        st.metric("UL Throughput", f"{avg_ul:.2f} Mbps")
    with col3:
        st.metric("Latency", f"{latest['Latency_ms']:.2f} ms", delta_color="inverse", delta=f"{latest['Latency_ms'] - df.iloc[-2]['Latency_ms']:.2f}")
    with col4:
        st.metric("Jitter", f"{latest['Jitter_ms']:.3f} ms")
        
    st.markdown("---")
    
    # Graphs
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 📉 Throughput Stability (Last 1hr)")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(last_1h['Timestamp'], last_1h['DL_Throughput_Mbps'], color='#00FFAA', linewidth=2, label='DL')
        ax.plot(last_1h['Timestamp'], last_1h['Predicted_Throughput'], color='white', linestyle='--', alpha=0.5, label='AI Predicted')
        ax.set_title("Download Speed vs AI Forecast", color='white')
        ax.tick_params(colors='white')
        ax.legend(frameon=False, labelcolor='white')
        
        # Highlight Low Throughput
        low_thru = last_1h[last_1h['DL_Throughput_Mbps'] < 500]
        if not low_thru.empty:
             ax.scatter(low_thru['Timestamp'], low_thru['DL_Throughput_Mbps'], color='orange', label='Congestion', zorder=5)
             
        st.pyplot(fig)
        
    with c2:
        st.markdown("### ⚡ Latency Performance (Last 1hr)")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        ax.plot(last_1h['Timestamp'], last_1h['Latency_ms'], color='#FF0055', linewidth=2)
        
        # Anomaly Points
        if not anomalies.empty:
            ax.scatter(anomalies['Timestamp'], anomalies['Latency_ms'], color='red', s=50, label='Anomaly!', zorder=5)
            
        ax.axhline(y=15, color='yellow', linestyle=':', label='Warning (15ms)')
        ax.axhline(y=20, color='red', linestyle='--', label='Critical (20ms)')
        ax.set_title("Latency (ms)", color='white')
        ax.tick_params(colors='white')
        ax.legend(frameon=False, labelcolor='white')
        st.pyplot(fig)
    
    # System Health Status (Moved below graphs)
    st.markdown("---")
    if is_anomaly:
        st.error(f"🚨 CRITICAL ALERT: High Latency Detected ({latest['Latency_ms']:.2f} ms). QoS degraded.")
    elif latest['DL_Throughput_Mbps'] < 1:
        st.info(f"ℹ️ NOTE: Network Idle / Low Traffic ({latest['DL_Throughput_Mbps']:.2f} Mbps).")
    elif latest['DL_Throughput_Mbps'] < 500:
        st.warning(f"⚠️ WARNING: Low Throughput ({latest['DL_Throughput_Mbps']:.2f} Mbps). Congestion likely.")
    else:
        st.success("✅ SYSTEM NORMAL: 6G Slice Operating Optimally.")


# --- Page 2: QoS Forecast ---
elif page == "🚀 QoS Future Forecast":
    st.header("Predictive QoS Assurance")
    st.info("AI Model predicting Downlink Throughput for the next 60 minutes.")
    
    if st.button("Generate Forecast ⚡"):
        model = load_model()
        if model is None:
            st.error("Model missing. Please run train_model.py first.")
            st.stop()
            
        # Recursive Forecasting Logic
        history_window = 100
        current_data = df.tail(history_window).copy()
        
        preds = []
        last_time = current_data['Timestamp'].iloc[-1]
        
        # We need to simulate 'Latency' for the future to feed into the model
        # Logic: Random walk based on last known latency
        last_latency = current_data['Latency_ms'].iloc[-1]
        
        progress = st.progress(0)
        
        for i in range(1, 61): # Predict next 60 minutes
            next_time = last_time + pd.Timedelta(minutes=i)
            
            # Use the UPDATED current_data which includes previous predictions
            recent_thru = current_data['DL_Throughput_Mbps'].values
            
            # Get lag values from the UPDATED history (including predictions)
            lag_1m = recent_thru[-1] if len(recent_thru) >= 1 else 0.1
            lag_5m = recent_thru[-5] if len(recent_thru) >= 5 else lag_1m
            lag_60m = recent_thru[-60] if len(recent_thru) >= 60 else lag_1m
            
            # Rolling stats from recent data
            tail_10 = recent_thru[-10:] if len(recent_thru) >= 10 else recent_thru
            roll_mean_10m = np.mean(tail_10)
            roll_std_10m = np.std(tail_10) if len(tail_10) > 1 else 0
            
            # Simulate Latency with random walk
            next_latency = max(1, last_latency + np.random.normal(0, 0.5))
            last_latency = next_latency
            
            # Input Vector (must match training features exactly)
            X_input = pd.DataFrame([{
                'Hour': next_time.hour,
                'Minute': next_time.minute,
                'DayOfWeek': next_time.dayofweek,
                'Lag_1m': lag_1m,
                'Lag_5m': lag_5m,
                'Lag_60m': lag_60m,
                'Rolling_Mean_10m': roll_mean_10m,
                'Rolling_Std_10m': roll_std_10m,
                'Lag_Latency': next_latency
            }])
            
            # Predict
            pred_throughput = model.predict(X_input)[0]
            
            # Ensure minimum throughput (prevent zero loop)
            pred_throughput = max(pred_throughput, 0.01)
            
            # Append prediction to history for next iteration
            new_row = pd.DataFrame([{
                'Timestamp': next_time,
                'DL_Throughput_Mbps': pred_throughput,
                'Latency_ms': next_latency
            }])
            current_data = pd.concat([current_data, new_row], ignore_index=True)
            
            preds.append({'Timestamp': next_time, 'Forecast_Throughput': pred_throughput, 'Forecast_Latency': next_latency})
            progress.progress(i/60)
            
        forecast_df = pd.DataFrame(preds)
        
        # Stitching
        last_hist = df.tail(1)[['Timestamp', 'DL_Throughput_Mbps']].rename(columns={'DL_Throughput_Mbps': 'Forecast_Throughput'})
        last_hist['Forecast_Latency'] = df.tail(1)['Latency_ms']
        # Concat for plotting
        plot_df = pd.concat([last_hist, forecast_df], ignore_index=True)
        
        # Plotting
        st.subheader("Throughput Forecast (Next 60 Mins)")
        
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        
        # History Context
        hist_ctx = df.tail(30)
        ax.plot(hist_ctx['Timestamp'], hist_ctx['DL_Throughput_Mbps'], color='gray', alpha=0.5, label='Past 30m')
        
        # Forecast
        ax.plot(plot_df['Timestamp'], plot_df['Forecast_Throughput'], color='#00FFAA', linewidth=2, label='AI Forecast')
        
        # Confidence
        ax.fill_between(plot_df['Timestamp'], 
                        plot_df['Forecast_Throughput']*0.9, 
                        plot_df['Forecast_Throughput']*1.1, 
                        color='#00FFAA', alpha=0.1)
        
        ax.set_title("Predicted Downlink Speed (Gbps)", color='white')
        ax.tick_params(colors='white')
        ax.legend(frameon=False, labelcolor='white')
        ax.grid(True, alpha=0.1)
        st.pyplot(fig)
        
        st.success("Successfully predicted Traffic Load for network slice.")
