"""
Streamlit Dashboard for Multi-Sensor IoT Predictive Maintenance
Real-time visualization of 6 sensor channels, anomaly detection,
sensor fusion health scoring, and actuator control.

Run with: streamlit run iot_dashboard.py

Features:
- 6-channel live sensor monitoring (DHT11, thermistor, sound, light, flame)
- Cross-validation display (DHT11 vs thermistor)
- LSTM autoencoder anomaly detection with per-sensor breakdown
- Sensor fusion composite health score
- Actuator control panel (buzzer, RGB LED, relay)
- Simulated data mode for testing without hardware
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

# Serial communication
import serial
import serial.tools.list_ports

# Import custom modules
try:
    from iot_health_score import SensorHealthMonitor, FaultType, AlertLevel
    from iot_lstm_model import LSTMAutoencoder
    from iot_preprocessing import SensorPreprocessor
    import config
    LSTM_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some modules not available: {e}. Running in statistical-only mode.")
    LSTM_AVAILABLE = False
    try:
        from iot_health_score import SensorHealthMonitor, FaultType, AlertLevel
        import config
    except ImportError as e2:
        st.error(f"Core modules not found: {e2}")
        st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="IoT Multi-Sensor Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #00c853, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .alert-critical {
        background: linear-gradient(135deg, #d32f2f, #b71c1c);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.05rem;
    }
    .alert-warning {
        background: linear-gradient(135deg, #f57c00, #e65100);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.05rem;
    }
    .alert-normal {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.05rem;
    }
    .connected-badge {
        background-color: #00c853;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .disconnected-badge {
        background-color: #ff1744;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .sensor-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    div[data-testid="column"] {
        background: rgba(30,30,46,0.4);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }
    div[data-testid="column"]:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        background: rgba(45,45,68,0.6);
        z-index: 10;
    }
    .local-alert {
        padding: 5px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
        animation: pulse 2s infinite;
    }
    .local-alert-critical { background-color: #d32f2f; color: white; }
    .local-alert-warning { background-color: #f57c00; color: white; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_ports():
    """List available COM ports."""
    ports = serial.tools.list_ports.comports()
    return {f"{p.device} - {p.description}": p.device for p in ports}

def parse_serial_line(line: str) -> Optional[dict]:
    """
    Parse multi-sensor serial line from Arduino.
    Format: temp_dht,humidity,temp_therm,sound_level,light_level,flame_intensity
    """
    line = line.strip()
    if not line or line == "ERROR" or line.startswith("MULTI_SENSOR_READY") or line.startswith("ACK:") or line.startswith("STATUS:"):
        return None
    try:
        parts = line.split(",")
        
        if len(parts) == 6:
            reading = {
                'timestamp': datetime.now(),
                'temp_dht': float(parts[0]),
                'humidity': float(parts[1]),
                'temp_therm': float(parts[2]),
                'sound_level': float(parts[3]),
                'light_level': float(parts[4]),
                'flame_intensity': float(parts[5]),
            }
        elif len(parts) == 2:
            temp = float(parts[0])
            hum = float(parts[1])
            reading = {
                'timestamp': datetime.now(),
                'temp_dht': temp,
                'humidity': hum,
                'temp_therm': temp + np.random.randn() * 0.3,
                'sound_level': 45 + np.random.randn() * 4,
                'light_level': 500 + np.random.randn() * 15,
                'flame_intensity': np.nan, # simulate dynamic drop
            }
        else:
            return None
        
        # Validate ranges (assign NaN instead of dropping entire row)
        for feat in config.FEATURE_NAMES:
            if feat in reading:
                lo, hi = config.SENSOR_RANGES.get(feat, (0, 1023))
                val = reading[feat]
                if pd.isna(val) or not (lo <= val <= hi):
                    reading[feat] = np.nan
        
        return reading
    except (ValueError, IndexError):
        return None
    try:
        parts = line.split(",")
        
        if len(parts) == 6:
            reading = {
                'timestamp': datetime.now(),
                'temp_dht': float(parts[0]),
                'humidity': float(parts[1]),
                'temp_therm': float(parts[2]),
                'sound_level': float(parts[3]),
                'light_level': float(parts[4]),
                'flame_intensity': float(parts[5]),
            }
        elif len(parts) == 2:
            # Legacy 2-field format
            temp = float(parts[0])
            hum = float(parts[1])
            reading = {
                'timestamp': datetime.now(),
                'temp_dht': temp,
                'humidity': hum,
                'temp_therm': temp + np.random.randn() * 0.3,
                'sound_level': 45 + np.random.randn() * 4,
                'light_level': 500 + np.random.randn() * 15,
                'flame_intensity': 8 + abs(np.random.randn() * 2),
            }
        else:
            return None
        
        # Validate ranges
        for feat in config.FEATURE_NAMES:
            if feat in reading:
                lo, hi = config.SENSOR_RANGES[feat]
                if not (lo <= reading[feat] <= hi):
                    return None
        
        return reading
    except (ValueError, IndexError):
        return None

def send_actuator_command(serial_port, command: str):
    """Send a command to Arduino actuators."""
    try:
        if serial_port and serial_port.is_open:
            serial_port.write(f"{command}\n".encode())
            return True
    except Exception:
        pass
    return False

def compute_anomaly_scores_statistical(df: pd.DataFrame) -> np.ndarray:
    """Compute anomaly scores using statistical z-score method."""
    scores = np.zeros(len(df))
    if len(df) < config.STAT_MIN_SAMPLES:
        return scores
    
    window = config.STAT_ROLLING_WINDOW
    n_features = 0
    
    for col in config.FEATURE_NAMES:
        if col not in df.columns:
            continue
        n_features += 1
        rolling_mean = df[col].rolling(window=window, min_periods=config.STAT_MIN_SAMPLES).mean()
        rolling_std = df[col].rolling(window=window, min_periods=config.STAT_MIN_SAMPLES).std()
        rolling_std = rolling_std.replace(0, 0.01)
        z_scores = np.abs((df[col] - rolling_mean) / rolling_std)
        z_scores = z_scores.fillna(0).values
        scores += z_scores
    
    if n_features > 0:
        scores /= n_features
    return scores

def compute_anomaly_scores_lstm(df: pd.DataFrame, model, preprocessor) -> np.ndarray:
    """Compute anomaly scores using the trained LSTM Autoencoder."""
    scores = np.zeros(len(df))
    if len(df) < config.WINDOW_SIZE:
        return scores
    
    try:
        feature_cols = [c for c in preprocessor.feature_columns if c in df.columns]
        feature_df = df[feature_cols].copy()
        
        # Spoof missing channels with exact scaler medians
        for i, col in enumerate(preprocessor.feature_columns):
            if col in feature_df.columns:
                if feature_df[col].isna().all() or len(feature_df) == 0:
                    midpoint = preprocessor.scaler.data_min_[i] + (preprocessor.scaler.data_max_[i] - preprocessor.scaler.data_min_[i]) / 2.0
                    feature_df[col] = midpoint
                else:
                    feature_df[col] = feature_df[col].ffill().bfill().fillna(0)
                    
        feature_scaled = preprocessor.scaler.transform(feature_df)
        
        windows = []
        for i in range(len(feature_scaled) - config.WINDOW_SIZE + 1):
            windows.append(feature_scaled[i:i + config.WINDOW_SIZE])
        
        if len(windows) == 0:
            return scores
        
        X = np.array(windows)
        errors = model.compute_reconstruction_error(X, per_sample=True)
        
        offset = config.WINDOW_SIZE - 1
        scores[offset:offset + len(errors)] = errors
        if len(errors) > 0:
            scores[:offset] = errors[0]
    except Exception as e:
        scores = compute_anomaly_scores_statistical(df)
    
    return scores

@st.cache_resource
def load_lstm_system():
    """Load trained LSTM model and preprocessor (cached)."""
    try:
        preprocessor = SensorPreprocessor()
        preprocessor.load(config.DEFAULT_PREPROCESSOR_PATH)
        
        model = LSTMAutoencoder(
            n_features=len(preprocessor.feature_columns)
        )
        model.load(config.DEFAULT_MODEL_PATH)
        
        threshold = np.load(config.DEFAULT_THRESHOLD_PATH)
        model.threshold = threshold
        
        return model, preprocessor, float(threshold)
    except Exception as e:
        st.warning(f"Could not load LSTM model: {e}")
        return None, None, None

def generate_simulated_reading(index: int) -> dict:
    """Generate simulated multi-sensor reading with fault injection."""
    base = {
        'temp_dht': 22.0,
        'humidity': 55.0,
        'temp_therm': 22.0,
        'sound_level': 45.0,
        'light_level': 500.0,
        'flame_intensity': 8.0,
    }
    
    # Normal variation
    reading = {
        'timestamp': datetime.now(),
        'temp_dht': base['temp_dht'] + np.random.normal(0, 0.3),
        'humidity': base['humidity'] + np.random.normal(0, 1.5),
        'temp_therm': base['temp_dht'] + np.random.normal(0.1, 0.25),
        'sound_level': base['sound_level'] + np.random.normal(0, 4),
        'light_level': base['light_level'] + np.random.normal(0, 15),
        'flame_intensity': base['flame_intensity'] + abs(np.random.normal(0, 2)),
    }
    
    # Inject faults after baseline period
    if index > 60 and index % 150 < 15:
        # Temperature drift episode
        reading['temp_dht'] += (index % 150) * 0.3
    
    if index > 100 and index % 200 < 10:
        # Sensor divergence episode
        reading['temp_therm'] += 7.0
    
    if index > 140 and index % 180 < 12:
        # Acoustic anomaly
        reading['sound_level'] += np.random.uniform(80, 150)
    
    if index > 200 and index % 250 < 8:
        # Fire hazard simulation
        reading['flame_intensity'] = 600 + np.random.normal(0, 30)
        reading['temp_dht'] += 5
        reading['temp_therm'] += 4.5
    
    if index > 250 and index % 300 < 6:
        # Light anomaly
        reading['light_level'] += 200 * np.sin(index * 0.5)
    
    # Clamp to valid ranges
    for feat in config.FEATURE_NAMES:
        lo, hi = config.SENSOR_RANGES[feat]
        reading[feat] = max(lo, min(hi, round(reading[feat], 2)))
    
    return reading

# ============================================================================
# CHART BUILDERS
# ============================================================================

def create_single_sensor_gauge(val: float, feat: str, min_val: float, max_val: float) -> go.Figure:
    color = config.SENSOR_COLORS.get(feat, '#bb86fc')
    name = config.FEATURE_LABELS.get(feat, feat)
    unit = config.FEATURE_UNITS.get(feat, '')
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = val,
        number = {'suffix': ' ' + unit, 'font': {'size': 32, 'color': 'white'}},
        title = {'text': name, 'font': {'size': 14, 'color': '#aaa'}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickcolor': '#555'},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': '#333',
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_single_sensor_timeseries(df: pd.DataFrame, feat: str, max_points: int = 200) -> go.Figure:
    display_df = df.tail(max_points)
    color = config.SENSOR_COLORS.get(feat, '#bb86fc')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df['timestamp'], y=display_df[feat],
        mode='lines', name=feat,
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=color.replace(')', ',0.1)').replace('rgb', 'rgba') if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)"
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        hovermode='x unified'
    )
    return fig


def create_multi_sensor_timeseries(df: pd.DataFrame, max_points: int = 200) -> go.Figure:
    """Create 6-channel live sensor time-series chart."""
    display_df = df.tail(max_points)
    
    # 3 rows x 2 cols layout
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            config.FEATURE_LABELS.get(f, f) for f in config.FEATURE_NAMES
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2)]
    
    for i, feat in enumerate(config.FEATURE_NAMES):
        if feat not in display_df.columns:
            continue
        row, col = positions[i]
        color = config.SENSOR_COLORS.get(feat, '#ffffff')
        
        fig.add_trace(
            go.Scatter(
                x=display_df['timestamp'], y=display_df[feat],
                mode='lines', name=feat,
                line=dict(color=color, width=2),
                fill='tozeroy',
                fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=550,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=30, b=30),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    return fig

def create_cross_validation_chart(df: pd.DataFrame, max_points: int = 200) -> go.Figure:
    """Create DHT11 vs Thermistor cross-validation chart."""
    display_df = df.tail(max_points)
    
    fig = go.Figure()
    
    if 'temp_dht' in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df['timestamp'], y=display_df['temp_dht'],
            mode='lines', name='DHT11 Temp',
            line=dict(color='#ff6b6b', width=2.5)
        ))
    
    if 'temp_therm' in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df['timestamp'], y=display_df['temp_therm'],
            mode='lines', name='Thermistor Temp',
            line=dict(color='#ff9f43', width=2.5)
        ))
    
    # Show divergence threshold band
    if 'temp_dht' in display_df.columns:
        fig.add_trace(go.Scatter(
            x=display_df['timestamp'],
            y=display_df['temp_dht'] + config.TEMP_CROSS_VALIDATION_MAX_DIFF,
            mode='lines', name='Max Divergence',
            line=dict(color='rgba(255,71,87,0.3)', width=1, dash='dash'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=display_df['timestamp'],
            y=display_df['temp_dht'] - config.TEMP_CROSS_VALIDATION_MAX_DIFF,
            mode='lines', name='Max Divergence',
            line=dict(color='rgba(255,71,87,0.3)', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(255,71,87,0.05)',
            showlegend=False
        ))
    
    fig.update_layout(
        title="🔄 Temperature Cross-Validation (DHT11 vs Thermistor)",
        height=280,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                     title_text='Temperature (°C)')
    
    return fig

def create_health_gauge(health_score: float) -> go.Figure:
    """Create health score gauge chart."""
    if health_score >= 80:
        bar_color = "#00c853"
    elif health_score >= 50:
        bar_color = "#ff9800"
    else:
        bar_color = "#ff1744"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        number={'suffix': '%', 'font': {'size': 40, 'color': 'white'}},
        title={'text': "System Health", 'font': {'size': 18, 'color': '#aaa'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#555'},
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': '#1e1e2e',
            'bordercolor': '#333',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,23,68,0.15)'},
                {'range': [50, 80], 'color': 'rgba(255,152,0,0.15)'},
                {'range': [80, 100], 'color': 'rgba(0,200,83,0.15)'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.8,
                'value': health_score
            }
        }
    ))
    
    fig.update_layout(
        height=260,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=60, b=20)
    )
    return fig

def create_anomaly_chart(df: pd.DataFrame, threshold: float, max_points: int = 200) -> go.Figure:
    """Create anomaly score timeline."""
    display_df = df.tail(max_points)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=display_df['timestamp'], y=display_df['anomaly_score'],
        mode='lines', name='Anomaly Score',
        line=dict(color='#bb86fc', width=2),
        fill='tozeroy', fillcolor='rgba(187,134,252,0.15)'
    ))
    
    fig.add_trace(go.Scatter(
        x=display_df['timestamp'],
        y=[threshold] * len(display_df),
        mode='lines', name='Threshold',
        line=dict(color='#ff1744', width=2, dash='dash')
    ))
    
    anomalies = display_df[display_df['anomaly_score'] > threshold]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'], y=anomalies['anomaly_score'],
            mode='markers', name='Anomaly',
            marker=dict(color='#ff1744', size=8, symbol='x',
                       line=dict(width=2, color='white'))
        ))
    
    fig.update_layout(
        title="📊 Multi-Sensor Anomaly Detection",
        height=280,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    return fig

def create_health_timeline(df: pd.DataFrame, max_points: int = 200) -> go.Figure:
    """Create health score timeline."""
    display_df = df.tail(max_points)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=display_df['timestamp'], y=display_df['health_score'],
        mode='lines', name='Health Score',
        line=dict(color='#64b5f6', width=2.5),
        fill='tozeroy', fillcolor='rgba(100,181,246,0.1)'
    ))
    
    fig.add_hline(y=80, line_dash="dash", line_color="rgba(0,200,83,0.6)",
                  annotation_text="Healthy", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,152,0,0.6)",
                  annotation_text="Warning", annotation_position="right")
    
    fig.update_layout(
        title="🫀 Composite Health Score",
        height=280,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_range=[0, 105],
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified'
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    return fig

def create_sensor_correlation_heatmap(df: pd.DataFrame, max_points: int = 200) -> go.Figure:
    """Create real-time sensor correlation heatmap."""
    display_df = df.tail(max_points)
    
    available = [c for c in config.FEATURE_NAMES if c in display_df.columns]
    if len(available) < 2:
        return go.Figure()
    
    corr = display_df[available].corr()
    
    labels = [config.FEATURE_LABELS.get(c, c).split(' ')[-1] for c in available]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 11, "color": "white"},
        hovertemplate='%{x} vs %{y}: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🔗 Sensor Correlation",
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=80, r=20, t=50, b=40),
        xaxis=dict(tickangle=30)
    )
    
    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================


@st.dialog("Sensor Analytics", width="large")
def sensor_lightbox(df: pd.DataFrame, feat: str):
    st.subheader(config.FEATURE_LABELS.get(feat, feat))
    fig = create_single_sensor_timeseries(df, feat, max_points=500)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical Min", f"{df[feat].min():.1f}")
    c2.metric("Historical Mean", f"{df[feat].mean():.1f}")
    c3.metric("Historical Max", f"{df[feat].max():.1f}")

def main():
    """Main dashboard application."""
    
    # Header moved to sidebar
    
    # ========================================================================
    # SESSION STATE INITIALIZATION
    # ========================================================================
    
    all_cols = ['timestamp'] + config.FEATURE_NAMES + [
        'anomaly_score', 'health_score', 'fault_type', 'alert_level'
    ]
    
    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = pd.DataFrame(
            {col: pd.Series(dtype='float64') for col in config.FEATURE_NAMES} |
            {'timestamp': pd.Series(dtype='datetime64[ns]'),
             'anomaly_score': pd.Series(dtype='float64'),
             'health_score': pd.Series(dtype='float64'),
             'fault_type': pd.Series(dtype='str'),
             'alert_level': pd.Series(dtype='str')}
        )
    if 'serial_connected' not in st.session_state:
        st.session_state.serial_connected = False
    if 'serial_port' not in st.session_state:
        st.session_state.serial_port = None
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'sim_mode' not in st.session_state:
        st.session_state.sim_mode = False
    if 'sim_index' not in st.session_state:
        st.session_state.sim_index = 0
    if 'monitor' not in st.session_state:
        st.session_state.monitor = SensorHealthMonitor(
            ema_alpha=config.EMA_ALPHA,
            drift_window=config.DRIFT_WINDOW,
            noise_window=config.NOISE_WINDOW,
            freeze_threshold=config.FREEZE_THRESHOLD
        )
    if 'lstm_model' not in st.session_state:
        st.session_state.lstm_model = None
        st.session_state.lstm_preprocessor = None
        st.session_state.lstm_threshold = None
        if LSTM_AVAILABLE:
            model, preprocessor, threshold = load_lstm_system()
            if model is not None:
                st.session_state.lstm_model = model
                st.session_state.lstm_preprocessor = preprocessor
                st.session_state.lstm_threshold = threshold
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown('<p class="main-header" style="font-size:1.5rem">🔧 IoT Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="font-size:0.8rem">Multi-Sensor Predictive Maintenance</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.header("🔌 Connection")
        
        available_ports = get_available_ports()
        
        if available_ports:
            selected_label = st.selectbox(
                "Serial Port",
                options=list(available_ports.keys()),
                help="Select the COM port your Arduino is connected to"
            )
            selected_port = available_ports[selected_label]
        else:
            st.warning("No serial ports detected")
            selected_port = None
        
        if st.button("🔄 Refresh Ports", use_container_width=True):
            st.rerun()
        
        baud_rate = st.selectbox("📡 Baud Rate", options=[9600, 115200], index=0)
        
        st.markdown("---")
        
        # Connection Controls
        st.subheader("Controls")
        col1, col2 = st.columns(2)
        with col1:
            connect_btn = st.button("🟢 Connect", use_container_width=True,
                                     disabled=st.session_state.serial_connected or not selected_port)
        with col2:
            disconnect_btn = st.button("🔴 Disconnect", use_container_width=True,
                                        disabled=not st.session_state.serial_connected)
        
        if connect_btn and selected_port:
            try:
                ser = serial.Serial(selected_port, baud_rate, timeout=config.SERIAL_TIMEOUT)
                st.session_state.serial_port = ser
                st.session_state.serial_connected = True
                st.session_state.streaming = True
                st.session_state.sim_mode = False
                time.sleep(2)
                st.rerun()
            except serial.SerialException as e:
                st.error(f"Connection failed: {e}")
        
        if disconnect_btn:
            if st.session_state.serial_port:
                try:
                    st.session_state.serial_port.close()
                except:
                    pass
            st.session_state.serial_port = None
            st.session_state.serial_connected = False
            st.session_state.streaming = False
            st.rerun()
        
        # Sensor Override Control
        st.markdown("---")
        st.subheader("🎛️ Active Modules")
        
        inv_labels = {v: k for k, v in config.FEATURE_LABELS.items()}
        selected_labels = st.multiselect(
            "Select Connected Hardware:",
            options=list(config.FEATURE_LABELS.values()),
            default=[],
            help="Uncheck missing sensors to safely drop their gauges and spoof their ML outputs"
        )
        st.session_state.selected_sensors = [inv_labels[label] for label in selected_labels]

        # Simulation Mode
        st.markdown("---")
        st.subheader("🧪 Test Mode")
        
        if st.button("▶️ Start Simulation", use_container_width=True,
                     disabled=st.session_state.serial_connected):
            st.session_state.sim_mode = True
            st.session_state.streaming = True
            st.session_state.sim_index = 0
            st.rerun()
        
        if st.button("⏹️ Stop Simulation", use_container_width=True,
                     disabled=not st.session_state.sim_mode):
            st.session_state.sim_mode = False
            st.session_state.streaming = False
            st.rerun()
        
        # Actuator Control Panel
        if st.session_state.serial_connected:
            st.markdown("---")
            st.subheader("🎛️ Actuator Control")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔔 Buzzer ON", use_container_width=True):
                    send_actuator_command(st.session_state.serial_port, "BUZZ:ON")
                if st.button("🟢 LED Green", use_container_width=True):
                    send_actuator_command(st.session_state.serial_port, "LED:GREEN")
            with col_b:
                if st.button("🔕 Buzzer OFF", use_container_width=True):
                    send_actuator_command(st.session_state.serial_port, "BUZZ:OFF")
                if st.button("🔴 LED Red", use_container_width=True):
                    send_actuator_command(st.session_state.serial_port, "LED:RED")
        
        # Reset
        st.markdown("---")
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.data_buffer = pd.DataFrame(
                columns=['timestamp'] + config.FEATURE_NAMES + [
                    'anomaly_score', 'health_score', 'fault_type', 'alert_level'
                ]
            )
            st.session_state.sim_index = 0
            st.rerun()
        
        # Status
        st.markdown("---")
        st.subheader("📡 Status")
        
        if st.session_state.serial_connected:
            st.markdown('<span class="connected-badge">● Connected</span>', unsafe_allow_html=True)
            st.caption(f"Port: {selected_port} @ {baud_rate} baud")
        elif st.session_state.sim_mode:
            st.markdown('<span class="connected-badge">● Simulating</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="disconnected-badge">● Disconnected</span>', unsafe_allow_html=True)
        
        # Sensor count
        st.caption(f"📊 Sensors: {config.N_FEATURES} channels")
        
        if st.session_state.lstm_model is not None:
            st.success("🧠 LSTM Model Active")
        else:
            st.info("📊 Statistical Mode")
        
        st.metric("Samples", len(st.session_state.data_buffer))
    
    # ========================================================================
    # DATA ACQUISITION & LIVE DISPLAY
    # ========================================================================
    
    @st.fragment(run_every=1.5 if st.session_state.streaming else None)
    def _live_dashboard_fragment():        
        reading = None
        
        if st.session_state.streaming:
            if st.session_state.serial_connected and st.session_state.serial_port:
                try:
                    raw_line = st.session_state.serial_port.readline().decode('utf-8', errors='ignore')
                    if raw_line.strip():
                        reading = parse_serial_line(raw_line)
                except (serial.SerialException, OSError):
                    st.session_state.serial_connected = False
                    st.session_state.streaming = False
                    st.error("Serial connection lost!")
            
            elif st.session_state.sim_mode:
                reading = generate_simulated_reading(st.session_state.sim_index)
                st.session_state.sim_index += 1
        
        # Process new reading
        if reading:
            df = st.session_state.data_buffer
            new_row = pd.DataFrame([reading])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Compute anomaly scores
            if st.session_state.lstm_model is not None:
                anomaly_scores = compute_anomaly_scores_lstm(
                    df, st.session_state.lstm_model, st.session_state.lstm_preprocessor
                )
                threshold = st.session_state.lstm_threshold
            else:
                anomaly_scores = compute_anomaly_scores_statistical(df)
                threshold = config.STAT_ZSCORE_THRESHOLD
            
            df['anomaly_score'] = anomaly_scores
            
            # Compute health scores
            monitor = st.session_state.monitor
            health_scores = monitor.anomaly_score_to_health(anomaly_scores, threshold)
            df['health_score'] = health_scores
            
            # Fault classification with confirmation window
            if len(df) >= config.FAULT_MIN_SAMPLES:
                recent_scores = anomaly_scores[-config.FAULT_CONFIRM_WINDOW:]
                anomaly_count = sum(1 for s in recent_scores if s > threshold)
                anomaly_ratio = anomaly_count / len(recent_scores)
                
                if anomaly_ratio >= config.FAULT_CONFIRM_RATIO:
                    fault_type = monitor.classify_fault(df, anomaly_scores[-1], threshold)
                    alert_level = monitor.determine_alert_level(health_scores[-1], fault_type)
                    
                    # Auto-actuator response (if connected)
                    if st.session_state.serial_connected and st.session_state.serial_port:
                        if alert_level == AlertLevel.CRITICAL:
                            send_actuator_command(st.session_state.serial_port, "BUZZ:ON")
                            send_actuator_command(st.session_state.serial_port, "LED:RED")
                        elif alert_level == AlertLevel.WARNING:
                            send_actuator_command(st.session_state.serial_port, "LED:YELLOW")
                        else:
                            send_actuator_command(st.session_state.serial_port, "BUZZ:OFF")
                            send_actuator_command(st.session_state.serial_port, "LED:GREEN")
                else:
                    fault_type = FaultType.HEALTHY
                    alert_level = AlertLevel.NORMAL
            else:
                fault_type = FaultType.HEALTHY
                alert_level = AlertLevel.NORMAL
            
            df.at[df.index[-1], 'fault_type'] = fault_type.value
            df.at[df.index[-1], 'alert_level'] = alert_level.value
            
            df['fault_type'] = df['fault_type'].fillna(FaultType.HEALTHY.value)
            df['alert_level'] = df['alert_level'].fillna(AlertLevel.NORMAL.value)
            
            if len(df) > config.MAX_DISPLAY_SAMPLES * 2:
                df = df.tail(config.MAX_DISPLAY_SAMPLES).reset_index(drop=True)
            
            st.session_state.data_buffer = df
        
        # ========================================================================
        # DISPLAY
        # ========================================================================
        
        df = st.session_state.data_buffer
        
        if len(df) == 0:
            if st.session_state.streaming:
                st.info("⏳ Waiting for multi-sensor data...")
                time.sleep(config.SERIAL_READ_INTERVAL)
                st.rerun()
            else:
                st.info("👆 Connect to Arduino or start simulation to begin monitoring")
                
                with st.expander("📖 Quick Start Guide"):
                    st.markdown("""
                    **With Arduino (Multi-Sensor Array):**
                    1. Upload `arduino/multi_sensor/multi_sensor.ino` to your Arduino
                    2. Wire all 5 sensors and 3 actuators per the pin diagram
                    3. Select the COM port in the sidebar and click **Connect**
                    
                    **Sensors Connected:**
                    - 🌡️ DHT11 (Temperature + Humidity) → Pin 7
                    - 🔥 Thermistor → Analog A0 (with 10K resistor)
                    - 🔊 Sound Sensor → Analog A1
                    - 💡 LDR → Analog A2 (with 10K resistor)
                    - 🔥 Flame Sensor → Analog A3 + Digital Pin 4
                    
                    **Actuators:**
                    - 🔔 Buzzer → Pin 8
                    - 🚦 RGB LED → Pins 9, 10, 11
                    - ⚡ Relay → Pin 12
                    
                    **Without Arduino (Test Mode):**
                    Click **Start Simulation** — generates realistic multi-sensor data with fault injection!
                    """)
            st.stop()
        
        # --- Current Status Metrics ---
        latest = df.iloc[-1] if len(df) > 0 else pd.Series()
        
        def get_sensor_alert(feat: str):
            if len(df) > 10 and feat in latest and not pd.isna(latest[feat]):
                mean = df[feat].mean()
                std = df[feat].std() + 0.01
                z = abs(latest[feat] - mean) / std
                if z > 3:
                    return "Critical: High Deviation"
            return None
        # Determine connected sensors based on last 5 readings
        connected_sensors = []
        if len(df) > 0:
            recent_df = df.tail(5)
            for feat in config.FEATURE_NAMES:
                if feat in recent_df.columns and recent_df[feat].notna().any():
                    connected_sensors.append(feat)
        # -------------------------------------------------------------
        # ROW 1: System Diagnostics
        # -------------------------------------------------------------
        st.subheader("🌐 System Diagnostics")
        cols_sys = st.columns([1, 2])
        
        with cols_sys[0]:
            health = latest.get('health_score', 100) if not latest.empty else 100
            fault = latest.get('fault_type', 'Healthy') if not latest.empty else 'Healthy'
            alert = latest.get('alert_level', 'Normal') if not latest.empty else 'Normal'
            
            if alert == 'Critical':
                st.markdown(f'<div class="alert-critical">🚨 {fault}</div>', unsafe_allow_html=True)
            elif alert == 'Warning':
                st.markdown(f'<div class="alert-warning">⚠️ {fault}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-normal">✅ Systems Nominal</div>', unsafe_allow_html=True)
                
            fig_health = create_health_gauge(health)
            st.plotly_chart(fig_health, use_container_width=True, key="sys_health_gauge")
            
            healthy_pct = (df['alert_level'] == 'Normal').sum() / len(df) * 100 if len(df) > 0 else 100
            st.metric("System Uptime", f"{healthy_pct:.1f}%")
        with cols_sys[1]:
            active_threshold = st.session_state.lstm_threshold if st.session_state.lstm_model else config.STAT_ZSCORE_THRESHOLD
            anomaly_count = int((df['anomaly_score'] > active_threshold).sum()) if 'anomaly_score' in df.columns else 0
            
            st.markdown(f"**Anomaly History** (Total Detected: {anomaly_count})")
            if len(df) > 0:
                fig_anomaly = create_anomaly_chart(df, active_threshold, max_points=300)
                fig_anomaly.update_layout(height=350) 
                st.plotly_chart(fig_anomaly, use_container_width=True, key="anomaly_main")
        st.markdown("---")
        # -------------------------------------------------------------
        # ROW 2+: Sensor Network Data (Dynamic Wrapping Layout)
        # -------------------------------------------------------------
        st.subheader("📊 Sensor Network Data")
        if not connected_sensors:
            st.warning("No sensors connected or reporting valid data.")
        else:
            for i in range(0, len(connected_sensors), 3):
                cols_r = st.columns(3)
                for j in range(3):
                    if i + j < len(connected_sensors):
                        feat = connected_sensors[i + j]
                        with cols_r[j]:
                            alert_msg = get_sensor_alert(feat)
                            if alert_msg:
                                st.markdown(f'<div class="local-alert local-alert-critical">⚠️ {alert_msg}</div>', unsafe_allow_html=True)
                            
                            val = latest.get(feat, 0)
                            min_val, max_val = config.SENSOR_RANGES.get(feat, (0, 100))
                            
                            fig = create_single_sensor_gauge(val, feat, min_val, max_val)
                            st.plotly_chart(fig, use_container_width=True, key=f"gauge_{feat}_{i}_{j}")
                            
                            if st.button(f"🔍 View Details", key=f"btn_modal_{feat}", use_container_width=True):
                                sensor_lightbox(df, feat)
    
    _live_dashboard_fragment()

if __name__ == "__main__":
    main()
