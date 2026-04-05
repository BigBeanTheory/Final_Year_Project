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

# Serial communication (optional — not available on cloud/Render)
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

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
    /* Global font scaling */
    html, body, [class*="css"] { font-size: 13px !important; }
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #00c853, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #888; font-size: 0.8rem; margin-top: -8px; }
    .alert-critical {
        background: linear-gradient(135deg, #d32f2f, #b71c1c);
        color: white; padding: 10px 16px; border-radius: 8px;
        font-weight: bold; font-size: 0.9rem;
    }
    .alert-warning {
        background: linear-gradient(135deg, #f57c00, #e65100);
        color: white; padding: 10px 16px; border-radius: 8px;
        font-weight: bold; font-size: 0.9rem;
    }
    .alert-normal {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white; padding: 10px 16px; border-radius: 8px;
        font-weight: bold; font-size: 0.9rem;
    }
    .connected-badge {
        background-color: #00c853; color: white;
        padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }
    .disconnected-badge {
        background-color: #ff1744; color: white;
        padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }
    div[data-testid="column"] {
        background: rgba(30,30,46,0.4);
        padding: 10px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="column"]:hover {
        transform: scale(1.015);
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
        background: rgba(45,45,68,0.5);
    }
    .local-alert {
        padding: 4px 8px; border-radius: 6px;
        font-size: 0.7rem; font-weight: bold;
        text-align: center; margin-bottom: 4px;
        animation: pulse 2s infinite;
    }
    .local-alert-critical { background-color: #d32f2f; color: white; }
    .local-alert-warning { background-color: #f57c00; color: white; }
    @keyframes pulse {
        0% { opacity: 1; } 50% { opacity: 0.8; } 100% { opacity: 1; }
    }
    /* Smaller Streamlit metrics */
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
    /* Smaller subheaders */
    .stMarkdown h2, .stMarkdown h3 { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_ports():
    """List available COM ports."""
    if not SERIAL_AVAILABLE:
        return {}
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
    """Compute anomaly scores using the trained LSTM Autoencoder.
    
    Also stores per-feature reconstruction errors in session state so the
    dashboard can show exactly which sensor is driving any anomaly.
    """
    scores = np.zeros(len(df))
    if len(df) < config.WINDOW_SIZE:
        return scores
    
    try:
        feature_cols = [c for c in preprocessor.feature_columns if c in df.columns]
        feature_df = df[feature_cols].copy()
        
        # Spoof / impute missing channels
        for i, col in enumerate(preprocessor.feature_columns):
            if col in feature_df.columns:
                if feature_df[col].isna().all() or len(feature_df) == 0:
                    midpoint = (preprocessor.scaler.data_min_[i] +
                                preprocessor.scaler.data_max_[i]) / 2.0
                    feature_df[col] = midpoint
                else:
                    feature_df[col] = feature_df[col].ffill().bfill().fillna(0)
        
        # Scale and CLIP to [0,1] so out-of-range live readings don't spike
        # the reconstruction error artificially (see iot_preprocessing.py fix)
        feature_scaled = np.clip(preprocessor.scaler.transform(feature_df), 0.0, 1.0)
        
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
        
        # ---- Per-feature error for the most recent window ----
        # This is what the fault-localization panel uses to answer
        # "which sensor is causing the anomaly?"
        if len(X) > 0:
            last_window = X[-1:]  # shape (1, window, features)
            per_feat = model.compute_per_feature_error(last_window)
            # per_feat is dict {name: array(1,)} — flatten to scalar
            st.session_state.per_feature_errors = {
                k: float(v[0]) for k, v in per_feat.items()
            }

    except Exception:
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


def create_fault_localization_chart(
    per_feature_errors: dict,
    threshold: float,
    selected_sensors: list
) -> go.Figure:
    """
    Horizontal bar chart showing per-sensor reconstruction error.
    This is the primary answer to 'which sensor is acting up?'
    
    Bars are colored green/orange/red based on their error relative to
    the anomaly threshold.  The threshold line makes it immediately
    obvious which sensors are over the limit.
    """
    feats = [f for f in config.FEATURE_NAMES
             if f in per_feature_errors and f in selected_sensors]
    if not feats:
        return go.Figure()
    
    errors = [per_feature_errors[f] for f in feats]
    labels = [config.FEATURE_LABELS.get(f, f) for f in feats]
    
    # Per-sensor severity color
    colors = []
    for e in errors:
        if e > threshold:
            colors.append('#ff1744')     # red — over threshold
        elif e > threshold * 0.7:
            colors.append('#ff9800')     # orange — approaching
        else:
            colors.append('#00c853')     # green — healthy
    
    # Sort highest error first
    paired = sorted(zip(errors, labels, colors), reverse=True)
    errors, labels, colors = zip(*paired) if paired else ([], [], [])
    
    # Percentage of threshold for hover text
    pct = [f"{e/threshold*100:.0f}% of threshold" for e in errors]
    
    fig = go.Figure(go.Bar(
        x=list(errors),
        y=list(labels),
        orientation='h',
        marker_color=list(colors),
        text=list(pct),
        textposition='outside',
        hovertemplate='%{y}<br>Error: %{x:.5f}<br>%{text}<extra></extra>',
        cliponaxis=False
    ))
    
    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash='dash', line_color='#ff1744', line_width=2,
        annotation_text=f'Threshold ({threshold:.4f})',
        annotation_position='top right',
        annotation_font_color='#ff1744'
    )
    
    fig.update_layout(
        title='🔬 Per-Sensor Fault Localization',
        height=max(180, len(feats) * 45 + 80),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=120, t=50, b=30),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   title='Reconstruction Error (MSE)'),
        yaxis=dict(showgrid=False),
        bargap=0.35
    )
    return fig


def create_sensor_sparkline_with_anomaly(
    df: pd.DataFrame,
    feat: str,
    threshold: float,
    per_feature_errors: dict,
    max_points: int = 150
) -> go.Figure:
    """
    Per-sensor time-series with anomaly regions highlighted in red.
    Shows the actual sensor value AND marks the windows where that
    specific sensor's reconstruction error exceeded the threshold —
    so you immediately see *when* and *how* it deviated.
    """
    display_df = df.tail(max_points).copy()
    color = config.SENSOR_COLORS.get(feat, '#bb86fc')
    unit = config.FEATURE_UNITS.get(feat, '')
    
    fig = go.Figure()
    
    # Main sensor trace
    fig.add_trace(go.Scatter(
        x=display_df['timestamp'],
        y=display_df[feat],
        mode='lines',
        name='Reading',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
        hovertemplate=f'%{{x}}<br>{feat}: %{{y:.2f}} {unit}<extra></extra>'
    ))
    
    # Overlay anomaly markers using the global anomaly_score column
    # (per-feature error not stored per-row, but overall anomaly flags the row)
    if 'anomaly_score' in display_df.columns and threshold > 0:
        anomalous = display_df[display_df['anomaly_score'] > threshold]
        if len(anomalous) > 0:
            fig.add_trace(go.Scatter(
                x=anomalous['timestamp'],
                y=anomalous[feat],
                mode='markers',
                name='Anomaly window',
                marker=dict(
                    color='#ff1744', size=7, symbol='circle',
                    line=dict(width=1.5, color='white')
                ),
                hovertemplate=f'Anomaly detected<br>{feat}: %{{y:.2f}} {unit}<extra></extra>'
            ))
    
    # Current per-feature error as subtitle
    current_err = per_feature_errors.get(feat, 0.0)
    err_pct = current_err / (threshold + 1e-10) * 100
    status = '🔴 OVER' if current_err > threshold else ('🟡' if current_err > threshold * 0.7 else '🟢')
    subtitle = f'{status} Error: {current_err:.4f} ({err_pct:.0f}% of threshold)'
    
    fig.update_layout(
        title=dict(
            text=f"{config.FEATURE_LABELS.get(feat, feat)}<br>"
                 f"<span style='font-size:11px;color:#aaa'>{subtitle}</span>",
            font=dict(size=13)
        ),
        height=200,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   ticksuffix=f' {unit}'),
        showlegend=False,
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
def sensor_lightbox(df: pd.DataFrame, feat: str, threshold: float = 0.0):
    """Detailed per-sensor diagnostic view."""
    label = config.FEATURE_LABELS.get(feat, feat)
    unit = config.FEATURE_UNITS.get(feat, '')
    per_feat_errors = st.session_state.get('per_feature_errors', {})
    
    st.subheader(label)
    
    # ---- Top metrics row ----
    current_err = per_feat_errors.get(feat, 0.0)
    err_pct = current_err / (threshold + 1e-10) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    latest_val = df[feat].iloc[-1] if feat in df.columns and len(df) > 0 else 0
    c1.metric("Current Reading", f"{latest_val:.2f} {unit}")
    c2.metric("Reconstruction Error", f"{current_err:.5f}",
              delta=f"{err_pct:.0f}% of threshold",
              delta_color="inverse")
    c3.metric("Historical Min", f"{df[feat].min():.2f} {unit}")
    c4.metric("Historical Max", f"{df[feat].max():.2f} {unit}")
    
    st.markdown("---")
    
    # ---- Sensor error status ----
    if threshold > 0:
        if current_err > threshold:
            st.error(f"🔴 **This sensor is OVER the anomaly threshold.** "
                     f"Error {current_err:.5f} > threshold {threshold:.5f}. "
                     f"It is a primary contributor to the current alert.")
        elif current_err > threshold * 0.7:
            st.warning(f"🟡 **Approaching threshold.** "
                       f"Error is at {err_pct:.0f}% of the threshold — monitor closely.")
        else:
            st.success(f"🟢 **Sensor healthy.** "
                       f"Error is at {err_pct:.0f}% of threshold — normal operation.")
    
    # ---- Full time-series with anomaly overlay ----
    fig_ts = create_sensor_sparkline_with_anomaly(
        df, feat, threshold, per_feat_errors, max_points=500
    )
    fig_ts.update_layout(height=300)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # ---- Stats row ----
    st.markdown("**Rolling Statistics (last 50 readings)**")
    if len(df) >= 10 and feat in df.columns:
        recent = df[feat].tail(50)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Mean", f"{recent.mean():.2f}")
        s2.metric("Std Dev", f"{recent.std():.3f}")
        s3.metric("Skewness", f"{recent.skew():.2f}")
        
        # Drift indicator
        if len(recent) >= 20:
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent.values, 1)[0]
            s4.metric("Drift Rate", f"{slope:+.4f}/sample",
                      delta="Drifting" if abs(slope) > 0.01 else "Stable",
                      delta_color="inverse" if abs(slope) > 0.01 else "normal")
    
    # ---- Cross-validation panel (temperature sensors only) ----
    if feat in ('temp_dht', 'temp_therm') and 'temp_dht' in df.columns and 'temp_therm' in df.columns:
        st.markdown("---")
        st.markdown("**Cross-Validation: DHT11 vs Thermistor**")
        diff = (df['temp_dht'] - df['temp_therm']).tail(50)
        max_diff = diff.abs().max()
        mean_diff = diff.abs().mean()
        cv_c1, cv_c2 = st.columns(2)
        cv_c1.metric("Mean |Δ temp|", f"{mean_diff:.2f} °C",
                     delta="OK" if mean_diff < config.TEMP_CROSS_VALIDATION_MAX_DIFF else "DIVERGING",
                     delta_color="normal" if mean_diff < config.TEMP_CROSS_VALIDATION_MAX_DIFF else "inverse")
        cv_c2.metric("Max |Δ temp|", f"{max_diff:.2f} °C",
                     delta=f"Limit: {config.TEMP_CROSS_VALIDATION_MAX_DIFF} °C")

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
    if 'per_feature_errors' not in st.session_state:
        # Latest per-sensor reconstruction errors: {feat: float}
        st.session_state.per_feature_errors = {f: 0.0 for f in config.FEATURE_NAMES}
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
    # CLOUD AUTO-SIMULATION (Render / Headless deployment)
    # ========================================================================
    # If no serial ports are detected and we're not already streaming,
    # auto-start simulation so the deployed site shows a live demo.
    if 'cloud_auto_started' not in st.session_state:
        st.session_state.cloud_auto_started = False
    
    available_ports = get_available_ports()
    is_cloud = len(available_ports) == 0 and not SERIAL_AVAILABLE
    
    if (is_cloud
        and not st.session_state.streaming
        and not st.session_state.cloud_auto_started):
        st.session_state.sim_mode = True
        st.session_state.streaming = True
        st.session_state.sim_index = 0
        st.session_state.cloud_auto_started = True

    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.markdown('<p class="main-header" style="font-size:1.5rem">🔧 IoT Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="font-size:0.8rem">Multi-Sensor Predictive Maintenance</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.header("🔌 Connection")
        
        # Reuse ports already fetched above
        # available_ports is already set before sidebar
        
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
        
        if connect_btn and selected_port and SERIAL_AVAILABLE:
            try:
                ser = serial.Serial(selected_port, baud_rate, timeout=config.SERIAL_TIMEOUT)
                st.session_state.serial_port = ser
                st.session_state.serial_connected = True
                st.session_state.streaming = True
                st.session_state.sim_mode = False
                st.session_state.cloud_auto_started = False
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
        # Default: all sensors selected. Deselect any sensor not physically connected.
        if 'selected_sensors' not in st.session_state:
            st.session_state.selected_sensors = list(config.FEATURE_NAMES)
        
        current_default = [config.FEATURE_LABELS[s] for s in st.session_state.selected_sensors
                           if s in config.FEATURE_LABELS]
        selected_labels = st.multiselect(
            "Select Connected Hardware:",
            options=list(config.FEATURE_LABELS.values()),
            default=current_default,
            help="Deselect sensors that are not physically wired — their channels will be spoofed so the LSTM doesn't misfire"
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
                except Exception:
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
            return  # Let fragment's run_every handle the next tick
        
        # --- Current Status Metrics ---
        latest = df.iloc[-1] if len(df) > 0 else pd.Series()
        active_threshold = (st.session_state.lstm_threshold
                            if st.session_state.lstm_model
                            else config.STAT_ZSCORE_THRESHOLD)
        per_feat_errors = st.session_state.get('per_feature_errors', {})

        # Sensors to display = intersection of user selection and sensors that
        # have reported at least one valid reading in the last 5 samples
        user_selection = st.session_state.get('selected_sensors', config.FEATURE_NAMES)
        recent_5 = df.tail(5)
        active_sensors = [
            f for f in config.FEATURE_NAMES
            if f in user_selection
            and f in recent_5.columns
            and recent_5[f].notna().any()
        ]

        def sensor_alert_info(feat: str):
            """
            Returns (severity, message) for a sensor using per-feature
            reconstruction error when the LSTM is active, falling back to
            z-score otherwise.  Much more accurate than pure z-score alone.
            """
            err = per_feat_errors.get(feat, 0.0)
            if active_threshold and active_threshold > 0 and err > 0:
                pct = err / active_threshold * 100
                if err > active_threshold:
                    return 'critical', f'⚠️ Error {pct:.0f}% of threshold — anomalous'
                elif err > active_threshold * 0.7:
                    return 'warning', f'⚠️ Error {pct:.0f}% of threshold — watch closely'
            # z-score fallback for statistical mode
            if len(df) > 10 and feat in latest and not pd.isna(latest.get(feat)):
                mean = df[feat].mean()
                std = df[feat].std() + 0.01
                z = abs(latest[feat] - mean) / std
                if z > 3.5:
                    return 'critical', f'⚠️ Z-score {z:.1f} — critical deviation'
                elif z > 2.5:
                    return 'warning', f'⚠️ Z-score {z:.1f} — elevated'
            return None, None

        # ---------------------------------------------------------------
        # STATUS BANNER (full width)
        # ---------------------------------------------------------------
        health = latest.get('health_score', 100) if not latest.empty else 100
        fault  = latest.get('fault_type',   'Healthy') if not latest.empty else 'Healthy'
        alert  = latest.get('alert_level',  'Normal')  if not latest.empty else 'Normal'

        if alert == 'Critical':
            st.markdown(f'<div class="alert-critical">🚨 ALERT — {fault}</div>', unsafe_allow_html=True)
        elif alert == 'Warning':
            st.markdown(f'<div class="alert-warning">⚠️ WARNING — {fault}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-normal">✅ All Systems Nominal</div>', unsafe_allow_html=True)

        st.markdown("")

        # ---------------------------------------------------------------
        # ROW 1: Health Gauge + Key Metrics (2 columns, breathing room)
        # ---------------------------------------------------------------
        col_health, col_stats = st.columns([1, 1])

        with col_health:
            fig_health = create_health_gauge(health)
            fig_health.update_layout(height=280)
            st.plotly_chart(fig_health, use_container_width=True, key="sys_health_gauge")

            healthy_pct = ((df['alert_level'] == 'Normal').sum() / len(df) * 100
                           if len(df) > 0 else 100)
            anomaly_count = (int((df['anomaly_score'] > active_threshold).sum())
                             if 'anomaly_score' in df.columns else 0)
            m1, m2 = st.columns(2)
            m1.metric("Uptime", f"{healthy_pct:.1f}%")
            m2.metric("Anomalies", f"{anomaly_count}")

        with col_stats:
            fig_anomaly = create_anomaly_chart(df, active_threshold, max_points=300)
            fig_anomaly.update_layout(height=320)
            st.plotly_chart(fig_anomaly, use_container_width=True, key="anomaly_main")

        st.markdown("")

        # ---------------------------------------------------------------
        # ROW 2: Fault Localization (full width — needs horizontal space)
        # ---------------------------------------------------------------
        if st.session_state.lstm_model and any(v > 0 for v in per_feat_errors.values()):
            fig_loc = create_fault_localization_chart(
                per_feat_errors, active_threshold, active_sensors
            )
            fig_loc.update_layout(height=280)
            st.plotly_chart(fig_loc, use_container_width=True, key="fault_loc")

            # Pinpoint the worst offender
            offenders = {f: e for f, e in per_feat_errors.items()
                         if f in active_sensors and e > active_threshold}
            if offenders:
                worst = max(offenders, key=offenders.get)
                worst_label = config.FEATURE_LABELS.get(worst, worst)
                st.error(f"**Primary fault source:** {worst_label}  \n"
                         f"Error {offenders[worst]:.5f} ({offenders[worst]/active_threshold*100:.0f}% of threshold)")
            else:
                st.success("All sensor reconstruction errors within normal bounds.")
        else:
            # Statistical mode: show correlation heatmap
            fig_corr = create_sensor_correlation_heatmap(df, max_points=200)
            fig_corr.update_layout(height=300)
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_main")

        st.markdown("---")

        # ---------------------------------------------------------------
        # ROW 3+: Per-Sensor Gauges (3 per row, click for details)
        # ---------------------------------------------------------------
        st.subheader("📊 Sensor Network")

        if not active_sensors:
            st.warning("No sensors selected or reporting data.  "
                       "Use **Active Modules** in the sidebar to choose sensors.")
        else:
            for i in range(0, len(active_sensors), 3):
                row_feats = active_sensors[i:i+3]
                cols_r = st.columns(3)

                for j, feat in enumerate(row_feats):
                    with cols_r[j]:
                        severity, alert_msg = sensor_alert_info(feat)
                        if severity == 'critical':
                            st.markdown(
                                f'<div class="local-alert local-alert-critical">{alert_msg}</div>',
                                unsafe_allow_html=True)
                        elif severity == 'warning':
                            st.markdown(
                                f'<div class="local-alert local-alert-warning">{alert_msg}</div>',
                                unsafe_allow_html=True)

                        val = latest.get(feat, 0) if not latest.empty else 0
                        if pd.isna(val):
                            val = 0
                        min_val, max_val = config.SENSOR_RANGES.get(feat, (0, 100))

                        fig_g = create_single_sensor_gauge(val, feat, min_val, max_val)
                        fig_g.update_layout(height=170, margin=dict(l=15, r=15, t=35, b=10))
                        st.plotly_chart(fig_g, use_container_width=True,
                                        key=f"gauge_{feat}_{i}_{j}")

                        if st.button("🔬 Diagnose", key=f"btn_{feat}_{i}_{j}",
                                     use_container_width=True):
                            sensor_lightbox(df, feat, threshold=active_threshold)


    _live_dashboard_fragment()

if __name__ == "__main__":
    main()