"""
FastAPI Backend — IoT Predictive Maintenance Dashboard
Real-time WebSocket server for sensor data, anomaly detection,
health scoring, and actuator control.

Run with:  uvicorn app:app --reload --port 8000
"""

import asyncio
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import config
from iot_health_score import SensorHealthMonitor, FaultType, AlertLevel


# ── Custom JSON encoder for datetime / numpy types ──────────────────────────
class _SafeEncoder(json.JSONEncoder):
    """Handles datetime, date, numpy scalars, and numpy arrays."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


async def safe_send_json(ws: WebSocket, data: dict):
    """Send *data* as JSON, using _SafeEncoder to avoid serialisation crashes."""
    await ws.send_text(json.dumps(data, cls=_SafeEncoder))

# Optional MQTT
try:
    from iot_mqtt_subscriber import MQTTSensorBridge
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

# Optional LSTM
try:
    from iot_lstm_model import LSTMAutoencoder
    from iot_preprocessing import SensorPreprocessor
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(title="IoT Predictive Maintenance")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================================
# GLOBAL STATE
# ============================================================================

class DashboardState:
    """Server-side state shared across WebSocket connections."""

    def __init__(self):
        self.data_buffer: list[dict] = []
        self.monitor = SensorHealthMonitor(
            ema_alpha=config.EMA_ALPHA,
            drift_window=config.DRIFT_WINDOW,
            noise_window=config.NOISE_WINDOW,
            freeze_threshold=config.FREEZE_THRESHOLD,
        )
        self.per_feature_errors: dict[str, float] = {f: 0.0 for f in config.FEATURE_NAMES}
        self.mqtt_bridge: Optional[MQTTSensorBridge] = None
        self.mqtt_mode = False
        self.sim_mode = False
        self.sim_index = 0
        self.streaming = False
        self.active_ws: list[WebSocket] = []
        self.last_actuator_state = None

        # LSTM model (loaded once)
        self.lstm_model = None
        self.lstm_preprocessor = None
        self.lstm_threshold = None
        self._load_lstm()

    def _load_lstm(self):
        if not LSTM_AVAILABLE:
            return
        try:
            preprocessor = SensorPreprocessor()
            preprocessor.load(config.DEFAULT_PREPROCESSOR_PATH)
            model = LSTMAutoencoder(n_features=len(preprocessor.feature_columns))
            model.load(config.DEFAULT_MODEL_PATH)
            threshold = float(np.load(config.DEFAULT_THRESHOLD_PATH))
            model.threshold = threshold
            self.lstm_model = model
            self.lstm_preprocessor = preprocessor
            self.lstm_threshold = threshold
            print(f"[LSTM] Model loaded — threshold={threshold:.5f}")
        except Exception as e:
            print(f"[LSTM] Could not load model: {e}")


state = DashboardState()

# ============================================================================
# ANOMALY / HEALTH COMPUTATION
# ============================================================================

def compute_anomaly_scores_statistical(df: pd.DataFrame) -> np.ndarray:
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
        rolling_std = df[col].rolling(window=window, min_periods=config.STAT_MIN_SAMPLES).std().replace(0, 0.01)
        z = np.abs((df[col] - rolling_mean) / rolling_std).fillna(0).values
        scores += z
    if n_features > 0:
        scores /= n_features
    return scores


def compute_anomaly_scores_lstm(df: pd.DataFrame) -> np.ndarray:
    model = state.lstm_model
    preprocessor = state.lstm_preprocessor
    scores = np.zeros(len(df))
    if model is None or preprocessor is None or len(df) < config.WINDOW_SIZE:
        return scores
    try:
        feature_cols = [c for c in preprocessor.feature_columns if c in df.columns]
        feature_df = df[feature_cols].copy()
        for i, col in enumerate(preprocessor.feature_columns):
            if col in feature_df.columns:
                if feature_df[col].isna().all():
                    mid = (preprocessor.scaler.data_min_[i] + preprocessor.scaler.data_max_[i]) / 2.0
                    feature_df[col] = mid
                else:
                    feature_df[col] = feature_df[col].ffill().bfill().fillna(0)
        scaled = np.clip(preprocessor.scaler.transform(feature_df), 0.0, 1.0)
        windows = [scaled[i:i + config.WINDOW_SIZE] for i in range(len(scaled) - config.WINDOW_SIZE + 1)]
        if not windows:
            return scores
        X = np.array(windows)
        errors = model.compute_reconstruction_error(X, per_sample=True)
        offset = config.WINDOW_SIZE - 1
        scores[offset:offset + len(errors)] = errors
        if len(errors):
            scores[:offset] = errors[0]

        # Per-feature error for latest window
        if len(X):
            per_feat = model.compute_per_feature_error(X[-1:])
            state.per_feature_errors = {k: float(v[0]) for k, v in per_feat.items()}
    except Exception:
        scores = compute_anomaly_scores_statistical(df)
    return scores


def process_reading(reading: dict) -> dict:
    """Process a single sensor reading through the anomaly/health pipeline.
    Returns a JSON-serializable dict to send to the frontend."""

    state.data_buffer.append(reading)
    if len(state.data_buffer) > config.MAX_DISPLAY_SAMPLES * 2:
        state.data_buffer = state.data_buffer[-config.MAX_DISPLAY_SAMPLES:]

    df = pd.DataFrame(state.data_buffer)

    # Anomaly scores
    if state.lstm_model is not None:
        anomaly_scores = compute_anomaly_scores_lstm(df)
        threshold = state.lstm_threshold
    else:
        anomaly_scores = compute_anomaly_scores_statistical(df)
        threshold = config.STAT_ZSCORE_THRESHOLD

    # Health scores
    health_scores = state.monitor.anomaly_score_to_health(anomaly_scores, threshold)

    anomaly_score = float(anomaly_scores[-1]) if len(anomaly_scores) else 0.0
    health_score = float(health_scores[-1]) if len(health_scores) else 100.0

    # Fault classification
    fault_type = FaultType.HEALTHY
    alert_level = AlertLevel.NORMAL
    if len(df) >= config.FAULT_MIN_SAMPLES:
        recent = anomaly_scores[-config.FAULT_CONFIRM_WINDOW:]
        ratio = sum(1 for s in recent if s > threshold) / len(recent)
        if ratio >= config.FAULT_CONFIRM_RATIO:
            fault_type = state.monitor.classify_fault(df, anomaly_score, threshold)
            alert_level = state.monitor.determine_alert_level(health_score, fault_type)

            # Auto-actuator via MQTT (only publish if state changed)
            if state.mqtt_mode and state.mqtt_bridge and state.mqtt_bridge.is_connected():
                if alert_level != state.last_actuator_state:
                    if alert_level == AlertLevel.CRITICAL:
                        state.mqtt_bridge.send_command("BUZZ:ON")
                        state.mqtt_bridge.send_command("LED:RED")
                    elif alert_level == AlertLevel.WARNING:
                        state.mqtt_bridge.send_command("LED:YELLOW")
                    else:
                        state.mqtt_bridge.send_command("BUZZ:OFF")
                        state.mqtt_bridge.send_command("LED:GREEN")
                    state.last_actuator_state = alert_level

    # Build payload for frontend
    payload = {
        "type": "sensor_data",
        "timestamp": reading["timestamp"].isoformat() if isinstance(reading["timestamp"], datetime) else reading["timestamp"],
        "sensors": {f: reading.get(f, 0) for f in config.FEATURE_NAMES},
        "anomaly_score": round(anomaly_score, 6),
        "health_score": round(health_score, 2),
        "threshold": round(threshold, 6),
        "fault_type": fault_type.value,
        "alert_level": alert_level.value,
        "per_feature_errors": {k: round(v, 6) for k, v in state.per_feature_errors.items()},
        "sample_count": len(state.data_buffer),
    }
    return payload


def generate_simulated_reading(index: int) -> dict:
    base = {
        'temp_dht': 22.0, 'humidity': 55.0, 'temp_therm': 22.0,
        'sound_level': 45.0, 'light_level': 500.0, 'flame_intensity': 8.0,
    }
    reading = {
        'timestamp': datetime.now(),
        'temp_dht': base['temp_dht'] + np.random.normal(0, 0.3),
        'humidity': base['humidity'] + np.random.normal(0, 1.5),
        'temp_therm': base['temp_dht'] + np.random.normal(0.1, 0.25),
        'sound_level': base['sound_level'] + np.random.normal(0, 4),
        'light_level': base['light_level'] + np.random.normal(0, 15),
        'flame_intensity': base['flame_intensity'] + abs(np.random.normal(0, 2)),
    }
    # Fault injection
    if index > 60 and index % 150 < 15:
        reading['temp_dht'] += (index % 150) * 0.3
    if index > 100 and index % 200 < 10:
        reading['temp_therm'] += 7.0
    if index > 140 and index % 180 < 12:
        reading['sound_level'] += np.random.uniform(80, 150)
    if index > 200 and index % 250 < 8:
        reading['flame_intensity'] = 600 + np.random.normal(0, 30)
        reading['temp_dht'] += 5
        reading['temp_therm'] += 4.5
    if index > 250 and index % 300 < 6:
        reading['light_level'] += 200 * np.sin(index * 0.5)
    for feat in config.FEATURE_NAMES:
        lo, hi = config.SENSOR_RANGES[feat]
        reading[feat] = max(lo, min(hi, round(reading[feat], 2)))
    return reading


# ============================================================================
# BACKGROUND DATA LOOP
# ============================================================================

async def data_loop():
    """Continuously read data and push to all connected WebSocket clients."""
    while True:
        reading = None

        if state.streaming:
            if state.mqtt_mode and state.mqtt_bridge:
                mqtt_reading = state.mqtt_bridge.get_latest_reading()
                if mqtt_reading:
                    reading = mqtt_reading
            elif state.sim_mode:
                reading = generate_simulated_reading(state.sim_index)
                state.sim_index += 1

        if reading:
            payload = process_reading(reading)
            dead = []
            for ws in state.active_ws:
                try:
                    await safe_send_json(ws, payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                state.active_ws.remove(ws)

        await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup():
    asyncio.create_task(data_loop())


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def get_config():
    return {
        "feature_names": config.FEATURE_NAMES,
        "feature_labels": config.FEATURE_LABELS,
        "feature_units": config.FEATURE_UNITS,
        "sensor_ranges": config.SENSOR_RANGES,
        "sensor_colors": config.SENSOR_COLORS,
        "sensor_weights": config.SENSOR_WEIGHTS,
        "lstm_available": state.lstm_model is not None,
        "mqtt_available": MQTT_AVAILABLE,
        "threshold": state.lstm_threshold or config.STAT_ZSCORE_THRESHOLD,
    }


@app.get("/api/status")
async def get_status():
    mqtt_connected = (state.mqtt_bridge is not None and state.mqtt_bridge.is_connected()) if state.mqtt_bridge else False
    device_status = state.mqtt_bridge.get_device_status() if state.mqtt_bridge else {}
    return {
        "streaming": state.streaming,
        "mqtt_mode": state.mqtt_mode,
        "mqtt_connected": mqtt_connected,
        "sim_mode": state.sim_mode,
        "sample_count": len(state.data_buffer),
        "device_status": device_status,
        "lstm_active": state.lstm_model is not None,
    }


@app.post("/api/mqtt/connect")
async def mqtt_connect():
    if not MQTT_AVAILABLE:
        return JSONResponse({"ok": False, "error": "paho-mqtt not installed"}, 400)
    bridge = MQTTSensorBridge()
    connected = bridge.start()
    state.mqtt_bridge = bridge
    state.mqtt_mode = True
    state.streaming = True
    state.sim_mode = False
    return {"ok": True, "connected": connected}


@app.post("/api/mqtt/disconnect")
async def mqtt_disconnect():
    if state.mqtt_bridge:
        state.mqtt_bridge.stop()
    state.mqtt_bridge = None
    state.mqtt_mode = False
    state.streaming = False
    return {"ok": True}


@app.post("/api/sim/start")
async def sim_start():
    state.sim_mode = True
    state.streaming = True
    state.mqtt_mode = False
    state.sim_index = 0
    return {"ok": True}


@app.post("/api/sim/stop")
async def sim_stop():
    state.sim_mode = False
    state.streaming = False
    return {"ok": True}


@app.post("/api/command/{cmd}")
async def send_command(cmd: str):
    """Send actuator command. cmd can be BUZZ:ON, LED:RED, etc."""
    if state.mqtt_bridge and state.mqtt_bridge.is_connected():
        state.mqtt_bridge.send_command(cmd)
        return {"ok": True, "sent": cmd}
    return JSONResponse({"ok": False, "error": "MQTT not connected"}, 400)


@app.post("/api/clear")
async def clear_data():
    state.data_buffer.clear()
    state.sim_index = 0
    state.per_feature_errors = {f: 0.0 for f in config.FEATURE_NAMES}
    # Reset health monitor accumulated state
    state.monitor.health_scores.clear()
    state.monitor.fault_history.clear()
    state.monitor.alert_history.clear()
    return {"ok": True}


# ============================================================================
# WEBSOCKET
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.active_ws.append(ws)
    print(f"[WS] Client connected ({len(state.active_ws)} active)")

    # Send initial config
    await safe_send_json(ws, {
        "type": "init",
        "config": {
            "feature_names": config.FEATURE_NAMES,
            "feature_labels": config.FEATURE_LABELS,
            "feature_units": config.FEATURE_UNITS,
            "sensor_ranges": config.SENSOR_RANGES,
            "sensor_colors": config.SENSOR_COLORS,
            "threshold": state.lstm_threshold or config.STAT_ZSCORE_THRESHOLD,
            "lstm_active": state.lstm_model is not None,
        }
    })

    # Send any existing buffered data (last 50 readings)
    if state.data_buffer:
        for reading in state.data_buffer[-50:]:
            payload = {
                "type": "sensor_data",
                "timestamp": reading["timestamp"].isoformat() if isinstance(reading["timestamp"], datetime) else reading["timestamp"],
                "sensors": {f: reading.get(f, 0) for f in config.FEATURE_NAMES},
                "anomaly_score": 0,
                "health_score": 100,
                "threshold": state.lstm_threshold or config.STAT_ZSCORE_THRESHOLD,
                "fault_type": "Healthy",
                "alert_level": "Normal",
                "per_feature_errors": state.per_feature_errors,
                "sample_count": len(state.data_buffer),
            }
            try:
                await safe_send_json(ws, payload)
            except Exception:
                break

    try:
        while True:
            # Listen for commands from the client
            data = await ws.receive_text()
            msg = json.loads(data)
            action = msg.get("action")

            if action == "mqtt_connect":
                result = await mqtt_connect()
                await safe_send_json(ws, {"type": "status", "action": "mqtt_connect", **result})
            elif action == "mqtt_disconnect":
                result = await mqtt_disconnect()
                await safe_send_json(ws, {"type": "status", "action": "mqtt_disconnect", **result})
            elif action == "sim_start":
                result = await sim_start()
                await safe_send_json(ws, {"type": "status", "action": "sim_start", **result})
            elif action == "sim_stop":
                result = await sim_stop()
                await safe_send_json(ws, {"type": "status", "action": "sim_stop", **result})
            elif action == "command":
                cmd = msg.get("cmd", "")
                result = await send_command(cmd)
                if isinstance(result, JSONResponse):
                    await safe_send_json(ws, {"type": "status", "action": "command", "ok": False})
                else:
                    await safe_send_json(ws, {"type": "status", "action": "command", **result})
            elif action == "clear":
                await clear_data()
                await safe_send_json(ws, {"type": "status", "action": "clear", "ok": True})
            elif action == "get_status":
                result = await get_status()
                await safe_send_json(ws, {"type": "server_status", **result})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        if ws in state.active_ws:
            state.active_ws.remove(ws)
        print(f"[WS] Client disconnected ({len(state.active_ws)} active)")
