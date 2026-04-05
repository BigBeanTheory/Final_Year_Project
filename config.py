"""
Configuration file for IoT Sensor Health Monitoring System
Multi-Sensor Predictive Maintenance with Sensor Fusion

Supports 6-feature multi-sensor array:
  - DHT11 (temperature + humidity)
  - Thermistor (analog temperature for cross-validation)
  - Sound sensor (acoustic monitoring)
  - LDR / Photoresistor (light level)
  - Flame sensor (fire/overheating detection)
"""

import os

# Base directory (Final Year Project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# MULTI-SENSOR FEATURE CONFIGURATION
# ============================================================================

# Feature names — order must match Arduino serial CSV output
FEATURE_NAMES = [
    'temp_dht',          # DHT11 temperature (°C)
    'humidity',          # DHT11 humidity (%)
    'temp_therm',        # Thermistor temperature (°C)
    'sound_level',       # Sound sensor RMS level (0-512)
    'light_level',       # LDR reading (0-1023)
    'flame_intensity',   # Flame sensor (0-1023, higher = more flame)
]

N_FEATURES = len(FEATURE_NAMES)  # 6

# Human-readable labels for dashboard display
FEATURE_LABELS = {
    'temp_dht':        '🌡️ Temperature (DHT11)',
    'humidity':        '💧 Humidity',
    'temp_therm':      '🔥 Temperature (Thermistor)',
    'sound_level':     '🔊 Sound Level',
    'light_level':     '💡 Light Level',
    'flame_intensity': '🔥 Flame Intensity',
}

# Units for each feature
FEATURE_UNITS = {
    'temp_dht':        '°C',
    'humidity':        '%',
    'temp_therm':      '°C',
    'sound_level':     'RMS',
    'light_level':     'lux (raw)',
    'flame_intensity': 'intensity',
}

# ============================================================================
# SENSOR VALIDATION RANGES
# ============================================================================

# Per-sensor valid ranges for data validation
SENSOR_RANGES = {
    'temp_dht':        (-10, 60),     # DHT11 spec: 0-50, with margin
    'humidity':        (0, 100),      # DHT11 spec: 20-90, with margin
    'temp_therm':      (-20, 80),     # Thermistor range
    'sound_level':     (0, 600),      # RMS of analog (0-512 theoretical max)
    'light_level':     (0, 1023),     # 10-bit ADC range
    'flame_intensity': (0, 1023),     # 10-bit ADC range (inverted)
}

# Cross-validation: max acceptable divergence between DHT11 and thermistor temp
TEMP_CROSS_VALIDATION_MAX_DIFF = 5.0  # °C — beyond this = sensor fault

# ============================================================================
# INTEL BERKELEY DATASET CONFIGURATION
# ============================================================================

# Intel Berkeley model and data paths
INTEL_MODEL_DIR = os.path.join(BASE_DIR, "intel_berkeley_results")
INTEL_DATA_DIR = os.path.join(BASE_DIR, "Intel Berkeley Research Lab Sensor Data")

INTEL_MODEL_PATH = os.path.join(INTEL_MODEL_DIR, "lstm_autoencoder_intel.h5")
INTEL_PREPROCESSOR_PATH = os.path.join(INTEL_MODEL_DIR, "preprocessor_intel.pkl")
INTEL_THRESHOLD_PATH = os.path.join(INTEL_MODEL_DIR, "threshold_intel.npy")
INTEL_RESULTS_PATH = os.path.join(INTEL_MODEL_DIR, "detection_results_intel.csv")

INTEL_RAW_DATA_PATH = os.path.join(INTEL_DATA_DIR, "data.txt")
INTEL_NODE_ID = 7.0  # Selected node with best data quality

# ============================================================================
# MULTI-SENSOR MODEL PATHS
# ============================================================================

MULTI_SENSOR_MODEL_DIR = os.path.join(BASE_DIR, "multi_sensor_results")
MULTI_SENSOR_MODEL_PATH = os.path.join(MULTI_SENSOR_MODEL_DIR, "lstm_autoencoder_multi.h5")
MULTI_SENSOR_PREPROCESSOR_PATH = os.path.join(MULTI_SENSOR_MODEL_DIR, "preprocessor_multi.pkl")
MULTI_SENSOR_THRESHOLD_PATH = os.path.join(MULTI_SENSOR_MODEL_DIR, "threshold_multi.npy")

# ============================================================================
# SYNTHETIC DATA CONFIGURATION (for testing/demo)
# ============================================================================

SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "data")
SYNTHETIC_DATA_PATH = os.path.join(SYNTHETIC_DATA_DIR, "sensor_test.csv")
MULTI_SENSOR_SYNTHETIC_PATH = os.path.join(SYNTHETIC_DATA_DIR, "multi_sensor_test.csv")

# ============================================================================
# DEFAULT CONFIGURATION (CHOOSE WHICH DATASET TO USE)
# ============================================================================

# Set this to switch between multi-sensor and Intel Berkeley
USE_MULTI_SENSOR = True
USE_INTEL_BERKELEY = False  # Legacy 2-feature mode

if USE_MULTI_SENSOR:
    DEFAULT_MODEL_PATH = MULTI_SENSOR_MODEL_PATH
    DEFAULT_PREPROCESSOR_PATH = MULTI_SENSOR_PREPROCESSOR_PATH
    DEFAULT_THRESHOLD_PATH = MULTI_SENSOR_THRESHOLD_PATH
    DEFAULT_DATA_PATH = MULTI_SENSOR_SYNTHETIC_PATH
    DEFAULT_RAW_DATA_PATH = MULTI_SENSOR_SYNTHETIC_PATH
elif USE_INTEL_BERKELEY:
    DEFAULT_MODEL_PATH = INTEL_MODEL_PATH
    DEFAULT_PREPROCESSOR_PATH = INTEL_PREPROCESSOR_PATH
    DEFAULT_THRESHOLD_PATH = INTEL_THRESHOLD_PATH
    DEFAULT_DATA_PATH = INTEL_RESULTS_PATH
    DEFAULT_RAW_DATA_PATH = INTEL_RAW_DATA_PATH
else:
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_autoencoder.h5")
    DEFAULT_PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
    DEFAULT_THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.npy")
    DEFAULT_DATA_PATH = SYNTHETIC_DATA_PATH
    DEFAULT_RAW_DATA_PATH = SYNTHETIC_DATA_PATH

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# LSTM Model architecture
WINDOW_SIZE = 20
ENCODING_DIM = 32       # Increased from 16 for 6-feature input
LSTM_UNITS = (128, 64)  # Increased from (64, 32) for more complex patterns
DROPOUT_RATE = 0.2

# Training parameters
TRAIN_RATIO = 0.7
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# Anomaly detection
THRESHOLD_METHOD = 'percentile'
THRESHOLD_PERCENTILE = 95

# ============================================================================
# HEALTH MONITORING PARAMETERS
# ============================================================================

# Health score calculation
EMA_ALPHA = 0.05  # Exponential moving average smoothing
DRIFT_WINDOW = 50  # Window for drift detection
NOISE_WINDOW = 20  # Window for noise detection
FREEZE_THRESHOLD = 30  # Consecutive identical readings = freeze

# Alert thresholds
HEALTH_WARNING_THRESHOLD = 80
HEALTH_CRITICAL_THRESHOLD = 50

# Fault confirmation — require sustained anomalies before declaring a fault
FAULT_CONFIRM_WINDOW = 15     # Look at last N readings for confirmation
FAULT_CONFIRM_RATIO = 0.60    # 60% of window must be anomalous to confirm fault
FAULT_MIN_SAMPLES = 20        # Minimum samples before any fault classification

# Sensor fusion weights for composite health score
# Higher weight = more influence on overall health
SENSOR_WEIGHTS = {
    'temp_dht':        0.20,
    'humidity':        0.15,
    'temp_therm':      0.15,
    'sound_level':     0.20,
    'light_level':     0.10,
    'flame_intensity': 0.20,  # High weight for safety-critical sensor
}

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================

DASHBOARD_TITLE = "IoT Multi-Sensor Predictive Maintenance"
DASHBOARD_PAGE_ICON = "🔧"
DEFAULT_STREAM_SPEED = 10  # Samples per second
MAX_DISPLAY_SAMPLES = 500  # Maximum samples to show on charts

# Chart colors for each sensor
SENSOR_COLORS = {
    'temp_dht':        '#ff6b6b',   # Red
    'humidity':        '#4ecdc4',   # Teal
    'temp_therm':      '#ff9f43',   # Orange
    'sound_level':     '#a55eea',   # Purple
    'light_level':     '#feca57',   # Yellow
    'flame_intensity': '#ff4757',   # Crimson
}

# ============================================================================
# SERIAL COMMUNICATION (Arduino)
# ============================================================================

SERIAL_BAUD_RATE = 9600
SERIAL_TIMEOUT = 1  # seconds
SERIAL_READ_INTERVAL = 1.0  # seconds between reads

# ============================================================================
# STATISTICAL ANOMALY DETECTION (live mode, no TensorFlow)
# ============================================================================

STAT_ROLLING_WINDOW = 30  # Rolling window for mean/std calculation
STAT_ZSCORE_THRESHOLD = 2.5  # Z-score threshold for anomaly flagging
STAT_MIN_SAMPLES = 10  # Minimum samples before anomaly detection starts

# ============================================================================
# DATA VALIDATION PARAMETERS (Legacy — use SENSOR_RANGES instead)
# ============================================================================

TEMP_MIN = -10  # Celsius
TEMP_MAX = 60   # Celsius
HUMIDITY_MIN = 0  # Percentage
HUMIDITY_MAX = 100  # Percentage

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_files_exist():
    """
    Verify that all required files exist.
    Returns a dictionary with file status.
    """
    status = {
        'model': os.path.exists(DEFAULT_MODEL_PATH),
        'preprocessor': os.path.exists(DEFAULT_PREPROCESSOR_PATH),
        'threshold': os.path.exists(DEFAULT_THRESHOLD_PATH),
        'data': os.path.exists(DEFAULT_DATA_PATH)
    }
    return status

def get_config_summary():
    """
    Print current configuration summary.
    """
    print("="*70)
    print("IoT MULTI-SENSOR PREDICTIVE MAINTENANCE - CONFIGURATION")
    print("="*70)
    
    mode = "Multi-Sensor (6 features)" if USE_MULTI_SENSOR else (
        "Intel Berkeley (2 features)" if USE_INTEL_BERKELEY else "Synthetic"
    )
    print(f"Mode: {mode}")
    print(f"Features ({N_FEATURES}): {', '.join(FEATURE_NAMES)}")
    
    print(f"\nModel paths:")
    print(f"  Model: {DEFAULT_MODEL_PATH}")
    print(f"  Preprocessor: {DEFAULT_PREPROCESSOR_PATH}")
    print(f"  Threshold: {DEFAULT_THRESHOLD_PATH}")
    print(f"  Data: {DEFAULT_DATA_PATH}")
    
    status = check_files_exist()
    print(f"\nFile status:")
    for name, exists in status.items():
        symbol = "[OK]" if exists else "[X]"
        print(f"  {symbol} {name}: {'Found' if exists else 'NOT FOUND'}")
    
    print(f"\nModel hyperparameters:")
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  Features: {N_FEATURES}")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  Encoding dim: {ENCODING_DIM}")
    print(f"\nSensor weights:")
    for name, weight in SENSOR_WEIGHTS.items():
        print(f"  {name}: {weight}")
    print("="*70)

if __name__ == "__main__":
    get_config_summary()
