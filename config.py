"""
Configuration file for IoT Sensor Health Monitoring System
Contains paths to trained models, data, and system settings.
"""

import os

# Base directory (Final Year Project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# SYNTHETIC DATA CONFIGURATION (for testing/demo)
# ============================================================================

SYNTHETIC_DATA_DIR = os.path.join(BASE_DIR, "data")
SYNTHETIC_DATA_PATH = os.path.join(SYNTHETIC_DATA_DIR, "sensor_test.csv")

# ============================================================================
# DEFAULT CONFIGURATION (CHOOSE WHICH DATASET TO USE)
# ============================================================================

# Set this to switch between Intel Berkeley and synthetic data
USE_INTEL_BERKELEY = True  # Set to False to use synthetic data

if USE_INTEL_BERKELEY:
    DEFAULT_MODEL_PATH = INTEL_MODEL_PATH
    DEFAULT_PREPROCESSOR_PATH = INTEL_PREPROCESSOR_PATH
    DEFAULT_THRESHOLD_PATH = INTEL_THRESHOLD_PATH
    DEFAULT_DATA_PATH = INTEL_RESULTS_PATH  # Pre-processed results for dashboard
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
N_FEATURES = 2  # Temperature + Humidity
ENCODING_DIM = 16
LSTM_UNITS = (64, 32)
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
FREEZE_THRESHOLD = 30  # Consecutive identical readings = freeze (DHT11 has 1°C resolution, so repeats are normal)

# Alert thresholds
HEALTH_WARNING_THRESHOLD = 80
HEALTH_CRITICAL_THRESHOLD = 50

# Fault confirmation — require sustained anomalies before declaring a fault
FAULT_CONFIRM_WINDOW = 15     # Look at last N readings for confirmation
FAULT_CONFIRM_RATIO = 0.60    # 60% of window must be anomalous to confirm fault
FAULT_MIN_SAMPLES = 20        # Minimum samples before any fault classification

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================

DASHBOARD_TITLE = "IoT Sensor Health Monitor - Intel Berkeley Dataset"
DASHBOARD_PAGE_ICON = "🔧"
DEFAULT_STREAM_SPEED = 10  # Samples per second
MAX_DISPLAY_SAMPLES = 500  # Maximum samples to show on charts

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
# DATA VALIDATION PARAMETERS (DHT11 Sensor)
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
    print("IoT SENSOR HEALTH MONITORING - CONFIGURATION")
    print("="*70)
    print(f"Dataset: {'Intel Berkeley' if USE_INTEL_BERKELEY else 'Synthetic'}")
    print(f"\nModel paths:")
    print(f"  Model: {DEFAULT_MODEL_PATH}")
    print(f"  Preprocessor: {DEFAULT_PREPROCESSOR_PATH}")
    print(f"  Threshold: {DEFAULT_THRESHOLD_PATH}")
    print(f"  Data: {DEFAULT_DATA_PATH}")
    
    status = check_files_exist()
    print(f"\nFile status:")
    for name, exists in status.items():
        symbol = "[OK]" if exists else "[X]"  # Use ASCII instead of unicode
        print(f"  {symbol} {name}: {'Found' if exists else 'NOT FOUND'}")
    
    print(f"\nModel hyperparameters:")
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  Features: {N_FEATURES}")
    print(f"  LSTM units: {LSTM_UNITS}")
    print(f"  Encoding dim: {ENCODING_DIM}")
    print("="*70)

if __name__ == "__main__":
    get_config_summary()
