"""
Data Ingestion Module (Multi-Sensor)
Handles parsing of multi-sensor CSV and serial data, validation,
cross-validation, and unified time-series output.

Supports 6-channel input:
  temp_dht, humidity, temp_therm, sound_level, light_level, flame_intensity

Engineering Notes:
- Cross-validates DHT11 vs thermistor temperature
- Per-sensor validation with physical limits
- Handles error/NaN/invalid readings gracefully
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, Optional, Dict

import config


class SensorDataIngestion:
    """
    Unified parser for multi-sensor IoT data.
    """
    
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.cross_validation_warnings = []
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV format multi-sensor data.
        Expected columns: timestamp, temp_dht, humidity, temp_therm,
                         sound_level, light_level, flame_intensity
        """
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.lower().str.strip()
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
            else:
                df['timestamp'] = pd.date_range(start='2024-01-01',
                                                 periods=len(df), freq='1S')
            
            # Ensure numeric types for all sensor columns
            for col in config.FEATURE_NAMES:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle legacy 2-feature format (backwards compatibility)
            if 'temperature' in df.columns and 'temp_dht' not in df.columns:
                df['temp_dht'] = df['temperature']
                df['temp_therm'] = df['temperature'] + np.random.randn(len(df)) * 0.3
                df['sound_level'] = 45 + np.random.randn(len(df)) * 4
                df['light_level'] = 500 + np.random.randn(len(df)) * 15
                df['flame_intensity'] = 8 + np.abs(np.random.randn(len(df)) * 2)
            
            # Select available columns
            available = ['timestamp'] + [c for c in config.FEATURE_NAMES if c in df.columns]
            return df[available]
            
        except Exception as e:
            raise ValueError(f"CSV parsing failed: {e}")
    
    def load_txt(self, filepath: str, node_id: float = 7.0) -> pd.DataFrame:
        """
        Load Intel Berkeley Lab format (legacy 2-feature).
        Generates synthetic values for the additional 4 sensors.
        """
        columns = ['date', 'time', 'epoch', 'moteid',
                    'temperature', 'humidity', 'light', 'voltage']
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None,
                             names=columns, on_bad_lines='skip')
            
            df['timestamp'] = pd.to_datetime(
                df['date'] + ' ' + df['time'], format='mixed', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            df['node_id'] = pd.to_numeric(df['moteid'], errors='coerce')
            df = df[df['node_id'] == node_id]
            
            df['temp_dht'] = pd.to_numeric(df['temperature'], errors='coerce')
            df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
            
            # Use Intel Berkeley 'light' column for light_level
            df['light_level'] = pd.to_numeric(df['light'], errors='coerce')
            
            # Generate correlated values for missing sensors
            n = len(df)
            df['temp_therm'] = df['temp_dht'] + np.random.randn(n) * 0.3
            df['sound_level'] = 45 + np.random.randn(n) * 4
            df['flame_intensity'] = 8 + np.abs(np.random.randn(n) * 2)
            
            return df[['timestamp'] + config.FEATURE_NAMES]
            
        except Exception as e:
            raise ValueError(f"TXT parsing failed: {e}")
    
    def parse_serial_line(self, line: str) -> Optional[dict]:
        """
        Parse a multi-sensor serial line from Arduino.
        
        Expected format: temp_dht,humidity,temp_therm,sound_level,light_level,flame_intensity
        Example: 22.50,55.00,22.80,312.00,678.00,15.00
        
        Also handles legacy 2-field format: temperature,humidity
        """
        line = line.strip()
        if not line or line == "ERROR" or line.startswith("MULTI_SENSOR_READY") or line.startswith("ACK:"):
            return None
        
        try:
            parts = line.split(",")
            
            if len(parts) == 6:
                # Full multi-sensor format
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
                # Legacy 2-field format (backwards compatible)
                reading = {
                    'timestamp': datetime.now(),
                    'temp_dht': float(parts[0]),
                    'humidity': float(parts[1]),
                    'temp_therm': float(parts[0]) + np.random.randn() * 0.3,
                    'sound_level': 45 + np.random.randn() * 4,
                    'light_level': 500 + np.random.randn() * 15,
                    'flame_intensity': 8 + abs(np.random.randn() * 2),
                }
            else:
                return None
            
            # Validate each sensor against its range
            for feat in config.FEATURE_NAMES:
                if feat in reading:
                    lo, hi = config.SENSOR_RANGES[feat]
                    if not (lo <= reading[feat] <= hi):
                        return None  # Invalid reading
            
            return reading
            
        except (ValueError, IndexError):
            return None
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply physical validation and remove invalid readings.
        Includes per-sensor range checks and cross-validation.
        """
        df = df.copy()
        df['is_valid'] = True
        
        # Drop NaT timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Per-sensor range validation
        for feat in config.FEATURE_NAMES:
            if feat not in df.columns:
                continue
            lo, hi = config.SENSOR_RANGES[feat]
            invalid = (df[feat] < lo) | (df[feat] > hi)
            df.loc[invalid, 'is_valid'] = False
            n_invalid = invalid.sum()
            if n_invalid > 0:
                print(f"  {feat}: {n_invalid} out-of-range values")
        
        # Flag NaN values
        for feat in config.FEATURE_NAMES:
            if feat in df.columns:
                df.loc[df[feat].isna(), 'is_valid'] = False
        
        # Cross-validation: DHT11 vs Thermistor
        if 'temp_dht' in df.columns and 'temp_therm' in df.columns:
            temp_diff = abs(df['temp_dht'] - df['temp_therm'])
            divergent = temp_diff > config.TEMP_CROSS_VALIDATION_MAX_DIFF
            n_divergent = divergent.sum()
            if n_divergent > 0:
                print(f"  Cross-validation: {n_divergent} divergent temp readings "
                      f"(>{config.TEMP_CROSS_VALIDATION_MAX_DIFF}°C diff)")
                self.cross_validation_warnings.append({
                    'count': n_divergent,
                    'max_diff': temp_diff.max()
                })
            # Note: we flag but DON'T invalidate divergent readings —
            # the LSTM should learn to detect this as anomalous
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\nData validation summary:")
        print(f"  Total records: {len(df)}")
        print(f"  Valid records: {df['is_valid'].sum()}")
        print(f"  Invalid records: {(~df['is_valid']).sum()}")
        if len(df) > 0:
            print(f"  Invalid rate: {(~df['is_valid']).sum() / len(df) * 100:.2f}%")
        
        return df
    
    def get_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only valid sensor readings."""
        cols = ['timestamp'] + [c for c in config.FEATURE_NAMES if c in df.columns]
        return df[df['is_valid']][cols].copy()
    
    def detect_data_gaps(self, df: pd.DataFrame,
                          max_gap_seconds: int = 5) -> pd.DataFrame:
        """Detect communication failures (large time gaps)."""
        df = df.sort_values('timestamp')
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > max_gap_seconds]
        
        if len(gaps) > 0:
            print(f"\nDetected {len(gaps)} communication gaps (>{max_gap_seconds}s):")
            for idx in gaps.index[:5]:  # Show first 5
                print(f"  At {df.loc[idx, 'timestamp']}: {time_diffs.loc[idx]:.1f}s gap")
            if len(gaps) > 5:
                print(f"  ... and {len(gaps) - 5} more")
        
        return gaps
    
    def load_and_process(self, filepath: str,
                          file_type: str = 'auto') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: load, validate, and return datasets.
        """
        if file_type == 'auto':
            if filepath.endswith('.csv'):
                file_type = 'csv'
            elif filepath.endswith('.txt'):
                file_type = 'txt'
            else:
                raise ValueError(f"Cannot auto-detect file type: {filepath}")
        
        print(f"\nLoading {file_type.upper()} file: {filepath}")
        
        if file_type == 'csv':
            df = self.load_csv(filepath)
        elif file_type == 'txt':
            df = self.load_txt(filepath, node_id=config.INTEL_NODE_ID)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        print(f"Loaded {len(df)} raw records ({len(df.columns)-1} features)")
        
        full_data = self.validate_and_clean(df)
        clean_data = self.get_clean_data(full_data)
        self.detect_data_gaps(clean_data)
        
        self.raw_data = full_data
        self.clean_data = clean_data
        
        return full_data, clean_data


# Example usage
if __name__ == "__main__":
    ingestion = SensorDataIngestion()
    
    # Test serial parsing
    test_lines = [
        "22.50,55.00,22.80,312.00,678.00,15.00",  # Valid 6-field
        "22.50,55.00",                               # Valid 2-field (legacy)
        "ERROR",                                     # Error
        "MULTI_SENSOR_READY",                        # Startup message
        "invalid_data",                              # Invalid
    ]
    
    print("=== Serial Line Parsing Test ===")
    for line in test_lines:
        result = ingestion.parse_serial_line(line)
        status = "PARSED" if result else "SKIPPED"
        print(f"  '{line}' → {status}")
        if result:
            for k, v in result.items():
                if k != 'timestamp':
                    print(f"    {k}: {v}")
