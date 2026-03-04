"""
Data Ingestion Module
Handles parsing of CSV and log files, validation, and unified time-series output.

Engineering Notes:
- DHT11 valid ranges: Temp: 0-50°C, Humidity: 20-90%
- Handles multiple error formats: "error", "nan", physically impossible values
- Preserves original timestamps for temporal analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Tuple, Optional

class SensorDataIngestion:
    """
    Unified parser for IoT sensor data from multiple formats.
    """
    
    # DHT11 physical limits
    TEMP_MIN, TEMP_MAX = -10, 60  # Conservative range (DHT11 spec: 0-50, but allow margin)
    HUMIDITY_MIN, HUMIDITY_MAX = 0, 100
    
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV format sensor data.
        Expected columns: timestamp, temperature, humidity
        """
        try:
            df = pd.read_csv(filepath)
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.strip()
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], errors='coerce')
            else:
                # Generate sequential timestamps if missing
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1S')
            
            # Ensure numeric types for sensor values
            df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
            df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
            
            return df[['timestamp', 'temperature', 'humidity']]
            
        except Exception as e:
            raise ValueError(f"CSV parsing failed: {e}")
    
    def load_log(self, filepath: str) -> pd.DataFrame:
        """
        Load log format sensor data.
        Expected format: [timestamp] Temp: XX.X°C, Humidity: XX.X%
        or similar variants with "error" messages
        """
        records = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Extract timestamp (multiple formats)
                ts_match = re.search(r'\[(.*?)\]|\b(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                timestamp = None
                if ts_match:
                    ts_str = ts_match.group(1) or ts_match.group(2)
                    try:
                        timestamp = pd.to_datetime(ts_str)
                    except:
                        pass
                
                # Extract temperature
                temp_match = re.search(r'temp(?:erature)?[:\s]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                temperature = float(temp_match.group(1)) if temp_match else np.nan
                
                # Extract humidity
                hum_match = re.search(r'hum(?:idity)?[:\s]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                humidity = float(hum_match.group(1)) if hum_match else np.nan
                
                # Check for explicit error markers
                if 'error' in line.lower() or 'fail' in line.lower():
                    temperature = np.nan
                    humidity = np.nan
                
                records.append({
                    'timestamp': timestamp,
                    'temperature': temperature,
                    'humidity': humidity
                })
        
        df = pd.DataFrame(records)
        
        # Fill missing timestamps with sequential values
        if df['timestamp'].isna().any():
            first_valid = df['timestamp'].first_valid_index()
            if first_valid is not None:
                start_time = df.loc[first_valid, 'timestamp']
            else:
                start_time = datetime.now()
            df['timestamp'] = pd.date_range(start=start_time, periods=len(df), freq='1S')
        
        return df
    
    def load_txt(self, filepath: str, node_id: float = 7.0) -> pd.DataFrame:
        """
        Load Intel Berkeley Lab format sensor data (.txt).
        Space-separated format with specific columns.
        """
        # Read the space-delimited text file
        # columns: date time epoch moteid temperature humidity light voltage
        columns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']
        try:
            # We'll read everything and handle errors
            df = pd.read_csv(filepath, sep=r'\s+', header=None, names=columns, on_bad_lines='skip')
            
            # Combine date and time
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            # Convert nodeid
            df['node_id'] = pd.to_numeric(df['moteid'], errors='coerce')
            
            # Filter for the specific node (e.g., node 7.0 was determined to be reliable)
            df = df[df['node_id'] == node_id]
            
            # Convert sensor values
            df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
            df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
            
            return df[['timestamp', 'temperature', 'humidity']]
            
        except Exception as e:
            raise ValueError(f"TXT parsing failed: {e}")
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply physical validation and remove invalid readings.
        
        Validation rules:
        1. Remove rows where timestamp is NaT
        2. Flag physically impossible values (outside DHT11 range)
        3. Remove NaN values
        4. Remove duplicate timestamps (keep first)
        
        Returns cleaned DataFrame with 'is_valid' flag for debugging
        """
        df = df.copy()
        df['is_valid'] = True
        
        # Drop NaT timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Flag invalid temperature
        temp_invalid = (df['temperature'] < self.TEMP_MIN) | (df['temperature'] > self.TEMP_MAX)
        df.loc[temp_invalid, 'is_valid'] = False
        
        # Flag invalid humidity
        hum_invalid = (df['humidity'] < self.HUMIDITY_MIN) | (df['humidity'] > self.HUMIDITY_MAX)
        df.loc[hum_invalid, 'is_valid'] = False
        
        # Flag NaN values
        df.loc[df['temperature'].isna() | df['humidity'].isna(), 'is_valid'] = False
        
        # Remove duplicates (keep first occurrence)
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Data validation summary:")
        print(f"  Total records: {len(df)}")
        print(f"  Valid records: {df['is_valid'].sum()}")
        print(f"  Invalid records: {(~df['is_valid']).sum()}")
        print(f"  Invalid rate: {(~df['is_valid']).sum() / len(df) * 100:.2f}%")
        
        return df
    
    def get_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only valid sensor readings for model training/inference.
        """
        return df[df['is_valid']][['timestamp', 'temperature', 'humidity']].copy()
    
    def detect_data_gaps(self, df: pd.DataFrame, max_gap_seconds: int = 5) -> pd.DataFrame:
        """
        Detect communication failures (large time gaps in data).
        Returns DataFrame with gap information.
        """
        df = df.sort_values('timestamp')
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > max_gap_seconds]
        
        if len(gaps) > 0:
            print(f"\nDetected {len(gaps)} communication gaps (>{max_gap_seconds}s):")
            for idx in gaps.index:
                gap_duration = time_diffs.loc[idx]
                print(f"  At {df.loc[idx, 'timestamp']}: {gap_duration:.1f}s gap")
        
        return gaps
    
    def load_and_process(self, filepath: str, file_type: str = 'auto') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: load, validate, and return both full and clean datasets.
        
        Args:
            filepath: Path to data file
            file_type: 'csv', 'log', or 'auto' (detect from extension)
        
        Returns:
            (full_data, clean_data) where full_data includes 'is_valid' flag
        """
        if file_type == 'auto':
            if filepath.endswith('.csv'):
                file_type = 'csv'
            elif filepath.endswith('.log'):
                file_type = 'log'
            elif filepath.endswith('.txt'):
                file_type = 'txt'
            else:
                raise ValueError(f"Cannot auto-detect file type for: {filepath}")
        
        print(f"\nLoading {file_type.upper()} file: {filepath}")
        
        if file_type == 'csv':
            df = self.load_csv(filepath)
        elif file_type == 'log':
            df = self.load_log(filepath)
        elif file_type == 'txt':
            from config import INTEL_NODE_ID
            df = self.load_txt(filepath, node_id=INTEL_NODE_ID)
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        print(f"Loaded {len(df)} raw records")
        
        # Validate and clean
        full_data = self.validate_and_clean(df)
        clean_data = self.get_clean_data(full_data)
        
        # Detect communication gaps
        self.detect_data_gaps(clean_data)
        
        self.raw_data = full_data
        self.clean_data = clean_data
        
        return full_data, clean_data


# Example usage for testing
if __name__ == "__main__":
    ingestion = SensorDataIngestion()
    
    # Test with a sample CSV
    import io
    sample_csv = """timestamp,temperature,humidity
2024-01-01 00:00:00,22.5,55.0
2024-01-01 00:00:01,22.6,error
2024-01-01 00:00:02,150.0,55.5
2024-01-01 00:00:03,23.0,56.0
"""
    
    import os
    test_path = 'test_sensor.csv'
    with open(test_path, 'w') as f:
        f.write(sample_csv)
    
    full, clean = ingestion.load_and_process(test_path, 'csv')
    print("\nClean data shape:", clean.shape)
    print(clean.head())
    
    if os.path.exists(test_path):
        os.remove(test_path)
    print("\nClean data shape:", clean.shape)
    print(clean.head())
