"""
Sample Data Generator
Creates synthetic DHT11 sensor data with realistic fault patterns for testing.

Usage:
    python generate_sample_data.py --output data/sensor_test.csv --samples 2000

Engineering Notes:
- Generates realistic sensor behavior with faults
- Useful for testing without physical hardware
- Includes multiple fault patterns: drift, noise, freeze, comm failures
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SensorDataGenerator:
    """
    Generate synthetic DHT11 sensor data with controlled fault patterns.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Baseline values (typical indoor conditions)
        self.temp_baseline = 22.0  # °C
        self.temp_noise = 0.3
        self.humidity_baseline = 55.0  # %
        self.humidity_noise = 1.5
        
        # Sampling rate
        self.sample_period = 1.0  # seconds
    
    def generate_healthy(self, n_samples: int) -> pd.DataFrame:
        """
        Generate healthy sensor readings with natural variance.
        """
        timestamps = [
            datetime(2024, 1, 1) + timedelta(seconds=i)
            for i in range(n_samples)
        ]
        
        # White noise + small random walk
        temp_walk = np.cumsum(np.random.randn(n_samples) * 0.05)
        temperature = self.temp_baseline + temp_walk + np.random.randn(n_samples) * self.temp_noise
        
        hum_walk = np.cumsum(np.random.randn(n_samples) * 0.1)
        humidity = self.humidity_baseline + hum_walk + np.random.randn(n_samples) * self.humidity_noise
        
        # Slight correlation (humidity decreases as temp increases)
        humidity -= (temperature - self.temp_baseline) * 0.5
        
        # Clip to physical limits
        temperature = np.clip(temperature, 15, 30)
        humidity = np.clip(humidity, 30, 80)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': humidity
        })
    
    def inject_drift(self, df: pd.DataFrame, start_idx: int, rate: float = 0.01) -> pd.DataFrame:
        """
        Inject gradual sensor drift (simulates aging/calibration error).
        
        Args:
            df: DataFrame to modify
            start_idx: Where to start drift
            rate: Drift rate per sample (e.g., 0.01 = 1% increase per reading)
        """
        df = df.copy()
        n_drift = len(df) - start_idx
        
        if n_drift > 0:
            drift_multiplier = 1 + np.linspace(0, rate * n_drift, n_drift)
            df.loc[start_idx:, 'temperature'] *= drift_multiplier
        
        return df
    
    def inject_noise(self, df: pd.DataFrame, start_idx: int, noise_multiplier: float = 3.0) -> pd.DataFrame:
        """
        Inject increased noise (simulates electrical interference).
        """
        df = df.copy()
        n_noisy = len(df) - start_idx
        
        if n_noisy > 0:
            extra_noise_temp = np.random.randn(n_noisy) * self.temp_noise * noise_multiplier
            extra_noise_hum = np.random.randn(n_noisy) * self.humidity_noise * noise_multiplier
            
            df.loc[start_idx:, 'temperature'] += extra_noise_temp
            df.loc[start_idx:, 'humidity'] += extra_noise_hum
        
        return df
    
    def inject_freeze(self, df: pd.DataFrame, start_idx: int, duration: int = 20) -> pd.DataFrame:
        """
        Inject stuck sensor readings (simulates sensor freeze).
        """
        df = df.copy()
        
        if start_idx < len(df):
            frozen_temp = df.loc[start_idx, 'temperature']
            frozen_hum = df.loc[start_idx, 'humidity']
            
            end_idx = min(start_idx + duration, len(df))
            df.loc[start_idx:end_idx, 'temperature'] = frozen_temp
            df.loc[start_idx:end_idx, 'humidity'] = frozen_hum
        
        return df
    
    def inject_comm_failures(self, df: pd.DataFrame, failure_rate: float = 0.02) -> pd.DataFrame:
        """
        Inject random communication failures (missing readings).
        
        Args:
            failure_rate: Probability of failure per sample
        """
        df = df.copy()
        
        # Randomly select samples to drop
        n_failures = int(len(df) * failure_rate)
        failure_indices = np.random.choice(len(df), n_failures, replace=False)
        
        df.loc[failure_indices, 'temperature'] = np.nan
        df.loc[failure_indices, 'humidity'] = np.nan
        
        return df
    
    def inject_invalid_readings(self, df: pd.DataFrame, invalid_rate: float = 0.01) -> pd.DataFrame:
        """
        Inject physically invalid readings (sensor malfunction).
        """
        df = df.copy()
        
        n_invalid = int(len(df) * invalid_rate)
        invalid_indices = np.random.choice(len(df), n_invalid, replace=False)
        
        for idx in invalid_indices:
            if np.random.rand() > 0.5:
                # Invalid temperature
                df.loc[idx, 'temperature'] = np.random.choice([-999, 999, 150, -50])
            else:
                # Invalid humidity
                df.loc[idx, 'humidity'] = np.random.choice([-50, 150, 200])
        
        return df
    
    def generate_realistic_scenario(self, total_samples: int = 2000) -> pd.DataFrame:
        """
        Generate realistic sensor data with multiple fault patterns.
        
        Timeline:
        - 0-40%: Healthy operation
        - 40-60%: Gradual drift begins
        - 60-75%: Increased noise
        - 75-85%: Sensor freeze event
        - 85-100%: Recovery with residual drift
        """
        print(f"Generating {total_samples} sensor readings...")
        
        # Generate base healthy data
        df = self.generate_healthy(total_samples)
        
        # Calculate phase boundaries
        phase1_end = int(total_samples * 0.40)  # 40% healthy
        phase2_end = int(total_samples * 0.60)  # 20% drift
        phase3_end = int(total_samples * 0.75)  # 15% noise
        phase4_end = int(total_samples * 0.85)  # 10% freeze
        
        print(f"  Phase 1 (0-{phase1_end}): Healthy operation")
        
        print(f"  Phase 2 ({phase1_end}-{phase2_end}): Gradual drift")
        df = self.inject_drift(df, start_idx=phase1_end, rate=0.008)
        
        print(f"  Phase 3 ({phase2_end}-{phase3_end}): Increased noise")
        df = self.inject_noise(df, start_idx=phase2_end, noise_multiplier=2.5)
        
        print(f"  Phase 4 ({phase3_end}-{phase4_end}): Sensor freeze")
        df = self.inject_freeze(df, start_idx=phase3_end, duration=30)
        
        # Sprinkle in communication failures throughout
        print(f"  Adding random communication failures (2% rate)")
        df = self.inject_comm_failures(df, failure_rate=0.02)
        
        # Add occasional invalid readings
        print(f"  Adding invalid readings (1% rate)")
        df = self.inject_invalid_readings(df, invalid_rate=0.01)
        
        print(f"Generated {len(df)} samples with realistic fault patterns")
        
        return df
    
    def save_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save data to CSV format."""
        df.to_csv(filepath, index=False)
        print(f"Saved CSV to {filepath}")
    
    def save_log(self, df: pd.DataFrame, filepath: str) -> None:
        """Save data to log format."""
        with open(filepath, 'w') as f:
            for _, row in df.iterrows():
                ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Handle missing/invalid values
                if pd.isna(row['temperature']) or pd.isna(row['humidity']):
                    f.write(f"[{ts}] Sensor communication error\n")
                elif abs(row['temperature']) > 100 or abs(row['humidity']) > 100:
                    f.write(f"[{ts}] Sensor reading error: invalid data\n")
                else:
                    temp = row['temperature']
                    hum = row['humidity']
                    f.write(f"[{ts}] Temp: {temp:.2f}°C, Humidity: {hum:.2f}%\n")
        
        print(f"Saved LOG to {filepath}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic DHT11 sensor data"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/sensor_test.csv',
        help='Output file path'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'log', 'both'],
        default='both',
        help='Output format'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("DHT11 SENSOR DATA GENERATOR")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Samples: {args.samples}")
    print(f"Format: {args.format}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")
    
    # Create output directory
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Generate data
    generator = SensorDataGenerator(seed=args.seed)
    df = generator.generate_realistic_scenario(args.samples)
    
    # Save in requested format(s)
    base_path = args.output.rsplit('.', 1)[0]
    
    if args.format in ['csv', 'both']:
        csv_path = base_path + '.csv'
        generator.save_csv(df, csv_path)
    
    if args.format in ['log', 'both']:
        log_path = base_path + '.log'
        generator.save_log(df, log_path)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    
    # Quick statistics
    print(f"\nData Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Temperature range: [{df['temperature'].min():.2f}, {df['temperature'].max():.2f}]")
    print(f"  Humidity range: [{df['humidity'].min():.2f}, {df['humidity'].max():.2f}]")
    print(f"  Missing values: {df.isna().sum().sum()}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
