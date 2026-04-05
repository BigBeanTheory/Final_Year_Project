"""
Multi-Sensor Data Generator
Creates synthetic 6-channel sensor data with realistic correlated patterns
and fault injection for training and testing the LSTM autoencoder.

Features generated:
  1. temp_dht        — DHT11 temperature (°C)
  2. humidity        — DHT11 humidity (%)
  3. temp_therm      — Thermistor temperature (°C), correlated with DHT11
  4. sound_level     — Acoustic RMS level
  5. light_level     — LDR reading
  6. flame_intensity — Flame sensor (low = normal, high = fire)

Usage:
    python iot_data_generator.py --output data/multi_sensor_test.csv --samples 5000
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


class MultiSensorDataGenerator:
    """
    Generate synthetic multi-sensor data with cross-correlated channels
    and realistic fault patterns for predictive maintenance.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Baseline values (typical indoor/equipment conditions)
        self.baselines = {
            'temp_dht':        22.0,    # °C
            'humidity':        55.0,    # %
            'temp_therm':      22.0,    # °C (should track DHT11)
            'sound_level':     45.0,    # RMS (quiet equipment hum)
            'light_level':     500.0,   # LDR mid-range
            'flame_intensity': 8.0,     # Low (no fire)
        }
        
        # Normal noise levels per sensor
        self.noise_levels = {
            'temp_dht':        0.3,
            'humidity':        1.5,
            'temp_therm':      0.25,
            'sound_level':     4.0,
            'light_level':     15.0,
            'flame_intensity': 2.0,
        }
        
        self.sample_period = 1.0  # seconds
    
    def generate_healthy(self, n_samples: int) -> pd.DataFrame:
        """
        Generate healthy multi-sensor readings with natural variance
        and realistic cross-correlations.
        """
        timestamps = [
            datetime(2024, 1, 1) + timedelta(seconds=i)
            for i in range(n_samples)
        ]
        
        # --- Temperature (DHT11) with slow random walk ---
        temp_walk = np.cumsum(np.random.randn(n_samples) * 0.02)
        temp_dht = self.baselines['temp_dht'] + temp_walk + \
                   np.random.randn(n_samples) * self.noise_levels['temp_dht']
        
        # --- Humidity: inversely correlated with temperature ---
        hum_walk = np.cumsum(np.random.randn(n_samples) * 0.05)
        humidity = self.baselines['humidity'] + hum_walk + \
                   np.random.randn(n_samples) * self.noise_levels['humidity']
        humidity -= (temp_dht - self.baselines['temp_dht']) * 0.8  # Anti-correlation
        
        # --- Thermistor: tracks DHT11 closely but with independent noise ---
        # Small constant offset (calibration difference) + correlated noise
        therm_offset = np.random.uniform(-0.5, 0.5)  # Fixed calibration offset
        temp_therm = temp_dht + therm_offset + \
                     np.random.randn(n_samples) * self.noise_levels['temp_therm']
        
        # --- Sound level: slow diurnal pattern + noise ---
        # Equipment hum varies slightly over time
        diurnal = 5 * np.sin(2 * np.pi * np.arange(n_samples) / (3600 * 8))  # 8-hour cycle
        sound_walk = np.cumsum(np.random.randn(n_samples) * 0.1)
        sound_level = self.baselines['sound_level'] + diurnal + sound_walk + \
                      np.random.randn(n_samples) * self.noise_levels['sound_level']
        
        # --- Light level: stable with occasional slow changes ---
        light_walk = np.cumsum(np.random.randn(n_samples) * 0.5)
        light_level = self.baselines['light_level'] + light_walk + \
                      np.random.randn(n_samples) * self.noise_levels['light_level']
        
        # --- Flame sensor: very low, stable (no fire) ---
        flame_intensity = self.baselines['flame_intensity'] + \
                          np.abs(np.random.randn(n_samples) * self.noise_levels['flame_intensity'])
        
        # Clip to physical ranges
        temp_dht = np.clip(temp_dht, 15, 35)
        humidity = np.clip(humidity, 25, 80)
        temp_therm = np.clip(temp_therm, 15, 35)
        sound_level = np.clip(sound_level, 5, 200)
        light_level = np.clip(light_level, 50, 950)
        flame_intensity = np.clip(flame_intensity, 0, 50)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'temp_dht': np.round(temp_dht, 2),
            'humidity': np.round(humidity, 2),
            'temp_therm': np.round(temp_therm, 2),
            'sound_level': np.round(sound_level, 2),
            'light_level': np.round(light_level, 2),
            'flame_intensity': np.round(flame_intensity, 2),
        })
    
    # ------------------------------------------------------------------
    # FAULT INJECTION
    # ------------------------------------------------------------------
    
    def inject_drift(self, df: pd.DataFrame, start_idx: int,
                     feature: str = 'temp_dht', rate: float = 0.008) -> pd.DataFrame:
        """Inject gradual sensor drift (aging/calibration error)."""
        df = df.copy()
        n_drift = len(df) - start_idx
        if n_drift > 0:
            drift = np.linspace(0, rate * n_drift, n_drift)
            df.loc[start_idx:, feature] += drift
        return df
    
    def inject_noise_burst(self, df: pd.DataFrame, start_idx: int,
                           duration: int = 100,
                           features: list = None,
                           multiplier: float = 3.0) -> pd.DataFrame:
        """Inject increased noise across specified sensors."""
        df = df.copy()
        if features is None:
            features = ['temp_dht', 'humidity', 'sound_level']
        
        end_idx = min(start_idx + duration, len(df))
        n = end_idx - start_idx
        
        for feat in features:
            if feat in df.columns:
                noise = np.random.randn(n) * self.noise_levels.get(feat, 1.0) * multiplier
                df.loc[start_idx:end_idx-1, feat] += noise
        
        return df
    
    def inject_freeze(self, df: pd.DataFrame, start_idx: int,
                      duration: int = 30,
                      features: list = None) -> pd.DataFrame:
        """Inject stuck sensor readings."""
        df = df.copy()
        if features is None:
            features = ['temp_dht', 'humidity']
        
        end_idx = min(start_idx + duration, len(df))
        for feat in features:
            if feat in df.columns:
                frozen_val = df.loc[start_idx, feat]
                df.loc[start_idx:end_idx-1, feat] = frozen_val
        
        return df
    
    def inject_sensor_divergence(self, df: pd.DataFrame, start_idx: int,
                                 duration: int = 150,
                                 divergence: float = 8.0) -> pd.DataFrame:
        """
        Inject divergence between DHT11 and thermistor.
        Simulates one sensor degrading while the other stays accurate.
        """
        df = df.copy()
        end_idx = min(start_idx + duration, len(df))
        n = end_idx - start_idx
        
        # Gradual divergence
        ramp = np.linspace(0, divergence, n)
        df.loc[start_idx:end_idx-1, 'temp_therm'] += ramp
        
        return df
    
    def inject_acoustic_event(self, df: pd.DataFrame, start_idx: int,
                              duration: int = 80,
                              intensity: float = 3.0) -> pd.DataFrame:
        """
        Inject acoustic anomaly (equipment vibration/mechanical fault).
        Sound level increases dramatically.
        """
        df = df.copy()
        end_idx = min(start_idx + duration, len(df))
        n = end_idx - start_idx
        
        # Sustained elevated sound with high variance
        sound_boost = intensity * self.baselines['sound_level'] * \
                      (0.5 + np.random.rand(n) * 0.5)
        df.loc[start_idx:end_idx-1, 'sound_level'] += sound_boost
        
        return df
    
    def inject_fire_event(self, df: pd.DataFrame, start_idx: int,
                          duration: int = 40) -> pd.DataFrame:
        """
        Inject fire hazard event.
        Flame sensor spikes, temperature rises, light may change.
        """
        df = df.copy()
        end_idx = min(start_idx + duration, len(df))
        n = end_idx - start_idx
        
        # Flame sensor goes high
        df.loc[start_idx:end_idx-1, 'flame_intensity'] = \
            600 + np.random.randn(n) * 50
        
        # Temperature also rises
        temp_rise = np.linspace(0, 8, n)
        df.loc[start_idx:end_idx-1, 'temp_dht'] += temp_rise
        df.loc[start_idx:end_idx-1, 'temp_therm'] += temp_rise * 0.9
        
        # Light level changes (flames emit light)
        df.loc[start_idx:end_idx-1, 'light_level'] += \
            100 + np.random.randn(n) * 30
        
        return df
    
    def inject_light_anomaly(self, df: pd.DataFrame, start_idx: int,
                             duration: int = 60) -> pd.DataFrame:
        """
        Inject light anomaly (equipment LED flickering, sparking, etc.)
        """
        df = df.copy()
        end_idx = min(start_idx + duration, len(df))
        n = end_idx - start_idx
        
        # Rapid light fluctuations
        flicker = 200 * np.sin(np.arange(n) * 0.5) + np.random.randn(n) * 80
        df.loc[start_idx:end_idx-1, 'light_level'] += flicker
        
        return df
    
    def inject_comm_failures(self, df: pd.DataFrame,
                             failure_rate: float = 0.02) -> pd.DataFrame:
        """Inject random communication failures (NaN values)."""
        df = df.copy()
        n_failures = int(len(df) * failure_rate)
        failure_indices = np.random.choice(len(df), n_failures, replace=False)
        
        for feat in ['temp_dht', 'humidity', 'temp_therm', 'sound_level',
                      'light_level', 'flame_intensity']:
            df.loc[failure_indices, feat] = np.nan
        
        return df
    
    # ------------------------------------------------------------------
    # SCENARIO GENERATORS
    # ------------------------------------------------------------------
    
    def generate_realistic_scenario(self, total_samples: int = 5000) -> pd.DataFrame:
        """
        Generate realistic multi-sensor data with multiple fault patterns.
        
        Timeline:
        - 0-35%:   Healthy operation (training baseline)
        - 35-45%:  Temperature drift begins
        - 45-55%:  Sensor divergence (DHT11 vs thermistor)
        - 55-65%:  Acoustic anomaly (equipment vibration)
        - 65-72%:  Noise burst across multiple sensors
        - 72-78%:  Sensor freeze event
        - 78-85%:  Light anomaly (flickering)
        - 85-90%:  Fire hazard event
        - 90-100%: Partial recovery
        """
        print(f"\n{'='*60}")
        print(f"MULTI-SENSOR DATA GENERATOR")
        print(f"{'='*60}")
        print(f"Generating {total_samples} multi-sensor readings...")
        
        df = self.generate_healthy(total_samples)
        
        # Phase boundaries
        p1 = int(total_samples * 0.35)   # End of healthy
        p2 = int(total_samples * 0.45)   # End of drift
        p3 = int(total_samples * 0.55)   # End of divergence
        p4 = int(total_samples * 0.65)   # End of acoustic
        p5 = int(total_samples * 0.72)   # End of noise
        p6 = int(total_samples * 0.78)   # End of freeze
        p7 = int(total_samples * 0.85)   # End of light anomaly
        p8 = int(total_samples * 0.90)   # End of fire event
        
        print(f"  Phase 1 (0-{p1}): Healthy operation")
        
        print(f"  Phase 2 ({p1}-{p2}): Temperature drift")
        df = self.inject_drift(df, p1, feature='temp_dht', rate=0.006)
        
        print(f"  Phase 3 ({p2}-{p3}): Sensor divergence (DHT11 vs thermistor)")
        df = self.inject_sensor_divergence(df, p2, duration=p3-p2, divergence=7.0)
        
        print(f"  Phase 4 ({p3}-{p4}): Acoustic anomaly")
        df = self.inject_acoustic_event(df, p3, duration=p4-p3, intensity=2.5)
        
        print(f"  Phase 5 ({p4}-{p5}): Multi-sensor noise burst")
        df = self.inject_noise_burst(df, p4, duration=p5-p4,
                                     features=['temp_dht', 'humidity', 'sound_level'],
                                     multiplier=3.0)
        
        print(f"  Phase 6 ({p5}-{p6}): Sensor freeze")
        df = self.inject_freeze(df, p5, duration=p6-p5,
                                features=['temp_dht', 'humidity'])
        
        print(f"  Phase 7 ({p6}-{p7}): Light anomaly")
        df = self.inject_light_anomaly(df, p6, duration=p7-p6)
        
        print(f"  Phase 8 ({p7}-{p8}): Fire hazard event")
        df = self.inject_fire_event(df, p7, duration=p8-p7)
        
        # Sprinkle communication failures
        print(f"  Adding random communication failures (1.5% rate)")
        df = self.inject_comm_failures(df, failure_rate=0.015)
        
        print(f"\nGenerated {len(df)} multi-sensor samples with {8} fault phases")
        
        return df
    
    def generate_training_data(self, total_samples: int = 5000) -> pd.DataFrame:
        """
        Generate ONLY healthy data for model training.
        No faults injected — this is the 'normal baseline'.
        """
        print(f"Generating {total_samples} healthy training samples...")
        df = self.generate_healthy(total_samples)
        
        # Drop any NaN that may exist
        df = df.dropna().reset_index(drop=True)
        
        print(f"Training data shape: {df.shape}")
        return df
    
    def save_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save to CSV format."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Saved CSV to {filepath}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-sensor data for predictive maintenance"
    )
    parser.add_argument('--output', type=str, default='data/multi_sensor_test.csv',
                        help='Output CSV file path')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, choices=['full', 'train', 'both'],
                        default='both',
                        help='full=with faults, train=healthy only, both=generate both')
    return parser.parse_args()


def main():
    args = parse_args()
    
    generator = MultiSensorDataGenerator(seed=args.seed)
    
    base_path = args.output.rsplit('.', 1)[0]
    
    if args.mode in ['full', 'both']:
        df_full = generator.generate_realistic_scenario(args.samples)
        generator.save_csv(df_full, base_path + '.csv')
        
        print(f"\nData Statistics (full scenario):")
        for col in ['temp_dht', 'humidity', 'temp_therm', 'sound_level',
                     'light_level', 'flame_intensity']:
            print(f"  {col}: [{df_full[col].min():.2f}, {df_full[col].max():.2f}]")
        print(f"  Missing values: {df_full.isna().sum().sum()}")
    
    if args.mode in ['train', 'both']:
        df_train = generator.generate_training_data(args.samples)
        generator.save_csv(df_train, base_path + '_healthy.csv')
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
