"""
Preprocessing Module (Multi-Sensor)
Handles normalization, outlier removal, and sliding window generation
for 6-feature LSTM input.

Features: temp_dht, humidity, temp_therm, sound_level, light_level, flame_intensity

Notes:
- Uses IQR method for outlier removal (robust to sensor spikes)
- Min-Max scaling per feature (different physical ranges)
- Cross-validation feature engineering: |temp_dht - temp_therm|
- Overlapping windows (stride=1) maximize training samples
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler
import pickle

class SensorPreprocessor:
    """
    Multi-sensor preprocessing pipeline.
    """
    
    def __init__(self, window_size: int = 20, stride: int = 1, feature_columns: List[str] = None):
        """
        Args:
            window_size: Number of timesteps per window
            stride: Step size for sliding window (1 = maximum overlap)
            feature_columns: List of feature column names
        """
        self.window_size = window_size
        self.stride = stride
        
        # Default to 6-feature multi-sensor setup
        if feature_columns is None:
            self.feature_columns = [
                'temp_dht', 'humidity', 'temp_therm',
                'sound_level', 'light_level', 'flame_intensity'
            ]
        else:
            self.feature_columns = feature_columns
        
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def remove_outliers_iqr(self, df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
        """
        Remove statistical outliers using Interquartile Range method.
        Applied independently per feature (each has different physical range).
        """
        df = df.copy()
        
        for col in self.feature_columns:
            if col not in df.columns:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                print(f"  {col}: removed {n_outliers} outliers "
                      f"(range: [{lower_bound:.2f}, {upper_bound:.2f}])")
                df = df[~outliers]
        
        return df.reset_index(drop=True)
    
    def fit_scaler(self, df: pd.DataFrame, remove_outliers: bool = True) -> None:
        """
        Fit normalization scaler on training (healthy) data.
        Each feature is scaled independently to [0, 1].
        """
        df = df.copy()
        
        if remove_outliers:
            print("Removing statistical outliers from training data:")
            df = self.remove_outliers_iqr(df)
        
        # Only fit on columns that exist
        available = [c for c in self.feature_columns if c in df.columns]
        self.scaler.fit(df[available])
        self.is_fitted = True
        
        print(f"\nScaler fitted on {len(df)} samples across {len(available)} features")
        for col in available:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization to sensor values."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        df = df.copy()
        available = [c for c in self.feature_columns if c in df.columns]
        df[available] = self.scaler.transform(df[available])
        return df
    
    def create_sequences(self, data: np.ndarray, include_targets: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding window sequences for LSTM input.
        
        Returns:
            X: array of shape (n_samples, window_size, n_features)
            y: array of shape (n_samples, window_size, n_features) if include_targets
        """
        sequences = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            sequences.append(window)
        
        X = np.array(sequences)
        
        if include_targets:
            y = X.copy()
            return X, y
        else:
            return X, None
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for multi-sensor training data.
        
        Steps:
        1. Remove outliers (per feature)
        2. Fit and apply scaler
        3. Create sequences
        """
        print("\n=== Preparing Multi-Sensor Training Data ===")
        
        # Fit scaler (includes outlier removal)
        self.fit_scaler(df, remove_outliers=True)
        
        # Normalize
        df_norm = self.normalize(df)
        
        # Convert to numpy
        available = [c for c in self.feature_columns if c in df_norm.columns]
        data = df_norm[available].values
        
        # Create sequences
        X, y = self.create_sequences(data, include_targets=True)
        
        print(f"Created {len(X)} training windows")
        print(f"Input shape: {X.shape}  (samples, timesteps, features)")
        
        return X, y
    
    def prepare_inference_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Preprocessing pipeline for inference/testing data.
        Uses pre-fitted scaler. Does NOT remove outliers.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Train model first.")
        
        print("\n=== Preparing Multi-Sensor Inference Data ===")
        
        df_norm = self.normalize(df)
        available = [c for c in self.feature_columns if c in df_norm.columns]
        data = df_norm[available].values
        
        X, _ = self.create_sequences(data, include_targets=False)
        
        df_aligned = df.iloc[self.window_size - 1::self.stride].reset_index(drop=True)
        
        print(f"Created {len(X)} inference windows")
        print(f"Input shape: {X.shape}")
        
        return X, df_aligned
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert normalized data back to original scale."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")
        
        original_shape = data.shape
        if len(original_shape) == 3:
            batch, timesteps, features = original_shape
            data_2d = data.reshape(-1, features)
            data_inv = self.scaler.inverse_transform(data_2d)
            return data_inv.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)
    
    def save(self, filepath: str) -> None:
        """Save fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'window_size': self.window_size,
                'stride': self.stride,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load fitted preprocessor from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.scaler = state['scaler']
        self.window_size = state['window_size']
        self.stride = state['stride']
        self.feature_columns = state['feature_columns']
        self.is_fitted = True
        print(f"Preprocessor loaded from {filepath} ({len(self.feature_columns)} features)")


def split_train_test_temporal(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data temporally (no shuffling).
    First train_ratio = training (assumed healthy).
    """
    split_idx = int(len(df) * train_ratio)
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    print(f"\n=== Temporal Split ===")
    print(f"Training period: {train['timestamp'].min()} to {train['timestamp'].max()}")
    print(f"Testing period: {test['timestamp'].min()} to {test['timestamp'].max()}")
    print(f"Train samples: {len(train)} ({train_ratio*100:.0f}%)")
    print(f"Test samples: {len(test)} ({(1-train_ratio)*100:.0f}%)")
    
    return train, test


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1S'),
        'temp_dht': 22 + np.random.randn(n) * 0.5,
        'humidity': 55 + np.random.randn(n) * 2,
        'temp_therm': 22 + np.random.randn(n) * 0.4,
        'sound_level': 50 + np.random.randn(n) * 5,
        'light_level': 500 + np.random.randn(n) * 20,
        'flame_intensity': 10 + np.random.randn(n) * 3,
    })
    
    # Add some outliers
    df.loc[100, 'temp_dht'] = 100
    df.loc[200, 'sound_level'] = 900
    
    train, test = split_train_test_temporal(df, train_ratio=0.7)
    
    preprocessor = SensorPreprocessor(window_size=20, stride=1)
    X_train, y_train = preprocessor.prepare_training_data(train)
    X_test, df_test_aligned = preprocessor.prepare_inference_data(test)
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
