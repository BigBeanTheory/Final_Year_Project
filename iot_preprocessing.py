"""
Preprocessing Module
Handles normalization, outlier removal, and sliding window generation for LSTM.

Notes:
- Uses IQR method for outlier removal (more robust than z-score for sensor data)
- Min-Max scaling preserves temporal patterns better than standardization
- Overlapping windows (stride=1) maximize training samples and smooth predictions
- Window size of 20 chosen to balance context vs temporal resolution
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import pickle

class SensorPreprocessor:
    """
    Preprocessing pipeline for sensor time-series data.
    """
    
    def __init__(self, window_size: int = 20, stride: int = 1):
        """
        Args:
            window_size: Number of timesteps per window
            stride: Step size for sliding window (1 = maximum overlap)
        """
        self.window_size = window_size
        self.stride = stride
        self.scaler = MinMaxScaler()
        self.feature_columns = ['temperature', 'humidity']
        self.is_fitted = False
        
    def remove_outliers_iqr(self, df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
        """
        Remove statistical outliers using Interquartile Range method.
        
        IQR method is robust to extreme values and works well for sensor drift detection.
        Only removes extreme outliers, preserving subtle anomalies for model to learn.
        
        Args:
            factor: IQR multiplier (1.5 = standard, 3.0 = very conservative)
        """
        df = df.copy()
        
        for col in self.feature_columns:
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
        
        CRITICAL: Only fit on known-good data to establish "normal" baseline.
        """
        df = df.copy()
        
        if remove_outliers:
            print("Removing statistical outliers from training data:")
            df = self.remove_outliers_iqr(df)
        
        # Fit scaler
        self.scaler.fit(df[self.feature_columns])
        self.is_fitted = True
        
        print(f"\nScaler fitted on {len(df)} samples")
        print(f"Temperature range: [{df['temperature'].min():.2f}, {df['temperature'].max():.2f}]")
        print(f"Humidity range: [{df['humidity'].min():.2f}, {df['humidity'].max():.2f}]")
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply min-max normalization to sensor values.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        df = df.copy()
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df
    
    def create_sequences(self, data: np.ndarray, include_targets: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding window sequences for LSTM input.
        
        Args:
            data: 2D array of shape (timesteps, features)
            include_targets: If True, return (X, y) where y = X (for autoencoder training)
        
        Returns:
            X: array of shape (n_samples, window_size, n_features)
            y: array of shape (n_samples, window_size, n_features) if include_targets else None
        """
        sequences = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            sequences.append(window)
        
        X = np.array(sequences)
        
        if include_targets:
            # For autoencoder: target = input
            y = X.copy()
            return X, y
        else:
            return X, None
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for training data.
        
        Steps:
        1. Remove outliers
        2. Fit and apply scaler
        3. Create sequences
        
        Returns:
            X_train, y_train (both same for autoencoder)
        """
        print("\n=== Preparing Training Data ===")
        
        # Fit scaler (includes outlier removal)
        self.fit_scaler(df, remove_outliers=True)
        
        # Normalize
        df_norm = self.normalize(df)
        
        # Convert to numpy
        data = df_norm[self.feature_columns].values
        
        # Create sequences
        X, y = self.create_sequences(data, include_targets=True)
        
        print(f"Created {len(X)} training windows")
        print(f"Input shape: {X.shape}")
        
        return X, y
    
    def prepare_inference_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Preprocessing pipeline for inference/testing data.
        
        Uses pre-fitted scaler from training phase.
        Does NOT remove outliers (we want to detect them).
        
        Returns:
            X_test: sequences
            df_aligned: DataFrame aligned with sequence timestamps
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Train model first.")
        
        print("\n=== Preparing Inference Data ===")
        
        # Normalize (no outlier removal)
        df_norm = self.normalize(df)
        
        # Convert to numpy
        data = df_norm[self.feature_columns].values
        
        # Create sequences
        X, _ = self.create_sequences(data, include_targets=False)
        
        # Align DataFrame with sequences (each sequence ends at a specific timestamp)
        # Sequence i corresponds to df.iloc[i:i+window_size], so timestamp is df.iloc[i+window_size-1]
        df_aligned = df.iloc[self.window_size - 1::self.stride].reset_index(drop=True)
        
        print(f"Created {len(X)} inference windows")
        print(f"Input shape: {X.shape}")
        print(f"Aligned timestamps: {len(df_aligned)}")
        
        return X, df_aligned
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        Useful for visualization and debugging.
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted.")
        
        # Handle 3D input (batch, timesteps, features)
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
        print(f"Preprocessor loaded from {filepath}")


def split_train_test_temporal(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data temporally (no shuffling).
    
    First train_ratio of data = training (assumed healthy)
    Remaining data = testing (may contain faults)
    
    This mimics real deployment: calibrate on initial healthy operation,
    then monitor for degradation over time.
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
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1S'),
        'temperature': 22 + np.random.randn(n) * 0.5,
        'humidity': 55 + np.random.randn(n) * 2
    })
    
    # Add some outliers
    df.loc[100, 'temperature'] = 100
    df.loc[200, 'humidity'] = 150
    
    # Split data
    train, test = split_train_test_temporal(df, train_ratio=0.7)
    
    # Preprocess
    preprocessor = SensorPreprocessor(window_size=20, stride=1)
    X_train, y_train = preprocessor.prepare_training_data(train)
    X_test, df_test_aligned = preprocessor.prepare_inference_data(test)
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
