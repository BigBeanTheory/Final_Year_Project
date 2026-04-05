"""
LSTM Autoencoder Model (Multi-Sensor)
Deep learning model for unsupervised anomaly detection across 6 sensor channels.

Architecture:
- Two-layer encoder: captures short-term and long-term multi-sensor patterns
- Bottleneck: forces compression of 6-channel temporal data
- Two-layer decoder: mirrors encoder for symmetric reconstruction
- TimeDistributed Dense: reconstructs all 6 features at each timestep

Anomaly detection:
- Reconstruction error = anomaly score (high error = abnormal pattern)
- Per-feature error breakdown enables fault localization
- Cross-modal learning: model learns normal correlations BETWEEN sensors
"""

import numpy as np
import os

# IMPORTANT: Use TensorFlow's bundled Keras to avoid version conflicts
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, models, optimizers

from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt


class LSTMAutoencoder:
    """
    LSTM-based autoencoder for multivariate time-series anomaly detection.
    Supports 6-feature multi-sensor input for predictive maintenance.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        n_features: int = 6,
        encoding_dim: int = 32,
        lstm_units: Tuple[int, int] = (128, 64),
        dropout_rate: float = 0.2,
        feature_names: List[str] = None
    ):
        """
        Args:
            window_size: Length of input sequences
            n_features: Number of sensor features (6 for multi-sensor)
            encoding_dim: Bottleneck dimension
            lstm_units: Tuple of (encoder_units, decoder_units)
            dropout_rate: Dropout probability for regularization
            feature_names: Optional list of feature names for diagnostics
        """
        self.window_size = window_size
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.feature_names = feature_names or [
            'temp_dht', 'humidity', 'temp_therm',
            'sound_level', 'light_level', 'flame_intensity'
        ]
        
        self.model = None
        self.history = None
        self.threshold = None
        
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Construct LSTM autoencoder architecture.
        
        Larger capacity than 2-feature version to handle:
        - 6 heterogeneous sensor channels
        - Cross-modal correlations (e.g., sound ↔ temperature)
        - Different temporal dynamics per sensor
        """
        # Input layer
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # ENCODER
        encoded = layers.LSTM(
            self.lstm_units[0],
            activation='tanh',
            return_sequences=True,
            name='encoder_lstm1'
        )(inputs)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        encoded = layers.LSTM(
            self.lstm_units[1],
            activation='tanh',
            return_sequences=False,
            name='encoder_lstm2'
        )(encoded)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # BOTTLENECK
        bottleneck = layers.Dense(
            self.encoding_dim,
            activation='relu',
            name='bottleneck'
        )(encoded)
        
        # DECODER
        decoded = layers.RepeatVector(self.window_size)(bottleneck)
        
        decoded = layers.LSTM(
            self.lstm_units[1],
            activation='tanh',
            return_sequences=True,
            name='decoder_lstm1'
        )(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        decoded = layers.LSTM(
            self.lstm_units[0],
            activation='tanh',
            return_sequences=True,
            name='decoder_lstm2'
        )(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # OUTPUT LAYER
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features),
            name='reconstruction'
        )(decoded)
        
        # Build model
        self.model = Model(inputs, outputs, name='LSTM_Autoencoder_MultiSensor')
        
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("\n=== Model Architecture ===")
        self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """Train autoencoder on healthy multi-sensor data."""
        print("\n=== Training Multi-Sensor LSTM Autoencoder ===")
        print(f"Input shape: {X_train.shape} ({self.n_features} features)")
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )
        
        print("\nTraining complete!")
        print(f"Final train loss: {self.history.history['loss'][-1]:.6f}")
        print(f"Final val loss: {self.history.history['val_loss'][-1]:.6f}")
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input sequences."""
        return self.model.predict(X, verbose=0)
    
    def compute_reconstruction_error(
        self,
        X: np.ndarray,
        per_sample: bool = True
    ) -> np.ndarray:
        """
        Compute reconstruction error (anomaly score).
        
        Args:
            X: Input sequences
            per_sample: If True, return one error per sample (MSE across time and features)
                       If False, return error per timestep per feature
        """
        X_reconstructed = self.predict(X)
        
        if per_sample:
            errors = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        else:
            errors = np.square(X - X_reconstructed)
        
        return errors
    
    def compute_per_feature_error(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute reconstruction error broken down by feature.
        Enables fault localization — which sensor is behaving abnormally?
        
        Returns:
            Dict mapping feature name to array of per-sample errors for that feature
        """
        X_reconstructed = self.predict(X)
        
        # Error per feature: mean across timesteps, keep feature dimension
        per_feature_errors = np.mean(np.square(X - X_reconstructed), axis=1)  # (samples, features)
        
        result = {}
        for i, name in enumerate(self.feature_names[:self.n_features]):
            result[name] = per_feature_errors[:, i]
        
        return result
    
    def set_threshold(
        self,
        X_train: np.ndarray,
        method: str = 'percentile',
        percentile: float = 95,
        std_multiplier: float = 2.0
    ) -> float:
        """Calculate anomaly detection threshold from training data."""
        train_errors = self.compute_reconstruction_error(X_train)
        
        if method == 'percentile':
            threshold = np.percentile(train_errors, percentile)
            print(f"\nThreshold set at {percentile}th percentile: {threshold:.6f}")
        elif method == 'std':
            mean_error = np.mean(train_errors)
            std_error = np.std(train_errors)
            threshold = mean_error + std_multiplier * std_error
            print(f"\nThreshold set at mean + {std_multiplier}*std: {threshold:.6f}")
            print(f"  Mean error: {mean_error:.6f}")
            print(f"  Std error: {std_error:.6f}")
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        self.threshold = threshold
        
        false_positives = (train_errors > threshold).sum()
        fpr = false_positives / len(train_errors)
        print(f"  False positive rate on training: {fpr*100:.2f}%")
        
        return threshold
    
    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using reconstruction error threshold."""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")
        
        errors = self.compute_reconstruction_error(X)
        is_anomaly = errors > self.threshold
        
        return errors, is_anomaly
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        """Plot training and validation loss curves."""
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_per_feature_error(self, X: np.ndarray, figsize: Tuple[int, int] = (14, 6)) -> None:
        """Plot reconstruction error breakdown per sensor feature."""
        per_feature = self.compute_per_feature_error(X)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        positions = range(len(per_feature))
        names = list(per_feature.keys())
        errors = [per_feature[name].mean() for name in names]
        stds = [per_feature[name].std() for name in names]
        
        bars = ax.bar(positions, errors, yerr=stds, capsize=5,
                      color=['#ff6b6b', '#4ecdc4', '#ff9f43', '#a55eea', '#feca57', '#ff4757'],
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('Mean Reconstruction Error')
        ax.set_title('Per-Sensor Reconstruction Error')
        ax.grid(True, alpha=0.3, axis='y')
        
        if self.threshold:
            ax.axhline(y=self.threshold, color='red', linestyle='--',
                       label=f'Threshold ({self.threshold:.4f})')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str) -> None:
        """Save model weights and threshold."""
        self.model.save(filepath)
        np.save(filepath.replace('.h5', '_threshold.npy'), self.threshold)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model weights and threshold."""
        self.model = models.load_model(filepath, compile=False, safe_mode=False)
        self.model.compile(optimizer='adam', loss='mse')
        
        threshold_path = filepath.replace('.h5', '_threshold.npy')
        try:
            self.threshold = np.load(threshold_path)
            print(f"Model and threshold loaded from {filepath}")
        except:
            print(f"Model loaded from {filepath} (threshold not found)")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    window_size = 20
    n_features = 6
    
    # Healthy multi-sensor data
    X_train = np.random.randn(n_samples, window_size, n_features) * 0.1
    y_train = X_train.copy()
    
    model = LSTMAutoencoder(
        window_size=window_size,
        n_features=n_features,
        encoding_dim=32,
        lstm_units=(128, 64),
        dropout_rate=0.2
    )
    
    model.train(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    threshold = model.set_threshold(X_train, method='percentile', percentile=95)
    
    # Test with anomalous data
    X_test = np.random.randn(100, window_size, n_features) * 0.5
    errors, is_anomaly = model.detect_anomalies(X_test)
    
    print(f"\nTest results:")
    print(f"  Anomalies detected: {is_anomaly.sum()} / {len(is_anomaly)}")
    print(f"  Mean error: {errors.mean():.6f}")
    
    # Per-feature breakdown
    per_feature = model.compute_per_feature_error(X_test)
    print(f"\nPer-feature errors:")
    for name, err in per_feature.items():
        print(f"  {name}: {err.mean():.6f}")