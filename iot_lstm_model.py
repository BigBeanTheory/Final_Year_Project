"""
LSTM Autoencoder Model
Deep learning model for unsupervised anomaly detection in sensor time-series.

Engineering Notes:
- Encoder compresses temporal patterns into low-dimensional representation
- Decoder reconstructs input from compressed representation
- Reconstruction error = anomaly score (high error = abnormal pattern)
- Dropout prevents overfitting on limited training data
- Returns sequences (not just final state) for time-aligned anomaly detection
"""

import numpy as np
import os

# IMPORTANT: Use TensorFlow's bundled Keras to avoid version conflicts
# Keras 3.9 (standalone) is incompatible with TensorFlow 2.17
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Force TF to use bundled Keras 2.x

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, models, optimizers

from typing import Tuple, Dict
import matplotlib.pyplot as plt

class LSTMAutoencoder:
    """
    LSTM-based autoencoder for multivariate time-series anomaly detection.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        n_features: int = 2,
        encoding_dim: int = 16,
        lstm_units: Tuple[int, int] = (64, 32),
        dropout_rate: float = 0.2
    ):
        """
        Args:
            window_size: Length of input sequences
            n_features: Number of sensor features (temp, humidity = 2)
            encoding_dim: Bottleneck dimension (compressed representation)
            lstm_units: Tuple of (encoder_units, decoder_units)
            dropout_rate: Dropout probability for regularization
        """
        self.window_size = window_size
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        self.threshold = None
        
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Construct LSTM autoencoder architecture.
        
        Architecture rationale:
        - Two-layer encoder: captures both short-term and long-term patterns
        - Bottleneck layer: forces compression (prevents identity mapping)
        - RepeatVector: broadcasts compressed state to decoder timesteps
        - Two-layer decoder: mirrors encoder for symmetric reconstruction
        - TimeDistributed Dense: produces output at each timestep
        """
        # Input layer
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # ENCODER
        # First LSTM layer (return sequences for stacking)
        encoded = layers.LSTM(
            self.lstm_units[0],
            activation='tanh',
            return_sequences=True,
            name='encoder_lstm1'
        )(inputs)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Second LSTM layer (return only final state)
        encoded = layers.LSTM(
            self.lstm_units[1],
            activation='tanh',
            return_sequences=False,
            name='encoder_lstm2'
        )(encoded)
        encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # BOTTLENECK (compressed representation)
        bottleneck = layers.Dense(
            self.encoding_dim,
            activation='relu',
            name='bottleneck'
        )(encoded)
        
        # DECODER
        # Repeat encoded state for all timesteps
        decoded = layers.RepeatVector(self.window_size)(bottleneck)
        
        # First LSTM layer
        decoded = layers.LSTM(
            self.lstm_units[1],
            activation='tanh',
            return_sequences=True,
            name='decoder_lstm1'
        )(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Second LSTM layer
        decoded = layers.LSTM(
            self.lstm_units[0],
            activation='tanh',
            return_sequences=True,
            name='decoder_lstm2'
        )(decoded)
        decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # OUTPUT LAYER (reconstruct original features)
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features),
            name='reconstruction'
        )(decoded)
        
        # Build model
        self.model = Model(inputs, outputs, name='LSTM_Autoencoder')
        
        # Compile with MSE loss (reconstruction error)
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
        """
        Train autoencoder on healthy sensor data.
        
        Args:
            X_train, y_train: Training sequences (y_train = X_train for autoencoder)
            validation_split: Fraction of training data for validation
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
        
        Returns:
            Training history dictionary
        """
        print("\n=== Training LSTM Autoencoder ===")
        
        # Callbacks
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
        
        # Train
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
        """
        Reconstruct input sequences.
        
        Returns:
            Reconstructed sequences of same shape as input
        """
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
        
        Returns:
            Array of reconstruction errors
        """
        X_reconstructed = self.predict(X)
        
        if per_sample:
            # Mean squared error across all timesteps and features
            errors = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        else:
            # Error per timestep per feature
            errors = np.square(X - X_reconstructed)
        
        return errors
    
    def set_threshold(
        self,
        X_train: np.ndarray,
        method: str = 'percentile',
        percentile: float = 95,
        std_multiplier: float = 2.0
    ) -> float:
        """
        Calculate anomaly detection threshold from training data.
        
        Methods:
        - 'percentile': Threshold at Nth percentile of training errors
        - 'std': Threshold = mean + k * std of training errors
        
        Rationale:
        - Use percentile for robustness to outliers
        - Use std for theoretical grounding (assumes Gaussian errors)
        
        Args:
            X_train: Training sequences
            method: 'percentile' or 'std'
            percentile: Percentile value (e.g., 95 = 95th percentile)
            std_multiplier: Standard deviation multiplier (e.g., 2.0 = 2-sigma)
        
        Returns:
            Threshold value
        """
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
        
        # Compute false positive rate on training data
        false_positives = (train_errors > threshold).sum()
        fpr = false_positives / len(train_errors)
        print(f"  False positive rate on training: {fpr*100:.2f}%")
        
        return threshold
    
    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using reconstruction error threshold.
        
        Args:
            X: Input sequences
        
        Returns:
            errors: Reconstruction errors per sample
            is_anomaly: Boolean array indicating anomalies
        """
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
        
        # Loss curve
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE curve
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath: str) -> None:
        """Save model weights and threshold."""
        self.model.save(filepath)
        
        # Save threshold separately
        np.save(filepath.replace('.h5', '_threshold.npy'), self.threshold)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model weights and threshold."""
        self.model = models.load_model(filepath, compile=False, safe_mode=False)
        self.model.compile(optimizer='adam', loss='mse')
        
        # Load threshold
        threshold_path = filepath.replace('.h5', '_threshold.npy')
        try:
            self.threshold = np.load(threshold_path)
            print(f"Model and threshold loaded from {filepath}")
        except:
            print(f"Model loaded from {filepath} (threshold not found)")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    window_size = 20
    n_features = 2
    
    # Healthy data (training)
    X_train = np.random.randn(n_samples, window_size, n_features) * 0.1
    y_train = X_train.copy()
    
    # Build and train model
    model = LSTMAutoencoder(
        window_size=window_size,
        n_features=n_features,
        encoding_dim=16,
        lstm_units=(64, 32),
        dropout_rate=0.2
    )
    
    model.train(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Set threshold
    threshold = model.set_threshold(X_train, method='percentile', percentile=95)
    
    # Test on anomalous data
    X_test = np.random.randn(100, window_size, n_features) * 0.5  # Higher variance
    errors, is_anomaly = model.detect_anomalies(X_test)
    
    print(f"\nTest results:")
    print(f"  Anomalies detected: {is_anomaly.sum()} / {len(is_anomaly)}")
    print(f"  Mean error: {errors.mean():.6f}")