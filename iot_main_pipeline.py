"""
Main Training Pipeline
End-to-end training and evaluation of IoT sensor predictive maintenance system.

Usage:
    python main.py --data_path sensor_data.csv --output_dir models/

Engineering Notes:
- Trains only on first 70% of data (assumed healthy)
- Tests on remaining 30% (may contain faults)
- Saves trained model, preprocessor, and evaluation metrics
- Generates comprehensive evaluation report
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from iot_sensor_ingest import SensorDataIngestion
from iot_preprocessing import SensorPreprocessor, split_train_test_temporal
from iot_lstm_model import LSTMAutoencoder
from iot_health_score import SensorHealthMonitor
import config  # Import centralized configuration

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train IoT sensor predictive maintenance system"
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to sensor data file (CSV or LOG)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        default=20,
        help='LSTM window size (timesteps)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Ratio of data for training (rest for testing)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Training batch size'
    )
    
    return parser.parse_args()

def create_threshold_comparison_plot(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    threshold: float,
    output_path: str
):
    """
    Visualize reconstruction error distributions for train vs test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram comparison
    axes[0].hist(train_errors, bins=50, alpha=0.6, label='Train', density=True)
    axes[0].hist(test_errors, bins=50, alpha=0.6, label='Test', density=True)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('Reconstruction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution: Train vs Test')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    train_sorted = np.sort(train_errors)
    test_sorted = np.sort(test_errors)
    
    axes[1].plot(train_sorted, np.linspace(0, 1, len(train_sorted)), label='Train', linewidth=2)
    axes[1].plot(test_sorted, np.linspace(0, 1, len(test_sorted)), label='Test', linewidth=2)
    axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_xlabel('Reconstruction Error')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('Cumulative Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved threshold comparison plot to {output_path}")
    plt.close()

def evaluate_detection_performance(
    df_test: pd.DataFrame,
    threshold: float,
    output_path: str
):
    """
    Evaluate anomaly detection performance.
    
    Metrics:
    - Detection rate (% of anomalies detected)
    - False positive rate
    - Mean detection delay
    """
    total_samples = len(df_test)
    anomalies_detected = (df_test['anomaly_score'] > threshold).sum()
    
    # Calculate false positive rate (assuming first 20% is healthy)
    assumed_healthy_count = int(total_samples * 0.2)
    if assumed_healthy_count > 0:
        false_positives = (df_test['anomaly_score'].iloc[:assumed_healthy_count] > threshold).sum()
        fpr = false_positives / assumed_healthy_count
    else:
        fpr = 0.0
    
    # Calculate detection delay (time to first anomaly)
    first_anomaly_idx = df_test[df_test['anomaly_score'] > threshold].index.min()
    if pd.notna(first_anomaly_idx):
        detection_delay = first_anomaly_idx
    else:
        detection_delay = None
    
    report = {
        'total_samples': total_samples,
        'anomalies_detected': anomalies_detected,
        'detection_rate': (anomalies_detected / total_samples) * 100,
        'false_positive_rate': fpr * 100,
        'detection_delay_samples': detection_delay
    }
    
    # Print report
    print("\n" + "="*60)
    print("ANOMALY DETECTION PERFORMANCE")
    print("="*60)
    print(f"Total test samples: {report['total_samples']}")
    print(f"Anomalies detected: {report['anomalies_detected']} ({report['detection_rate']:.2f}%)")
    print(f"False positive rate: {report['false_positive_rate']:.2f}%")
    if detection_delay:
        print(f"Detection delay: {detection_delay} samples")
    print("="*60)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("ANOMALY DETECTION PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
    
    return report

def compare_with_simple_threshold(
    df_test: pd.DataFrame,
    threshold: float,
    output_path: str
):
    """
    Compare LSTM detection with simple statistical thresholding.
    
    Simple method: Flag if value exceeds mean ± 3*std from training data
    """
    # Use first 20% as baseline
    baseline_count = int(len(df_test) * 0.2)
    baseline = df_test.iloc[:baseline_count]
    
    temp_mean = baseline['temperature'].mean()
    temp_std = baseline['temperature'].std()
    hum_mean = baseline['humidity'].mean()
    hum_std = baseline['humidity'].std()
    
    # Simple threshold method
    temp_anomalies = (
        (df_test['temperature'] < temp_mean - 3*temp_std) |
        (df_test['temperature'] > temp_mean + 3*temp_std)
    )
    hum_anomalies = (
        (df_test['humidity'] < hum_mean - 3*hum_std) |
        (df_test['humidity'] > hum_mean + 3*hum_std)
    )
    simple_anomalies = temp_anomalies | hum_anomalies
    
    # LSTM method
    lstm_anomalies = df_test['anomaly_score'] > threshold
    
    # Comparison
    lstm_count = lstm_anomalies.sum()
    simple_count = simple_anomalies.sum()
    both_count = (lstm_anomalies & simple_anomalies).sum()
    lstm_only = (lstm_anomalies & ~simple_anomalies).sum()
    simple_only = (~lstm_anomalies & simple_anomalies).sum()
    
    print("\n" + "="*60)
    print("COMPARISON: LSTM vs Simple Threshold")
    print("="*60)
    print(f"LSTM detected: {lstm_count} anomalies")
    print(f"Simple threshold detected: {simple_count} anomalies")
    print(f"Both methods agreed: {both_count} anomalies")
    print(f"LSTM only: {lstm_only} anomalies (temporal patterns)")
    print(f"Simple only: {simple_only} anomalies (static outliers)")
    print("="*60)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    timestamps = df_test['timestamp'].values
    ax.plot(timestamps, df_test['temperature'], label='Temperature', alpha=0.7)
    
    # Mark anomalies
    lstm_times = df_test[lstm_anomalies]['timestamp'].values
    simple_times = df_test[simple_anomalies]['timestamp'].values
    
    ax.scatter(lstm_times, df_test[lstm_anomalies]['temperature'], 
              color='red', marker='x', s=100, label='LSTM detected', zorder=3)
    ax.scatter(simple_times, df_test[simple_anomalies]['temperature'],
              color='blue', marker='o', s=50, alpha=0.5, label='Simple detected', zorder=2)
    
    ax.axhline(temp_mean + 3*temp_std, color='gray', linestyle='--', alpha=0.5, label='±3σ bounds')
    ax.axhline(temp_mean - 3*temp_std, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Anomaly Detection Comparison: LSTM vs Simple Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()

def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("IOT SENSOR PREDICTIVE MAINTENANCE - TRAINING PIPELINE")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Train ratio: {args.train_ratio}")
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and ingest data
    print("STEP 1: Data Ingestion")
    print("-" * 60)
    ingestion = SensorDataIngestion()
    full_data, clean_data = ingestion.load_and_process(args.data_path)
    
    # Step 2: Split data temporally
    print("\nSTEP 2: Temporal Data Split")
    print("-" * 60)
    train_data, test_data = split_train_test_temporal(clean_data, args.train_ratio)
    
    # Step 3: Preprocessing
    print("\nSTEP 3: Preprocessing")
    print("-" * 60)
    preprocessor = SensorPreprocessor(window_size=args.window_size, stride=1)
    
    X_train, y_train = preprocessor.prepare_training_data(train_data)
    X_test, df_test_aligned = preprocessor.prepare_inference_data(test_data)
    
    # Save preprocessor
    preprocessor.save(output_dir / 'preprocessor.pkl')
    
    # Step 4: Train LSTM Autoencoder
    print("\nSTEP 4: Training LSTM Autoencoder")
    print("-" * 60)
    model = LSTMAutoencoder(
        window_size=args.window_size,
        n_features=2,
        encoding_dim=16,
        lstm_units=(64, 32),
        dropout_rate=0.2
    )
    
    history = model.train(
        X_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Save model
    model.save(output_dir / 'lstm_autoencoder.h5')
    
    # Plot training history
    model.plot_training_history()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Saved training history to {output_dir / 'training_history.png'}")
    
    # Step 5: Set anomaly threshold
    print("\nSTEP 5: Setting Anomaly Threshold")
    print("-" * 60)
    threshold = model.set_threshold(X_train, method='percentile', percentile=95)
    
    # Step 6: Evaluate on test data
    print("\nSTEP 6: Evaluation on Test Data")
    print("-" * 60)
    
    # Compute errors
    train_errors = model.compute_reconstruction_error(X_train)
    test_errors = model.compute_reconstruction_error(X_test)
    
    print(f"Training error: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
    print(f"Test error: mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")
    print(f"Error increase: {(test_errors.mean() / train_errors.mean() - 1) * 100:.2f}%")
    
    # Visualize threshold
    create_threshold_comparison_plot(
        train_errors,
        test_errors,
        threshold,
        output_dir / 'threshold_comparison.png'
    )
    
    # Step 7: Health monitoring
    print("\nSTEP 7: Health Monitoring")
    print("-" * 60)
    monitor = SensorHealthMonitor()
    df_test_processed = monitor.process_batch(df_test_aligned, test_errors, threshold)
    
    # Save processed results
    df_test_processed.to_csv(output_dir / 'test_results.csv', index=False)
    print(f"Saved test results to {output_dir / 'test_results.csv'}")
    
    # Generate report
    report = monitor.generate_report(df_test_processed)
    print("\n" + "="*60)
    print("SENSOR HEALTH REPORT")
    print("="*60)
    for key, value in report.items():
        if key != 'fault_counts':
            print(f"{key}: {value}")
    print("\nFault distribution:")
    for fault, count in report['fault_counts'].items():
        print(f"  {fault}: {count}")
    print("="*60)
    
    # Step 8: Performance evaluation
    print("\nSTEP 8: Performance Evaluation")
    print("-" * 60)
    
    # Detection performance
    eval_report = evaluate_detection_performance(
        df_test_processed,
        threshold,
        output_dir / 'detection_report.txt'
    )
    
    # Comparison with simple method
    compare_with_simple_threshold(
        df_test_processed,
        threshold,
        output_dir / 'method_comparison.png'
    )
    
    # Step 9: Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {output_dir / 'lstm_autoencoder.h5'}")
    print(f"Preprocessor saved to: {output_dir / 'preprocessor.pkl'}")
    print(f"Test results saved to: {output_dir / 'test_results.csv'}")
    print(f"Evaluation plots saved to: {output_dir}")
    print("\nTo run dashboard:")
    print(f"  streamlit run dashboard.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
