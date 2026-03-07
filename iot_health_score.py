"""
Health Monitoring & Predictive Maintenance Logic
Converts anomaly scores into actionable health metrics and detects specific fault patterns.

Engineering Notes:
- Health score (0-100): actionable metric for maintenance scheduling
- Exponential moving average: weights recent behavior more heavily
- Fault pattern detection: identifies specific degradation modes
- Alert system: triggers based on health trends, not single anomalies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum

class FaultType(Enum):
    """Enumeration of detectable sensor faults."""
    HEALTHY = "Healthy"
    DRIFT = "Sensor Drift"
    NOISE = "Excessive Noise"
    FREEZE = "Stuck/Frozen"
    COMMS_FAILURE = "Communication Failure"
    GENERAL_ANOMALY = "General Anomaly"

class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = "Normal"
    WARNING = "Warning"
    CRITICAL = "Critical"

class SensorHealthMonitor:
    """
    Converts anomaly scores into health metrics and detects fault patterns.
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.05,  # Smoothing factor (lower = more smoothing)
        drift_window: int = 50,    # Window for drift detection
        noise_window: int = 20,    # Window for noise detection
        freeze_threshold: int = 5  # Consecutive identical readings = freeze
    ):
        """
        Args:
            ema_alpha: EMA smoothing factor (0-1). Lower = more history weight
            drift_window: Rolling window size for drift detection
            noise_window: Rolling window size for noise detection
            freeze_threshold: Min consecutive identical values to flag freeze
        """
        self.ema_alpha = ema_alpha
        self.drift_window = drift_window
        self.noise_window = noise_window
        self.freeze_threshold = freeze_threshold
        
        # State tracking
        self.health_scores = []
        self.fault_history = []
        self.alert_history = []
        
    def anomaly_score_to_health(
        self,
        anomaly_scores: np.ndarray,
        threshold: float,
        smooth: bool = True
    ) -> np.ndarray:
        """
        Convert reconstruction errors to health scores (0-100).
        
        Mapping logic:
        - Error <= threshold → Health = 100
        - Error = 2*threshold → Health = 50
        - Error >= 4*threshold → Health = 0
        
        Uses exponential decay to map errors to health scores.
        
        Args:
            anomaly_scores: Array of reconstruction errors
            threshold: Anomaly detection threshold
            smooth: Apply exponential moving average
        
        Returns:
            Array of health scores (0-100)
        """
        # Normalize errors relative to threshold
        normalized_errors = anomaly_scores / threshold
        
        # Map to health score using exponential decay
        # Health = 100 * exp(-k * normalized_error)
        # Choose k such that error=2*threshold gives health=50
        # 50 = 100 * exp(-k * 2) → k = -ln(0.5)/2 ≈ 0.347
        k = 0.347
        health = 100 * np.exp(-k * normalized_errors)
        health = np.clip(health, 0, 100)
        
        if smooth:
            # Apply exponential moving average
            health_smooth = np.zeros_like(health)
            health_smooth[0] = health[0]
            
            for i in range(1, len(health)):
                health_smooth[i] = (self.ema_alpha * health[i] + 
                                   (1 - self.ema_alpha) * health_smooth[i-1])
            
            return health_smooth
        
        return health
    
    def detect_drift(
        self,
        df: pd.DataFrame,
        feature: str = 'temperature'
    ) -> Tuple[bool, float]:
        """
        Detect gradual drift in sensor readings.
        
        Method: Linear regression on rolling window. Significant positive/negative
        slope indicates drift.
        
        Returns:
            (is_drifting, drift_rate)
        """
        if len(df) < self.drift_window:
            return False, 0.0
        
        # Use most recent window
        window = df[feature].iloc[-self.drift_window:].values
        x = np.arange(len(window))
        
        # Linear regression
        slope, _ = np.polyfit(x, window, 1)
        
        # Normalize slope by value range to get rate
        value_range = window.max() - window.min()
        drift_rate = slope / (value_range + 1e-6)  # Avoid division by zero
        
        # Threshold: drift if rate > 1% per reading
        is_drifting = abs(drift_rate) > 0.01
        
        return is_drifting, drift_rate
    
    def detect_noise(
        self,
        df: pd.DataFrame,
        feature: str = 'temperature',
        baseline_std: float = None
    ) -> Tuple[bool, float]:
        """
        Detect increased noise in sensor readings.
        
        Method: Compare rolling standard deviation to baseline.
        
        Returns:
            (is_noisy, current_noise_level)
        """
        if len(df) < self.noise_window:
            return False, 0.0
        
        # Use most recent window
        window = df[feature].iloc[-self.noise_window:].values
        current_std = np.std(window)
        
        # If no baseline provided, use first window as baseline
        if baseline_std is None:
            if len(df) >= 2 * self.noise_window:
                baseline_window = df[feature].iloc[:self.noise_window].values
                baseline_std = np.std(baseline_window)
            else:
                return False, current_std
        
        # Threshold: noise increased by >50%
        noise_ratio = current_std / (baseline_std + 1e-6)
        is_noisy = noise_ratio > 1.5
        
        return is_noisy, current_std
    
    def detect_freeze(
        self,
        df: pd.DataFrame,
        feature: str = 'temperature'
    ) -> Tuple[bool, int]:
        """
        Detect stuck/frozen sensor (consecutive identical readings).
        
        Returns:
            (is_frozen, consecutive_count)
        """
        if len(df) < self.freeze_threshold:
            return False, 0
        
        # Check most recent readings
        recent = df[feature].iloc[-self.freeze_threshold:].values
        
        # Count consecutive identical values
        if len(np.unique(recent)) == 1:
            # All values identical - check how many
            full_data = df[feature].values
            consecutive = 1
            for i in range(len(full_data) - 2, -1, -1):
                if full_data[i] == full_data[-1]:
                    consecutive += 1
                else:
                    break
            
            return True, consecutive
        
        return False, 0
    
    def detect_communication_failure(
        self,
        df: pd.DataFrame,
        max_gap_seconds: int = 5
    ) -> Tuple[bool, int]:
        """
        Detect communication failures (large time gaps).
        
        Returns:
            (has_failure, num_gaps)
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            return False, 0
        
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > max_gap_seconds]
        
        has_failure = len(gaps) > 0
        
        return has_failure, len(gaps)
    
    def classify_fault(
        self,
        df: pd.DataFrame,
        anomaly_score: float,
        threshold: float,
        feature: str = 'temperature'
    ) -> FaultType:
        """
        Classify the type of fault based on multiple indicators.
        
        Priority order:
        1. Communication failure (missing data)
        2. Freeze (stuck sensor)
        3. Drift (gradual deviation)
        4. Noise (increased variance)
        5. General anomaly (high reconstruction error)
        6. Healthy
        """
        # Check communication
        comms_fail, _ = self.detect_communication_failure(df)
        if comms_fail:
            return FaultType.COMMS_FAILURE
        
        # Check freeze
        is_frozen, _ = self.detect_freeze(df, feature)
        if is_frozen:
            return FaultType.FREEZE
        
        # Only check drift/noise if we have enough data
        if len(df) >= self.drift_window:
            # Check drift
            is_drifting, _ = self.detect_drift(df, feature)
            if is_drifting:
                return FaultType.DRIFT
        
        if len(df) >= self.noise_window:
            # Check noise
            is_noisy, _ = self.detect_noise(df, feature)
            if is_noisy:
                return FaultType.NOISE
        
        # Check general anomaly
        if anomaly_score > threshold:
            return FaultType.GENERAL_ANOMALY
        
        return FaultType.HEALTHY
    
    def determine_alert_level(
        self,
        health_score: float,
        fault_type: FaultType
    ) -> AlertLevel:
        """
        Determine alert severity based on health score and fault type.
        
        Thresholds:
        - Health > 80: Normal
        - Health 50-80: Warning
        - Health < 50: Critical
        
        Exception: Communication failure always critical
        """
        if fault_type == FaultType.COMMS_FAILURE:
            return AlertLevel.CRITICAL
        
        if health_score >= 80:
            return AlertLevel.NORMAL
        elif health_score >= 50:
            return AlertLevel.WARNING
        else:
            return AlertLevel.CRITICAL
    
    def process_batch(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        threshold: float
    ) -> pd.DataFrame:
        """
        Process batch of sensor data and generate health report.
        
        Args:
            df: DataFrame with sensor readings and timestamps
            anomaly_scores: Array of reconstruction errors (one per row)
            threshold: Anomaly detection threshold
        
        Returns:
            DataFrame with added columns: health_score, fault_type, alert_level
        """
        df = df.copy()
        
        # Calculate health scores
        health_scores = self.anomaly_score_to_health(anomaly_scores, threshold, smooth=True)
        df['health_score'] = health_scores
        df['anomaly_score'] = anomaly_scores
        
        # Classify faults for each timestep
        fault_types = []
        alert_levels = []
        
        for i in range(len(df)):
            # Use data up to current timestep for pattern detection
            df_window = df.iloc[:i+1]
            
            fault = self.classify_fault(
                df_window,
                anomaly_scores[i],
                threshold,
                feature='temperature'
            )
            
            alert = self.determine_alert_level(
                health_scores[i],
                fault
            )
            
            fault_types.append(fault.value)
            alert_levels.append(alert.value)
        
        df['fault_type'] = fault_types
        df['alert_level'] = alert_levels
        
        # Store in history
        self.health_scores.extend(health_scores.tolist())
        self.fault_history.extend(fault_types)
        self.alert_history.extend(alert_levels)
        
        return df
    
    def get_maintenance_recommendation(
        self,
        current_health: float,
        fault_type: FaultType,
        alert_level: AlertLevel
    ) -> str:
        """
        Generate human-readable maintenance recommendation.
        """
        if alert_level == AlertLevel.CRITICAL:
            if fault_type == FaultType.COMMS_FAILURE:
                return "CRITICAL: Check sensor wiring and power supply immediately"
            elif fault_type == FaultType.FREEZE:
                return "CRITICAL: Sensor frozen - replace sensor unit"
            else:
                return f"CRITICAL: Sensor health {current_health:.0f}% - immediate replacement recommended"
        
        elif alert_level == AlertLevel.WARNING:
            if fault_type == FaultType.DRIFT:
                return f"WARNING: Sensor drifting - schedule recalibration (health: {current_health:.0f}%)"
            elif fault_type == FaultType.NOISE:
                return f"WARNING: Increased noise - check for electromagnetic interference (health: {current_health:.0f}%)"
            else:
                return f"WARNING: Sensor degradation detected - monitor closely (health: {current_health:.0f}%)"
        
        else:
            return f"NORMAL: Sensor operating within spec (health: {current_health:.0f}%)"
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary report of sensor health status.
        """
        if len(df) == 0:
            return {"error": "No data to analyze"}
        
        current_health = df['health_score'].iloc[-1]
        current_fault = df['fault_type'].iloc[-1]
        current_alert = df['alert_level'].iloc[-1]
        
        # Count fault occurrences
        fault_counts = df['fault_type'].value_counts().to_dict()
        
        # Calculate uptime
        total_readings = len(df)
        healthy_readings = (df['fault_type'] == FaultType.HEALTHY.value).sum()
        uptime_percent = (healthy_readings / total_readings) * 100
        
        report = {
            'current_health': current_health,
            'current_fault': current_fault,
            'current_alert': current_alert,
            'uptime_percent': uptime_percent,
            'fault_counts': fault_counts,
            'total_readings': total_readings,
            'recommendation': self.get_maintenance_recommendation(
                current_health,
                FaultType(current_fault),
                AlertLevel(current_alert)
            )
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Create sample data with drift
    np.random.seed(42)
    n = 200
    
    # Healthy period
    temp_healthy = 22 + np.random.randn(100) * 0.5
    
    # Drifting period
    drift = np.linspace(0, 5, 100)
    temp_drift = 22 + drift + np.random.randn(100) * 0.5
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1S'),
        'temperature': np.concatenate([temp_healthy, temp_drift]),
        'humidity': 55 + np.random.randn(n) * 2
    })
    
    # Simulate anomaly scores
    anomaly_scores = np.concatenate([
        np.random.rand(100) * 0.01,  # Healthy
        np.random.rand(100) * 0.05   # Drifting
    ])
    threshold = 0.02
    
    # Process
    monitor = SensorHealthMonitor()
    df_processed = monitor.process_batch(df, anomaly_scores, threshold)
    
    # Generate report
    report = monitor.generate_report(df_processed)
    
    print("\n=== Sensor Health Report ===")
    print(f"Current Health: {report['current_health']:.1f}%")
    print(f"Current Fault: {report['current_fault']}")
    print(f"Uptime: {report['uptime_percent']:.1f}%")
    print(f"\nRecommendation: {report['recommendation']}")
