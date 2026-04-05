"""
Health Monitoring & Predictive Maintenance Logic (Multi-Sensor)
Converts anomaly scores into actionable health metrics with sensor fusion.

Supports 6-feature multi-sensor array with fault types:
  - Environmental: drift, noise, freeze
  - Cross-validation: sensor divergence (DHT11 vs thermistor)
  - Safety: fire hazard (flame sensor)
  - Acoustic: sound anomaly
  - Optical: light anomaly
  - Communication: data gaps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum

class FaultType(Enum):
    """Enumeration of detectable sensor/equipment faults."""
    HEALTHY           = "Healthy"
    DRIFT             = "Sensor Drift"
    NOISE             = "Excessive Noise"
    FREEZE            = "Stuck/Frozen"
    COMMS_FAILURE     = "Communication Failure"
    GENERAL_ANOMALY   = "General Anomaly"
    SENSOR_DIVERGENCE = "Sensor Divergence"
    FIRE_HAZARD       = "Fire Hazard"
    ACOUSTIC_ANOMALY  = "Acoustic Anomaly"
    LIGHT_ANOMALY     = "Light Anomaly"

class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL   = "Normal"
    WARNING  = "Warning"
    CRITICAL = "Critical"

class SensorHealthMonitor:
    """
    Multi-sensor health monitor with sensor fusion.
    Converts reconstruction errors into health metrics and detects
    specific fault patterns across all sensor modalities.
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.05,
        drift_window: int = 50,
        noise_window: int = 20,
        freeze_threshold: int = 5
    ):
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
        smooth: bool = True,
        initial_health: float = None
    ) -> np.ndarray:
        """
        Convert reconstruction errors to health scores (0-100).
        
        Pass `initial_health` (the last health score from the previous call) to
        continue the EMA from where it left off.  Without it the smoother restarts
        from health[0] every tick, causing visible jumps on every buffer trim.
        """
        normalized_errors = anomaly_scores / (threshold + 1e-10)
        k = 0.347
        health = 100 * np.exp(-k * normalized_errors)
        health = np.clip(health, 0, 100)
        
        if smooth:
            health_smooth = np.zeros_like(health)
            # Continue from last known state instead of restarting cold.
            health_smooth[0] = initial_health if initial_health is not None else health[0]
            for i in range(1, len(health)):
                health_smooth[i] = (self.ema_alpha * health[i] + 
                                   (1 - self.ema_alpha) * health_smooth[i-1])
            return health_smooth
        
        return health
    
    # ------------------------------------------------------------------
    # PATTERN DETECTORS
    # ------------------------------------------------------------------
    
    def detect_drift(
        self,
        df: pd.DataFrame,
        feature: str = 'temp_dht'
    ) -> Tuple[bool, float]:
        """Detect gradual drift in sensor readings using linear regression."""
        if feature not in df.columns or len(df) < self.drift_window:
            return False, 0.0
        
        window = df[feature].iloc[-self.drift_window:].values
        x = np.arange(len(window))
        slope, _ = np.polyfit(x, window, 1)
        value_range = window.max() - window.min()
        drift_rate = slope / (value_range + 1e-6)
        is_drifting = abs(drift_rate) > 0.01
        
        return is_drifting, drift_rate
    
    def detect_noise(
        self,
        df: pd.DataFrame,
        feature: str = 'temp_dht',
        baseline_std: float = None
    ) -> Tuple[bool, float]:
        """Detect increased noise by comparing rolling std to baseline."""
        if feature not in df.columns or len(df) < self.noise_window:
            return False, 0.0
        
        window = df[feature].iloc[-self.noise_window:].values
        current_std = np.std(window)
        
        if baseline_std is None:
            if len(df) >= 2 * self.noise_window:
                baseline_window = df[feature].iloc[:self.noise_window].values
                baseline_std = np.std(baseline_window)
            else:
                return False, current_std
        
        noise_ratio = current_std / (baseline_std + 1e-6)
        is_noisy = noise_ratio > 1.5
        
        return is_noisy, current_std
    
    def detect_freeze(
        self,
        df: pd.DataFrame,
        feature: str = 'temp_dht'
    ) -> Tuple[bool, int]:
        """Detect stuck/frozen sensor (consecutive identical readings)."""
        if feature not in df.columns or len(df) < self.freeze_threshold:
            return False, 0
        
        recent = df[feature].iloc[-self.freeze_threshold:].values
        
        if len(np.unique(recent)) == 1:
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
        """Detect communication failures (large time gaps)."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return False, 0
        
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        gaps = time_diffs[time_diffs > max_gap_seconds]
        return len(gaps) > 0, len(gaps)
    
    # ------------------------------------------------------------------
    # MULTI-SENSOR SPECIFIC DETECTORS
    # ------------------------------------------------------------------
    
    def detect_sensor_divergence(
        self,
        df: pd.DataFrame,
        max_diff: float = 5.0
    ) -> Tuple[bool, float]:
        """
        Detect divergence between DHT11 and thermistor temperatures.
        If both sensors measure the same environment but disagree,
        one sensor is degrading — this is sensor health monitoring.
        """
        if 'temp_dht' not in df.columns or 'temp_therm' not in df.columns:
            return False, 0.0
        if len(df) < 5:
            return False, 0.0
        
        # Use recent window for robustness
        recent = df.tail(min(10, len(df)))
        temp_diff = abs(recent['temp_dht'] - recent['temp_therm']).mean()
        is_diverging = temp_diff > max_diff
        
        return is_diverging, temp_diff
    
    def detect_fire_hazard(
        self,
        df: pd.DataFrame,
        flame_threshold: float = 500.0
    ) -> Tuple[bool, float]:
        """
        Detect fire hazard from flame sensor.
        Higher flame_intensity = more IR detected = potential fire.
        """
        if 'flame_intensity' not in df.columns or len(df) < 3:
            return False, 0.0
        
        # Check if recent readings show sustained high flame
        recent = df['flame_intensity'].tail(min(5, len(df)))
        avg_flame = recent.mean()
        is_fire = avg_flame > flame_threshold
        
        return is_fire, avg_flame
    
    def detect_acoustic_anomaly(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, float]:
        """
        Detect unusual sound patterns.
        Sudden sound level changes indicate equipment faults
        (bearing wear, loose parts, unusual operation).
        """
        if 'sound_level' not in df.columns or len(df) < self.noise_window:
            return False, 0.0
        
        window = df['sound_level'].iloc[-self.noise_window:].values
        current_std = np.std(window)
        current_mean = np.mean(window)
        
        # Compare to baseline
        if len(df) >= 2 * self.noise_window:
            baseline = df['sound_level'].iloc[:self.noise_window].values
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)
            
            # Anomaly if mean shifted significantly or variance exploded
            mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
            variance_ratio = current_std / (baseline_std + 1e-6)
            
            is_anomalous = mean_shift > 3.0 or variance_ratio > 2.5
            return is_anomalous, mean_shift
        
        return False, 0.0
    
    def detect_light_anomaly(
        self,
        df: pd.DataFrame
    ) -> Tuple[bool, float]:
        """
        Detect unexpected light pattern changes.
        Sudden changes in light level may indicate equipment
        state changes, sparking, or environmental issues.
        """
        if 'light_level' not in df.columns or len(df) < self.noise_window:
            return False, 0.0
        
        window = df['light_level'].iloc[-self.noise_window:].values
        current_mean = np.mean(window)
        
        if len(df) >= 2 * self.noise_window:
            baseline = df['light_level'].iloc[:self.noise_window].values
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)
            
            change_ratio = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
            is_anomalous = change_ratio > 3.0
            return is_anomalous, change_ratio
        
        return False, 0.0
    
    # ------------------------------------------------------------------
    # FAULT CLASSIFICATION (Multi-Sensor)
    # ------------------------------------------------------------------
    
    def classify_fault(
        self,
        df: pd.DataFrame,
        anomaly_score: float,
        threshold: float,
        feature: str = 'temp_dht'
    ) -> FaultType:
        """
        Classify fault type using all available sensor modalities.
        
        Priority order (safety first):
        1. Fire hazard (flame sensor — immediate danger)
        2. Communication failure (missing data)
        3. Sensor divergence (DHT11 vs thermistor mismatch)
        4. Freeze (stuck sensor)
        5. Acoustic anomaly (unusual sound pattern)
        6. Light anomaly (unusual light pattern)
        7. Drift (gradual deviation)
        8. Noise (increased variance)
        9. General anomaly (high reconstruction error)
        10. Healthy
        """
        # 1. Fire hazard — highest priority (safety)
        is_fire, _ = self.detect_fire_hazard(df)
        if is_fire:
            return FaultType.FIRE_HAZARD
        
        # 2. Communication failure
        comms_fail, _ = self.detect_communication_failure(df)
        if comms_fail:
            return FaultType.COMMS_FAILURE
        
        # 3. Sensor divergence (cross-validation)
        is_diverging, _ = self.detect_sensor_divergence(df)
        if is_diverging:
            return FaultType.SENSOR_DIVERGENCE
        
        # 4. Freeze
        is_frozen, _ = self.detect_freeze(df, feature)
        if is_frozen:
            return FaultType.FREEZE
        
        # 5. Acoustic anomaly
        is_acoustic, _ = self.detect_acoustic_anomaly(df)
        if is_acoustic:
            return FaultType.ACOUSTIC_ANOMALY
        
        # 6. Light anomaly
        is_light, _ = self.detect_light_anomaly(df)
        if is_light:
            return FaultType.LIGHT_ANOMALY
        
        # 7-8. Drift and noise (need enough data)
        if len(df) >= self.drift_window:
            is_drifting, _ = self.detect_drift(df, feature)
            if is_drifting:
                return FaultType.DRIFT
        
        if len(df) >= self.noise_window:
            is_noisy, _ = self.detect_noise(df, feature)
            if is_noisy:
                return FaultType.NOISE
        
        # 9. General anomaly
        if anomaly_score > threshold:
            return FaultType.GENERAL_ANOMALY
        
        return FaultType.HEALTHY
    
    # ------------------------------------------------------------------
    # SENSOR FUSION HEALTH SCORE
    # ------------------------------------------------------------------
    
    def compute_per_sensor_health(
        self,
        per_feature_errors: Dict[str, float],
        threshold: float,
        sensor_weights: Dict[str, float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted composite health score from per-sensor errors.
        
        Returns:
            composite_health: weighted average health score (0-100)
            per_sensor_health: dict of individual sensor health scores
        """
        if sensor_weights is None:
            # Equal weights if not specified
            sensor_weights = {k: 1.0/len(per_feature_errors) for k in per_feature_errors}
        
        per_sensor_health = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature, error in per_feature_errors.items():
            k = 0.347
            normalized = error / (threshold + 1e-10)
            health = 100 * np.exp(-k * normalized)
            health = max(0, min(100, health))
            per_sensor_health[feature] = health
            
            weight = sensor_weights.get(feature, 1.0 / len(per_feature_errors))
            weighted_sum += health * weight
            total_weight += weight
        
        composite_health = weighted_sum / (total_weight + 1e-10)
        return composite_health, per_sensor_health
    
    # ------------------------------------------------------------------
    # ALERT LEVELS & RECOMMENDATIONS
    # ------------------------------------------------------------------
    
    def determine_alert_level(
        self,
        health_score: float,
        fault_type: FaultType
    ) -> AlertLevel:
        """
        Determine alert severity based on health and fault type.
        Fire hazard is always critical regardless of health score.
        """
        # Safety-critical faults are always critical
        if fault_type == FaultType.FIRE_HAZARD:
            return AlertLevel.CRITICAL
        if fault_type == FaultType.COMMS_FAILURE:
            return AlertLevel.CRITICAL
        
        if health_score >= 80:
            return AlertLevel.NORMAL
        elif health_score >= 50:
            return AlertLevel.WARNING
        else:
            return AlertLevel.CRITICAL
    
    def get_maintenance_recommendation(
        self,
        current_health: float,
        fault_type: FaultType,
        alert_level: AlertLevel
    ) -> str:
        """Generate human-readable maintenance recommendation."""
        
        recommendations = {
            FaultType.FIRE_HAZARD: 
                "🔥 CRITICAL: Fire/overheating detected — shut down equipment immediately!",
            FaultType.COMMS_FAILURE: 
                "📡 CRITICAL: Check sensor wiring and power supply immediately",
            FaultType.SENSOR_DIVERGENCE: 
                f"⚠️ WARNING: DHT11 and thermistor readings diverging — sensor calibration needed (health: {current_health:.0f}%)",
            FaultType.FREEZE: 
                "❄️ CRITICAL: Sensor frozen — replace sensor unit",
            FaultType.ACOUSTIC_ANOMALY: 
                f"🔊 WARNING: Unusual sound pattern detected — inspect equipment for mechanical issues (health: {current_health:.0f}%)",
            FaultType.LIGHT_ANOMALY: 
                f"💡 WARNING: Unexpected light level change — check equipment state (health: {current_health:.0f}%)",
            FaultType.DRIFT: 
                f"📈 WARNING: Sensor drifting — schedule recalibration (health: {current_health:.0f}%)",
            FaultType.NOISE: 
                f"📊 WARNING: Increased noise — check for electromagnetic interference (health: {current_health:.0f}%)",
            FaultType.GENERAL_ANOMALY: 
                f"⚠️ WARNING: Anomalous multi-sensor pattern detected — monitor closely (health: {current_health:.0f}%)",
        }
        
        if fault_type in recommendations:
            return recommendations[fault_type]
        
        if alert_level == AlertLevel.CRITICAL:
            return f"🚨 CRITICAL: Sensor health {current_health:.0f}% — immediate attention required"
        elif alert_level == AlertLevel.WARNING:
            return f"⚠️ WARNING: Sensor degradation detected (health: {current_health:.0f}%)"
        else:
            return f"✅ NORMAL: All sensors operating within spec (health: {current_health:.0f}%)"
    
    def process_batch(
        self,
        df: pd.DataFrame,
        anomaly_scores: np.ndarray,
        threshold: float
    ) -> pd.DataFrame:
        """
        Process batch of multi-sensor data and generate health report.
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
            df_window = df.iloc[:i+1]
            
            fault = self.classify_fault(
                df_window,
                anomaly_scores[i],
                threshold,
                feature='temp_dht'
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
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate summary report of sensor health status."""
        if len(df) == 0:
            return {"error": "No data to analyze"}
        
        current_health = df['health_score'].iloc[-1]
        current_fault = df['fault_type'].iloc[-1]
        current_alert = df['alert_level'].iloc[-1]
        
        fault_counts = df['fault_type'].value_counts().to_dict()
        
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
    np.random.seed(42)
    n = 200
    
    # Healthy period with 6 features
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1S'),
        'temp_dht': 22 + np.random.randn(n) * 0.5,
        'humidity': 55 + np.random.randn(n) * 2,
        'temp_therm': 22 + np.random.randn(n) * 0.4,
        'sound_level': 50 + np.random.randn(n) * 5,
        'light_level': 500 + np.random.randn(n) * 20,
        'flame_intensity': 10 + np.random.randn(n) * 3,
    })
    
    # Inject a drift period
    drift = np.linspace(0, 5, 100)
    df.loc[100:199, 'temp_dht'] += drift
    
    # Simulate anomaly scores
    anomaly_scores = np.concatenate([
        np.random.rand(100) * 0.01,
        np.random.rand(100) * 0.05
    ])
    threshold = 0.02
    
    monitor = SensorHealthMonitor()
    df_processed = monitor.process_batch(df, anomaly_scores, threshold)
    report = monitor.generate_report(df_processed)
    
    print("\n=== Multi-Sensor Health Report ===")
    print(f"Current Health: {report['current_health']:.1f}%")
    print(f"Current Fault: {report['current_fault']}")
    print(f"Uptime: {report['uptime_percent']:.1f}%")
    print(f"\nRecommendation: {report['recommendation']}")
    print(f"\nFault distribution:")
    for fault, count in report['fault_counts'].items():
        print(f"  {fault}: {count}")