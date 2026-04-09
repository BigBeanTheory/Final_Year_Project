"""
MQTT Subscriber for IoT Predictive Maintenance Dashboard
Connects to HiveMQ Cloud and receives sensor data from ESP8266.
Also publishes actuator commands (buzzer, LED) to the ESP8266.

Usage:
    # Standalone test
    python iot_mqtt_subscriber.py

    # Import in dashboard
    from iot_mqtt_subscriber import MQTTSensorBridge
    bridge = MQTTSensorBridge()
    bridge.start()
    reading = bridge.get_latest_reading()
    bridge.send_command("BUZZ:ON")
"""

import json
import ssl
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from typing import Optional, Dict, Callable

import paho.mqtt.client as mqtt

import config


# ============================================================================
# MQTT CONFIGURATION (mirrors the ESP8266 sketch settings)
# ============================================================================

MQTT_BROKER_HOST = "2a61e68f29f4485b86a1a978303a268f.s1.eu.hivemq.cloud"
MQTT_BROKER_PORT = 8883
MQTT_USERNAME    = "son21"
MQTT_PASSWORD    = "Balls369"
MQTT_CLIENT_ID   = "iot-dashboard-subscriber"
MQTT_USE_TLS     = True

# Topics — must match the ESP8266 sketch
TOPIC_SENSOR_DATA   = "iot/sensors/data"
TOPIC_ACTUATOR_CMD  = "iot/actuators/command"
TOPIC_STATUS        = "iot/status"


class MQTTSensorBridge:
    """
    Thread-safe MQTT bridge for receiving sensor data and sending
    actuator commands via HiveMQ Cloud.

    Provides the same interface as serial-based ingestion so the
    dashboard can swap between serial and MQTT transparently.
    """

    def __init__(self,
                 host: str = MQTT_BROKER_HOST,
                 port: int = MQTT_BROKER_PORT,
                 username: str = MQTT_USERNAME,
                 password: str = MQTT_PASSWORD,
                 client_id: str = MQTT_CLIENT_ID,
                 use_tls: bool = MQTT_USE_TLS,
                 max_queue_size: int = 1000):
        """
        Initialize the MQTT bridge.

        Args:
            host: HiveMQ Cloud broker URL
            port: Broker port (8883 for TLS)
            username: MQTT username from HiveMQ Access Management
            password: MQTT password
            client_id: Unique client ID for this subscriber
            use_tls: Whether to use TLS (must be True for HiveMQ Cloud)
            max_queue_size: Max readings to buffer before dropping oldest
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.use_tls = use_tls

        # Thread-safe data storage
        self._data_queue = Queue(maxsize=max_queue_size)
        self._latest_reading = None
        self._lock = threading.Lock()
        self._all_readings = []  # Rolling history

        # Connection state
        self._connected = False
        self._running = False
        self._thread = None
        self._device_status = {}

        # Callbacks for dashboard integration
        self._on_data_callback: Optional[Callable] = None
        self._on_connect_callback: Optional[Callable] = None
        self._on_disconnect_callback: Optional[Callable] = None

        # MQTT client setup
        self._client = mqtt.Client(
            client_id=self.client_id,
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self._client.username_pw_set(self.username, self.password)

        if self.use_tls:
            self._client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
            # For production: verify server certificate
            # self._client.tls_set(ca_certs="path/to/isrg-root-x1.pem",
            #                      tls_version=ssl.PROTOCOL_TLSv1_2)

        # Bind callbacks
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the MQTT connection in a background thread."""
        if self._running:
            print("[MQTT] Already running.")
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[MQTT] Connecting to {self.host}:{self.port}...")

        # Wait briefly for connection
        for _ in range(10):
            if self._connected:
                return True
            time.sleep(0.5)

        if not self._connected:
            print("[MQTT] Connection timeout — will keep retrying in background.")
        return self._connected

    def stop(self):
        """Stop the MQTT connection gracefully."""
        self._running = False
        try:
            self._client.disconnect()
            self._client.loop_stop()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._connected = False
        print("[MQTT] Stopped.")

    def is_connected(self) -> bool:
        """Check if currently connected to HiveMQ Cloud."""
        return self._connected

    def get_latest_reading(self) -> Optional[Dict]:
        """
        Get the most recent sensor reading.
        Returns None if no data has been received yet.

        Format matches SensorDataIngestion.parse_serial_line() output:
        {
            'timestamp': datetime,
            'temp_dht': float,
            'humidity': float,
            'temp_therm': float,
            'sound_level': float,
            'light_level': float,
            'flame_intensity': float,
        }
        """
        with self._lock:
            return self._latest_reading

    def get_buffered_readings(self, max_count: int = 100) -> list:
        """
        Get buffered readings from the queue (non-blocking, FIFO).
        Returns up to max_count readings.
        """
        readings = []
        while len(readings) < max_count:
            try:
                readings.append(self._data_queue.get_nowait())
            except Empty:
                break
        return readings

    def get_all_readings(self) -> list:
        """Get all readings received since start (rolling history)."""
        with self._lock:
            return list(self._all_readings)

    def get_reading_count(self) -> int:
        """Get total number of readings received."""
        with self._lock:
            return len(self._all_readings)

    def get_device_status(self) -> Dict:
        """Get the latest device status (heartbeat info)."""
        with self._lock:
            return dict(self._device_status)

    def send_command(self, command: str) -> bool:
        """
        Send an actuator command to the ESP8266.

        Supported commands:
            BUZZ:ON, BUZZ:OFF
            LED:RED, LED:GREEN, LED:YELLOW, LED:OFF
            STATUS

        Returns True if published successfully.
        """
        if not self._connected:
            print(f"[MQTT] Cannot send command — not connected.")
            return False

        try:
            result = self._client.publish(TOPIC_ACTUATOR_CMD, command, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] Command sent: {command}")
                return True
            else:
                print(f"[MQTT] Command failed (rc={result.rc})")
                return False
        except Exception as e:
            print(f"[MQTT] Command error: {e}")
            return False

    def set_on_data_callback(self, callback: Callable):
        """
        Set a callback for when new sensor data arrives.
        Callback signature: callback(reading: dict)
        """
        self._on_data_callback = callback

    def set_on_connect_callback(self, callback: Callable):
        """Callback when MQTT connection is established."""
        self._on_connect_callback = callback

    def set_on_disconnect_callback(self, callback: Callable):
        """Callback when MQTT connection is lost."""
        self._on_disconnect_callback = callback

    # ────────────────────────────────────────────────────────────
    #  INTERNAL MQTT CALLBACKS
    # ────────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Called when connected to HiveMQ Cloud."""
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Connected to {self.host}")

            # Subscribe to topics
            client.subscribe(TOPIC_SENSOR_DATA, qos=1)
            client.subscribe(TOPIC_STATUS, qos=0)
            print(f"[MQTT] Subscribed to: {TOPIC_SENSOR_DATA}, {TOPIC_STATUS}")

            if self._on_connect_callback:
                self._on_connect_callback()
        else:
            self._connected = False
            error_messages = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized",
            }
            msg = error_messages.get(rc, f"Unknown error (rc={rc})")
            print(f"[MQTT] Connection failed: {msg}")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        """Called when disconnected from HiveMQ Cloud."""
        self._connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnection (rc={rc}). Will retry...")
        else:
            print("[MQTT] Disconnected cleanly.")

        if self._on_disconnect_callback:
            self._on_disconnect_callback()

    def _on_message(self, client, userdata, msg):
        """Called when a message is received from a subscribed topic."""
        topic = msg.topic
        try:
            payload = msg.payload.decode('utf-8')
        except UnicodeDecodeError:
            return

        if topic == TOPIC_SENSOR_DATA:
            self._process_sensor_data(payload)
        elif topic == TOPIC_STATUS:
            self._process_status(payload)

    # ────────────────────────────────────────────────────────────
    #  DATA PROCESSING
    # ────────────────────────────────────────────────────────────

    def _process_sensor_data(self, payload: str):
        """Parse incoming JSON sensor data from ESP8266."""
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            print(f"[MQTT] Invalid JSON: {payload}")
            return

        # Build reading dict (same format as SensorDataIngestion.parse_serial_line)
        reading = {'timestamp': datetime.now()}

        for feat in config.FEATURE_NAMES:
            if feat in data:
                val = float(data[feat])
                # Validate against sensor ranges and clamp if out of bounds
                lo, hi = config.SENSOR_RANGES[feat]
                if lo <= val <= hi:
                    reading[feat] = val
                else:
                    print(f"[MQTT] Out-of-range {feat}={val} (valid: {lo}-{hi}) — clamping to range")
                    reading[feat] = max(lo, min(hi, val))
            else:
                print(f"[MQTT] Missing field '{feat}' in payload")
                # Fill missing with middle of the range to prevent crashes
                lo, hi = config.SENSOR_RANGES[feat]
                reading[feat] = (hi + lo) / 2.0

        # Store reading
        with self._lock:
            self._latest_reading = reading
            self._all_readings.append(reading)
            # Cap history at 10,000 readings (~2.7 hours at 1/sec)
            if len(self._all_readings) > 10000:
                self._all_readings = self._all_readings[-5000:]

        # Queue for dashboard consumption
        try:
            self._data_queue.put_nowait(reading)
        except Exception:
            # Queue full — drop oldest
            try:
                self._data_queue.get_nowait()
                self._data_queue.put_nowait(reading)
            except Exception:
                pass

        # Fire callback
        if self._on_data_callback:
            try:
                self._on_data_callback(reading)
            except Exception as e:
                print(f"[MQTT] Callback error: {e}")

    def _process_status(self, payload: str):
        """Parse device status/heartbeat messages."""
        try:
            data = json.loads(payload)
            with self._lock:
                self._device_status.update(data)
                self._device_status['last_seen'] = datetime.now()
        except json.JSONDecodeError:
            pass

    # ────────────────────────────────────────────────────────────
    #  BACKGROUND THREAD
    # ────────────────────────────────────────────────────────────

    def _run_loop(self):
        """Background thread: connect and maintain MQTT loop."""
        while self._running:
            if not self._connected:
                try:
                    self._client.connect(self.host, self.port, keepalive=60)
                    self._client.loop_start()
                    # Wait for on_connect callback
                    time.sleep(3)
                except Exception as e:
                    print(f"[MQTT] Connection error: {e}")
                    time.sleep(5)
                    continue

            # Keep thread alive, check periodically
            time.sleep(1)


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MQTT Sensor Bridge — Standalone Test")
    print("=" * 60)
    print(f"Broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
    print(f"User:   {MQTT_USERNAME}")
    print(f"Topics: {TOPIC_SENSOR_DATA}, {TOPIC_ACTUATOR_CMD}")
    print()

    bridge = MQTTSensorBridge()

    # Set up a callback to print each reading
    def on_data(reading):
        print(f"\n[DATA] {reading['timestamp'].strftime('%H:%M:%S')}")
        for feat in config.FEATURE_NAMES:
            if feat in reading:
                label = config.FEATURE_LABELS.get(feat, feat)
                unit = config.FEATURE_UNITS.get(feat, '')
                print(f"  {label}: {reading[feat]:.2f} {unit}")

    bridge.set_on_data_callback(on_data)

    # Start connection
    connected = bridge.start()
    if connected:
        print("\n✅ Connected! Waiting for sensor data...\n")
    else:
        print("\n⏳ Connecting in background... Waiting for sensor data...\n")

    try:
        while True:
            time.sleep(1)
            status = bridge.get_device_status()
            count = bridge.get_reading_count()
            conn = "🟢 Connected" if bridge.is_connected() else "🔴 Disconnected"

            # Status line (overwrite)
            print(f"\r{conn} | Readings: {count} | "
                  f"Device: {status.get('status', 'unknown')} | "
                  f"Heap: {status.get('heap', '?')}",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        bridge.stop()
        print("Done.")
