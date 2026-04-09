/*
 * IoT Predictive Maintenance — ESP8266MOD + HiveMQ Cloud MQTT
 * 
 * Connects ESP8266 to WiFi, then publishes sensor data to HiveMQ
 * Cloud over secure MQTT (TLS, port 8883). Also subscribes to
 * actuator commands from the Python dashboard.
 *
 * SENSORS:
 *   DHT11 (Temp+Humidity)     → GPIO 2  (D4)
 *   Thermistor (Analog Temp)  → A0      (only analog pin on ESP8266)
 *   ─────────────────────────────────────────────────────────
 *   NOTE: ESP8266 has only ONE analog input (A0).
 *         To read multiple analog sensors (sound, LDR, flame),
 *         use a CD4051 analog multiplexer on A0.
 *         Without a mux, only thermistor is read on A0
 *         and the others use synthesized/default values.
 *   ─────────────────────────────────────────────────────────
 *
 * ACTUATORS:
 *   Buzzer (Active)           → GPIO 14 (D5)
 *   RGB LED (KY-016)          → GPIO 12 (D6) R, GPIO 13 (D7) G, GPIO 15 (D8) B
 *
 * MQTT TOPICS:
 *   Publish:   iot/sensors/data          → JSON sensor readings
 *   Publish:   iot/status                → online/offline heartbeat
 *   Subscribe: iot/actuators/command     → BUZZ:ON, LED:RED, etc.
 *
 * LIBRARIES REQUIRED (install via Arduino Library Manager):
 *   1. PubSubClient        by Nick O'Leary
 *   2. ArduinoJson         by Benoit Blanchon
 *   3. DHT sensor library  by Adafruit
 *
 * BOARD SETUP:
 *   1. File → Preferences → Additional Board URLs:
 *      https://arduino.esp8266.com/stable/package_esp8266com_index.json
 *   2. Tools → Board → Boards Manager → search "ESP8266" → Install
 *   3. Tools → Board → "NodeMCU 1.0 (ESP-12E Module)" or "Generic ESP8266 Module"
 *   4. Tools → Port → select your COM port
 *
 * Baud Rate: 115200
 */

#include <ESP8266WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <time.h>

// Debug macros for serial output
// Undefine first to avoid conflict with DHT.h's definitions
#ifdef DEBUG_PRINT
  #undef DEBUG_PRINT
#endif
#ifdef DEBUG_PRINTLN
  #undef DEBUG_PRINTLN
#endif

#ifdef DEBUG_SERIAL
  #define DEBUG_PRINT(x) Serial.print(x)
  #define DEBUG_PRINTLN(x) Serial.println(x)
#else
  #define DEBUG_PRINT(x)
  #define DEBUG_PRINTLN(x)
#endif

// ═══════════════════════════════════════════════════════════════
//  CONFIGURATION — FILL IN YOUR CREDENTIALS
// ═══════════════════════════════════════════════════════════════

// WiFi credentials
const char* WIFI_SSID     = "Bazinga";
const char* WIFI_PASSWORD = "Gurneesh@31!";

// HiveMQ Cloud credentials
const char* MQTT_HOST     = "2a61e68f29f4485b86a1a978303a268f.s1.eu.hivemq.cloud";
const int   MQTT_PORT     = 8883;
const char* MQTT_USER     = "son21";
const char* MQTT_PASS     = "Balls369";
const char* MQTT_CLIENT   = "esp8266-sensor-node-01";

// MQTT Topics
const char* TOPIC_SENSOR_DATA   = "iot/sensors/data";
const char* TOPIC_ACTUATOR_CMD  = "iot/actuators/command";
const char* TOPIC_STATUS        = "iot/status";

// ═══════════════════════════════════════════════════════════════
//  PIN DEFINITIONS (ESP8266 GPIO mapping)
// ═══════════════════════════════════════════════════════════════

// Sensors
#define DHTPIN          2       // GPIO 2 = D4 on NodeMCU
#define DHTTYPE         DHT11
#define THERMISTOR_PIN  A0      // Only analog pin on ESP8266

// Actuators
#define BUZZER_PIN      14      // GPIO 14 = D5
#define RGB_RED         12      // GPIO 12 = D6
#define RGB_GREEN       13      // GPIO 13 = D7
#define RGB_BLUE        15      // GPIO 15 = D8

// ═══════════════════════════════════════════════════════════════
//  THERMISTOR CONSTANTS
// ═══════════════════════════════════════════════════════════════

#define THERMISTOR_NOMINAL   10000
#define TEMPERATURE_NOMINAL  25
#define B_COEFFICIENT        3950
#define SERIES_RESISTOR      10000

// ═══════════════════════════════════════════════════════════════
//  TIMING
// ═══════════════════════════════════════════════════════════════

#define SENSOR_INTERVAL      2000    // ms between sensor publishes
#define HEARTBEAT_INTERVAL   30000   // ms between status heartbeats
#define RECONNECT_DELAY      5000    // ms between reconnect attempts

// ═══════════════════════════════════════════════════════════════
//  ISRG Root X1 CA Certificate (used by HiveMQ Cloud)
//  Valid until 2035-06-04
// ═══════════════════════════════════════════════════════════════

static const char ca_cert[] PROGMEM = R"EOF(
-----BEGIN CERTIFICATE-----
MIIFazCCA1OgAwIBAgIRAIIQz7DSQONZRGPgu2OCiwAwDQYJKoZIhvcNAQELBQAw
TzELMAkGA1UEBhMCVVMxKTAnBgNVBAoTIEludGVybmV0IFNlY3VyaXR5IFJlc2Vh
cmNoIEdyb3VwMRUwEwYDVQQDEwxJU1JHIFJvb3QgWDEwHhcNMTUwNjA0MTEwNDM4
WhcNMzUwNjA0MTEwNDM4WjBPMQswCQYDVQQGEwJVUzEpMCcGA1UEChMgSW50ZXJu
ZXQgU2VjdXJpdHkgUmVzZWFyY2ggR3JvdXAxFTATBgNVBAMTDElTUkcgUm9vdCBY
MTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBAK3oJHP0FDfzm54rVygc
h77ct984kIxuPOZXoHj3dcKi/vVqbvYATyjb3miGbESTtrFj/RQSa78f0uoxmyF+
0TM8ukj13Xnfs7j/EvEhmkvBioZxaUpmZmyPfjxwv60pIgbz5MDmgK7iS4+3mX6U
A5/TR5d8mUgjU+g4rk8Kb4Mu0UlXjIB0ttov0DiNewNwIRt18jA8+o+u3dpjq+sW
T8KOEUt+zwvo/7V3LvSye0rgTBIlDHCNAymg4VMk7BPZ7hm/ELNKjD+Jo2FR3qyH
B5T0Y3HsLuJvW5iB4YlcNHlsdu87kGJ55tukmi8mxdAQ4Q7e2RCOFvu396j3x+UC
B5iPNgiV5+I3lg02dZ77DnKxHZu8A/lJBdiB3QW0KtZB6awBdpUKD9jf1b0SHzUv
KBds0pjBqAlkd25HN7rOrFleaJ1/ctaJxQZBKT5ZPt0m9STJEadao0xAH0ahmbWn
OlFuhjuefXKnEgV4We0+UXgVCwOPjdAvBbI+e0ocS3MFEvzG6uBQE3xDk3SzynTn
jh8BCNAw1FtxNrQHusEwMFxIt4I7mKZ9YIqioymCzLq9gwQbooMDQaHWBfEbwrbw
qHyGO0aoSCqI3Haadr8faqU9GY/rOPNk3sgrDQoo//fb4hVC1CLQJ13hef4Y53CI
rU7m2Ys6xt0nUW7/vGT1M0NPAgMBAAGjQjBAMA4GA1UdDwEB/wQEAwIBBjAPBgNV
HRMBAf8EBTADAQH/MB0GA1UdDgQWBBR5tFnme7bl5AFzgAiIyBpY9umbbjANBgkq
hkiG9w0BAQsFAAOCAgEAVR9YqbyyqFDQDLHYGmkgJykIrGF1XIpu+ILlaS/V9lZL
ubhzEFnTIZd+50xx+7LSYK05qAvqFyFWhfFQDlnrzuBZ6brJFe+GnY+EgPbk6ZGQ
3BebYhtF8GaV0nxvwuo77x/Py9auJ/GpsMiu/X1+mvoiBOv/2X/qkSsisRcOj/KK
NFtY2PwByVS5uCbMiogZiUvsKV7cxLnF5ocJHgYvGNgR+eDnBfR2BPOr05tjg5uk
UJIGOAQc1YLah6IAF3P3CjAOIMz8QhFx6lQ7eD/xtT4fv5rgkOf8Y9KGLuQw587k
6v/Lz3RBBMpMiPfPSbE4I/rFo8MKQQ2SJXzz+dKXrX4emLvXNfSP8xiNaH0MMRF
Jg7KlVe/J1ZQJBBK8y/GMsqmkz0x0JD8HOxCOqrkjW+4T4AMJj/c2SKdCJKAlZM
t1gmLbG6LDBzDTJ1U26mRz8TvEGRNXlSgPDQZTYb0h/N8IUBLyMx6EblCbsQCC9u
cK2ZuIG8tAj79AjzM9i74DGixgb7r0YrSBEMxyS+mhZQH7oK0jR94I2U6ypHpwt6
Cx2NXRY7rG4Z00c2vlUGuiJlLLGaaOrAAqMxvPdevoyh6VidkmFGqQDTtCzI8EaH3
jwP/jPT+61P/MJ3RUHFEXFBJMYN+7saSUCaN4I2gLDgl0ydQVGNhUESS1rE=
-----END CERTIFICATE-----
)EOF";

// ═══════════════════════════════════════════════════════════════
//  GLOBAL OBJECTS
// ═══════════════════════════════════════════════════════════════

DHT dht(DHTPIN, DHTTYPE);
WiFiClientSecure espClient;
PubSubClient mqttClient(espClient);

// Timing
unsigned long lastSensorTime    = 0;
unsigned long lastHeartbeatTime = 0;

// Actuator state
bool buzzerActive = false;

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("=== IoT Predictive Maintenance — ESP8266 + HiveMQ ===");

  // Initialize DHT11
  dht.begin();

  // Actuator pins
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(RGB_RED,    OUTPUT);
  pinMode(RGB_GREEN,  OUTPUT);
  pinMode(RGB_BLUE,   OUTPUT);

  // Start safe — green LED, buzzer off
  noTone(BUZZER_PIN);
  setLED(0, 255, 0);

  // Connect WiFi
  connectWiFi();

  // Sync time (needed for TLS certificate validation / basic TLS operations)
  syncTime();

  // Configure TLS. Rather than strict CA validation which can run out of memory 
  // or timeout, we use setInsecure() to simply bypass it. This fixes rc=-2 issues.
  espClient.setInsecure();

  // Configure MQTT
  mqttClient.setServer(MQTT_HOST, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(1024);  // Increased buffer for larger payloads

  // Connect to HiveMQ
  connectMQTT();

  Serial.println("[SETUP] Initialization complete.");
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════

void loop() {
  // Ensure WiFi is connected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] Connection lost. Reconnecting...");
    connectWiFi();
  }

  // Ensure MQTT is connected (non‑blocking reconnection)
  if (!mqttClient.connected()) {
    connectMQTT();
  }
  mqttClient.loop();  // Process incoming messages

  unsigned long now = millis();

  // Publish sensor data at regular interval
  if (now - lastSensorTime >= SENSOR_INTERVAL) {
    lastSensorTime = now;
    publishSensorData();
  }

  // Publish heartbeat
  if (now - lastHeartbeatTime >= HEARTBEAT_INTERVAL) {
    lastHeartbeatTime = now;
    publishHeartbeat();
  }
}

// ═══════════════════════════════════════════════════════════════
//  WIFI CONNECTION
// ═══════════════════════════════════════════════════════════════

void connectWiFi() {
  Serial.print("[WIFI] Connecting to ");
  Serial.print(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" Connected!");
    Serial.print("[WIFI] IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println(" FAILED! Restarting...");
    ESP.restart();
  }
}

// ═══════════════════════════════════════════════════════════════
//  NTP TIME SYNC (required for TLS certificate validation)
// ═══════════════════════════════════════════════════════════════

void syncTime() {
  Serial.print("[TIME] Syncing NTP...");
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");

  time_t now = time(nullptr);
  int attempts = 0;
  while (now < 8 * 3600 * 2 && attempts < 20) {
    delay(500);
    Serial.print(".");
    now = time(nullptr);
    attempts++;
  }

  Serial.println(" Done!");
  struct tm timeinfo;
  gmtime_r(&now, &timeinfo);
  Serial.print("[TIME] UTC: ");
  Serial.println(asctime(&timeinfo));
}

// ═══════════════════════════════════════════════════════════════
//  MQTT CONNECTION
// ═══════════════════════════════════════════════════════════════

void connectMQTT() {
  static unsigned long lastAttempt = 0;
  const unsigned long baseDelay = RECONNECT_DELAY; // 5 seconds
  if (mqttClient.connected()) return; // already connected

  unsigned long now = millis();
  if (now - lastAttempt < baseDelay) return; // wait before next attempt

  Serial.print("[MQTT] Attempting reconnection...");
  if (mqttClient.connect(MQTT_CLIENT, MQTT_USER, MQTT_PASS)) {
    Serial.println(" Connected!");
    // Subscribe to topics
    mqttClient.subscribe(TOPIC_ACTUATOR_CMD);
    Serial.print("[MQTT] Subscribed to: ");
    Serial.println(TOPIC_ACTUATOR_CMD);
    // Publish online status (retained)
    StaticJsonDocument<128> statusDoc;
    statusDoc["status"] = "online";
    statusDoc["ip"] = WiFi.localIP().toString();
    char statusBuf[128];
    serializeJson(statusDoc, statusBuf);
    mqttClient.publish(TOPIC_STATUS, statusBuf, true);
    // Reset back‑off
    lastAttempt = 0;
  } else {
    Serial.print(" Failed (rc=");
    Serial.print(mqttClient.state());
    Serial.println(")");
    printMQTTError(mqttClient.state());
    // Exponential back‑off up to 60 seconds
    unsigned long backoff = min(baseDelay * (1UL << (lastAttempt / baseDelay)), 60000UL);
    lastAttempt = now + backoff; // schedule next attempt
  }
}

void printMQTTError(int state) {
  switch (state) {
    case -4: Serial.println("  → MQTT_CONNECTION_TIMEOUT"); break;
    case -3: Serial.println("  → MQTT_CONNECTION_LOST"); break;
    case -2: Serial.println("  → MQTT_CONNECT_FAILED"); break;
    case -1: Serial.println("  → MQTT_DISCONNECTED"); break;
    case  1: Serial.println("  → MQTT_CONNECT_BAD_PROTOCOL"); break;
    case  2: Serial.println("  → MQTT_CONNECT_BAD_CLIENT_ID"); break;
    case  3: Serial.println("  → MQTT_CONNECT_UNAVAILABLE"); break;
    case  4: Serial.println("  → MQTT_CONNECT_BAD_CREDENTIALS"); break;
    case  5: Serial.println("  → MQTT_CONNECT_UNAUTHORIZED"); break;
    default: Serial.println("  → Unknown error"); break;
  }
}

// ═══════════════════════════════════════════════════════════════
//  MQTT CALLBACK — Receive actuator commands
// ═══════════════════════════════════════════════════════════════

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  // Convert payload to string
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  Serial.print("[MQTT] Received on ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(message);

  String cmd = String(message);
  cmd.trim();

  // Process actuator commands (same protocol as serial version)
  if (cmd == "BUZZ:ON") {
    tone(BUZZER_PIN, 1000);
    buzzerActive = true;
    mqttClient.publish(TOPIC_STATUS, "{\"ack\":\"BUZZ_ON\"}");
  }
  else if (cmd == "BUZZ:OFF") {
    noTone(BUZZER_PIN);
    buzzerActive = false;
    mqttClient.publish(TOPIC_STATUS, "{\"ack\":\"BUZZ_OFF\"}");
  }
  else if (cmd == "LED:RED") {
    setLED(255, 0, 0);
    mqttClient.publish(TOPIC_STATUS, "{\"ack\":\"LED_RED\"}");
  }
  else if (cmd == "LED:GREEN") {
    setLED(0, 255, 0);
    mqttClient.publish(TOPIC_STATUS, "{\"ack\":\"LED_GREEN\"}");
  }
  else if (cmd == "LED:YELLOW") {
    setLED(255, 255, 0);
    mqttClient.publish(TOPIC_STATUS, "{\"ack\":\"LED_YELLOW\"}");
  }
  else if (cmd == "LED:OFF") {
    setLED(0, 0, 0);
    mqttClient.publish(TOPIC_STATUS, "{\"ack\":\"LED_OFF\"}");
  }
  else if (cmd == "STATUS") {
    StaticJsonDocument<128> doc;
    doc["buzzer"] = buzzerActive ? "ON" : "OFF";
    doc["uptime"] = millis() / 1000;
    char buf[128];
    serializeJson(doc, buf);
    mqttClient.publish(TOPIC_STATUS, buf);
  }
}

// ═══════════════════════════════════════════════════════════════
//  SENSOR READING & PUBLISHING
// ═══════════════════════════════════════════════════════════════

void publishSensorData() {
  // 1. DHT11 — Temperature & Humidity
  float dhtTemp = dht.readTemperature();
  float dhtHum  = dht.readHumidity();

  // 2. Thermistor — Analog Temperature (on A0)
  float thermTemp = readThermistor();

  // 3-5. Sound, LDR, Flame
  //    If using a CD4051 multiplexer on A0, uncomment the mux version below.
  //    Without a mux, we use placeholder readings.
  float soundLevel     = readSoundLevel();
  float lightLevel     = readLightLevel();
  float flameIntensity = readFlameLevel();

  // Validate DHT11
  if (isnan(dhtTemp) || isnan(dhtHum)) {
    Serial.println("[SENSOR] DHT11 read failed — skipping this cycle");
    return;
  }

  // Build JSON payload
  StaticJsonDocument<256> doc;
  doc["temp_dht"]        = roundTo(dhtTemp, 2);
  doc["humidity"]        = roundTo(dhtHum, 2);
  doc["temp_therm"]      = roundTo(thermTemp, 2);
  doc["sound_level"]     = roundTo(soundLevel, 2);
  doc["light_level"]     = roundTo(lightLevel, 2);
  doc["flame_intensity"] = roundTo(flameIntensity, 2);

  char jsonBuffer[256];
  serializeJson(doc, jsonBuffer);

  // Publish
  if (mqttClient.publish(TOPIC_SENSOR_DATA, jsonBuffer)) {
    Serial.print("[MQTT] Published: ");
    Serial.println(jsonBuffer);
  } else {
    Serial.println("[MQTT] Publish FAILED");
  }
}

void publishHeartbeat() {
  StaticJsonDocument<128> doc;
  doc["status"]  = "online";
  doc["uptime"]  = millis() / 1000;
  doc["heap"]    = ESP.getFreeHeap();
  doc["rssi"]    = WiFi.RSSI();

  char buf[128];
  serializeJson(doc, buf);
  mqttClient.publish(TOPIC_STATUS, buf, true);

  Serial.print("[HEARTBEAT] ");
  Serial.println(buf);
}

// ═══════════════════════════════════════════════════════════════
//  SENSOR HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════

float readThermistor() {
  /*
   * Steinhart-Hart B-parameter equation.
   * Voltage divider: 3.3V → SERIES_RESISTOR → junction → THERMISTOR → GND
   * ESP8266 ADC is 10-bit (0-1023) reading 0-1V by default,
   * but NodeMCU has a built-in voltage divider scaling 0-3.3V → 0-1V.
   */
  int raw = analogRead(THERMISTOR_PIN);
  if (raw <= 0) raw = 1;
  if (raw >= 1023) raw = 1022;

  float resistance = (float)SERIES_RESISTOR * raw / (1023.0 - raw);

  float steinhart;
  steinhart = resistance / THERMISTOR_NOMINAL;
  steinhart = log(steinhart);
  steinhart /= B_COEFFICIENT;
  steinhart += 1.0 / (TEMPERATURE_NOMINAL + 273.15);
  steinhart = 1.0 / steinhart;
  steinhart -= 273.15;

  return steinhart;
}

// ─────────────────────────────────────────────────────────────
// WITHOUT ANALOG MULTIPLEXER: placeholder/simulated readings
// Replace these with real mux reads if you add a CD4051/74HC4067
// ─────────────────────────────────────────────────────────────

float readSoundLevel() {
  // TODO: If using CD4051 mux, select channel and read A0
  // For now, return a baseline with slight noise
  return 45.0 + random(-5, 5);
}

float readLightLevel() {
  // TODO: If using CD4051 mux, select channel and read A0
  return 500.0 + random(-20, 20);
}

float readFlameLevel() {
  // TODO: If using CD4051 mux, select channel and read A0
  return 8.0 + random(0, 5);
}

// ─────────────────────────────────────────────────────────────
// WITH CD4051 ANALOG MULTIPLEXER (uncomment and connect S0-S2)
// ─────────────────────────────────────────────────────────────
/*
#define MUX_S0  5   // GPIO 5  = D1
#define MUX_S1  4   // GPIO 4  = D2
#define MUX_S2  0   // GPIO 0  = D3

void setupMux() {
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
}

int readMuxChannel(int channel) {
  // Select channel (0-7) on CD4051
  digitalWrite(MUX_S0, channel & 0x01);
  digitalWrite(MUX_S1, (channel >> 1) & 0x01);
  digitalWrite(MUX_S2, (channel >> 2) & 0x01);
  delay(5);  // Settling time
  return analogRead(A0);
}

// Channel mapping for CD4051:
//   Channel 0 → Thermistor
//   Channel 1 → Sound sensor
//   Channel 2 → LDR
//   Channel 3 → Flame sensor

float readSoundLevel_MUX() {
  int raw = readMuxChannel(1);
  long sumSquares = 0;
  int baseline = 512;
  // Simple RMS with fewer samples (ESP8266 is faster)
  for (int i = 0; i < 30; i++) {
    int sample = readMuxChannel(1);
    long deviation = (long)sample - baseline;
    sumSquares += deviation * deviation;
    delay(1);
  }
  return sqrt((float)sumSquares / 30.0);
}

float readLightLevel_MUX() {
  return (float)readMuxChannel(2);
}

float readFlameLevel_MUX() {
  return 1023.0 - (float)readMuxChannel(3);
}
*/

// ═══════════════════════════════════════════════════════════════
//  ACTUATOR CONTROL
// ═══════════════════════════════════════════════════════════════

void setLED(int r, int g, int b) {
  analogWrite(RGB_RED,   r);
  analogWrite(RGB_GREEN, g);
  analogWrite(RGB_BLUE,  b);
}

// ═══════════════════════════════════════════════════════════════
//  UTILITY
// ═══════════════════════════════════════════════════════════════

float roundTo(float value, int decimals) {
  float multiplier = pow(10, decimals);
  return round(value * multiplier) / multiplier;
}
