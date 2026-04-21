/**
 * =============================================================
 *  ESP32  +  ZPHS01B  →  PM2.5 Prediction Dashboard
 * =============================================================
 *
 *  Wiring (ZPHS01B → ESP32)
 *  ─────────────────────────
 *  ZPHS01B  VCC  →  ESP32  5 V (or 3.3 V — check your module label)
 *  ZPHS01B  GND  →  ESP32  GND
 *  ZPHS01B  TX   →  ESP32  GPIO 16  (Serial2 RX)
 *  ZPHS01B  RX   →  ESP32  GPIO 17  (Serial2 TX)
 *
 *  Serial port mapping
 *  ────────────────────
 *  Serial     → USB debug (115200 baud)
 *  Serial2    → ZPHS01B sensor (9600 8N1)
 *
 *  Protocol summary (26-byte response, big-endian pairs)
 *  ───────────────────────────────────────────────────────
 *  Send query:    FF 01 86 00 00 00 00 00 79
 *  Receive response (26 bytes):
 *    [0]        0xFF  (start)
 *    [1]        0x86  (command echo)
 *    [2][3]     PM1.0  (µg/m³)   = (b[2]<<8)|b[3]
 *    [4][5]     PM2.5  (µg/m³)   = (b[4]<<8)|b[5]
 *    [6][7]     PM10   (µg/m³)   = (b[6]<<8)|b[7]
 *    [8][9]     CO2    (ppm)      = (b[8]<<8)|b[9]
 *    [10]       VOC    (level 1-5)
 *    [11][12]   Temp   (°C×10)   = ((b[11]<<8)|b[12] - 500) * 0.1
 *    [13][14]   Hum    (%RH×10)  = ((b[13]<<8)|b[14]) * 0.1
 *    [15][16]   CH2O   (mg/m³)   = ((b[15]<<8)|b[16]) * 0.001
 *    [17][18]   CO     (mg/m³)   = ((b[17]<<8)|b[18]) * 0.1
 *    [19][20]   O3     (ppb)     = ((b[19]<<8)|b[20]) * 0.01   →  ×1000 for µg/m³ rough
 *    [21][22]   NO2    (ppb)     = ((b[21]<<8)|b[22]) * 0.01
 *    [23][24]   Reserved
 *    [25]       Checksum  = (0xFF - sum(b[1..24])) & 0xFF + 1
 *
 *  Note: NO and NOx are NOT measured by ZPHS01B — the server
 *        fills them with defaults (3.0 / 9.0 µg/m³). If you
 *        have a separate NOx sensor, add it to the JSON payload.
 *
 *  Required libraries (install via Arduino Library Manager):
 *    · ArduinoJson  ≥ 7.x  (by Benoit Blanchon)
 *    · WiFi         (bundled with ESP32 Arduino core)
 *    · HTTPClient   (bundled with ESP32 Arduino core)
 * =============================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ─── USER CONFIG ─────────────────────────────────────────────────────
const char* WIFI_SSID     = "IQOO";      // ← change this
const char* WIFI_PASSWORD = "qwerty12345";  // ← change this

// IP of the PC running server.py  (run `ipconfig` on that PC)
// Make sure ESP32 and PC are on the same WiFi network.
const char* DASHBOARD_URL = "http://10.150.21.7:5000/api/reading";  // ← change IP

// How often to send a reading (milliseconds). Minimum ~3 s for stable values.
const unsigned long SEND_INTERVAL_MS = 10000UL;   // 10 seconds

// Sensor warm-up time (milliseconds). ZPHS01B needs ~3 min to stabilise.
const unsigned long WARMUP_MS = 30000UL;   // 30 s for quick testing; use 180000 in production
// ─────────────────────────────────────────────────────────────────────

// ZPHS01B query command (9 bytes)
const uint8_t QUERY_CMD[9] = { 0xFF, 0x01, 0x86, 0x00, 0x00, 0x00, 0x00, 0x00, 0x79 };
const uint8_t RESPONSE_LEN  = 26;
const uint16_t READ_TIMEOUT_MS = 2000;   // max wait for full 26-byte response

// Serial2 pins
#define SENSOR_RX_PIN 16
#define SENSOR_TX_PIN 17

// Onboard LED
#define LED_PIN 2

// ─── Parsed reading ───────────────────────────────────────────────────
struct ZPHS01B_Data {
  float pm1;    // µg/m³
  float pm25;   // µg/m³
  float pm10;   // µg/m³
  float co2;    // ppm
  uint8_t voc;  // level 1–5
  float temp;   // °C
  float hum;    // %RH
  float ch2o;   // mg/m³ (formaldehyde / HCHO)
  float co;     // mg/m³
  float o3;     // ppb
  float no2;    // ppb
  bool  valid;
};

// ─── State ────────────────────────────────────────────────────────────
unsigned long lastSendTime = 0;
bool warmupDone = false;

// ─────────────────────────────────────────────────────────────────────
// ZPHS01B: send query, read 26-byte response, validate checksum, parse
// ─────────────────────────────────────────────────────────────────────
ZPHS01B_Data readSensor() {
  ZPHS01B_Data d = {};
  d.valid = false;

  // Flush any stray bytes
  while (Serial2.available()) Serial2.read();

  // Send query
  Serial2.write(QUERY_CMD, sizeof(QUERY_CMD));

  // Wait for response
  uint8_t buf[RESPONSE_LEN];
  unsigned long start = millis();
  uint8_t idx = 0;

  while ((millis() - start) < READ_TIMEOUT_MS && idx < RESPONSE_LEN) {
    if (Serial2.available()) {
      buf[idx++] = (uint8_t)Serial2.read();
    }
  }

  if (idx < RESPONSE_LEN) {
    Serial.printf("[Sensor] Timeout – only %d / %d bytes received\n", idx, RESPONSE_LEN);
    return d;
  }

  // Validate header
  if (buf[0] != 0xFF || buf[1] != 0x86) {
    Serial.printf("[Sensor] Bad header: 0x%02X 0x%02X\n", buf[0], buf[1]);
    // Attempt to resync: look for 0xFF in the buffer
    return d;
  }

  // Validate checksum  (sum of bytes 1–24 + checksum byte should = 0xFF)
  uint8_t sum = 0;
  for (uint8_t i = 1; i <= 24; i++) sum += buf[i];
  uint8_t expected_cs = (uint8_t)(~sum + 1);   // two's complement
  if (buf[25] != expected_cs) {
    Serial.printf("[Sensor] Checksum mismatch: got 0x%02X, expected 0x%02X\n",
                  buf[25], expected_cs);
    return d;
  }

  // Parse all fields
  d.pm1   = (float)((buf[2]  << 8) | buf[3]);
  d.pm25  = (float)((buf[4]  << 8) | buf[5]);
  d.pm10  = (float)((buf[6]  << 8) | buf[7]);
  d.co2   = (float)((buf[8]  << 8) | buf[9]);
  d.voc   = buf[10];
  d.temp  = (float)(((buf[11] << 8) | buf[12]) - 500) * 0.1f;
  d.hum   = (float)((buf[13]  << 8) | buf[14]) * 0.1f;
  d.ch2o  = (float)((buf[15]  << 8) | buf[16]) * 0.001f;
  d.co    = (float)((buf[17]  << 8) | buf[18]) * 0.1f;
  d.o3    = (float)((buf[19]  << 8) | buf[20]) * 0.01f;   // ppb
  d.no2   = (float)((buf[21]  << 8) | buf[22]) * 0.01f;   // ppb
  d.valid = true;
  return d;
}

// ─────────────────────────────────────────────────────────────────────
// POST reading to Flask dashboard
// ─────────────────────────────────────────────────────────────────────
bool postToDashboard(const ZPHS01B_Data& d) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Not connected – skipping POST");
    return false;
  }

  // Build JSON  ── model needs: PM10, PM25, PM1, O3, CO, NO, NO2, NOx, CO2
  // NO and NOx are not available from ZPHS01B; server uses default values.
  JsonDocument doc;
  doc["PM1"]  = round(d.pm1  * 100) / 100.0;
  doc["PM25"] = round(d.pm25 * 100) / 100.0;
  doc["PM10"] = round(d.pm10 * 100) / 100.0;
  doc["CO2"]  = round(d.co2);
  doc["O3"]   = round(d.o3   * 100) / 100.0;   // ppb
  doc["CO"]   = round(d.co   * 1000) / 1000.0; // mg/m³
  doc["NO2"]  = round(d.no2  * 100) / 100.0;   // ppb
  // Extras (for dashboard display only — not all used by model)
  doc["TEMP"] = round(d.temp * 10) / 10.0;
  doc["HUM"]  = round(d.hum  * 10) / 10.0;
  doc["CH2O"] = round(d.ch2o * 1000) / 1000.0;
  doc["VOC"]  = d.voc;

  String payload;
  serializeJson(doc, payload);

  HTTPClient http;
  http.begin(DASHBOARD_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);

  int httpCode = http.POST(payload);

  if (httpCode == 200) {
    String resp = http.getString();
    Serial.printf("[HTTP] OK  →  %s\n", resp.c_str());
    http.end();
    return true;
  } else {
    Serial.printf("[HTTP] Error %d: %s\n", httpCode, http.errorToString(httpCode).c_str());
    http.end();
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────
// WiFi connect helper
// ─────────────────────────────────────────────────────────────────────
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to  %s ", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WiFi] Connected!  IP: %s\n", WiFi.localIP().toString().c_str());
}

// ─────────────────────────────────────────────────────────────────────
// setup
// ─────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Init sensor UART
  Serial2.begin(9600, SERIAL_8N1, SENSOR_RX_PIN, SENSOR_TX_PIN);
  Serial.println("\n[ZPHS01B] Sensor UART started on Serial2 (GPIO16/17)");

  // Connect WiFi
  connectWiFi();

  // Warm-up notice
  if (WARMUP_MS > 0) {
    Serial.printf("[Sensor] Warming up for %lu s …\n", WARMUP_MS / 1000);
    for (unsigned long t = 0; t < WARMUP_MS; t += 1000) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(1000);
      Serial.print(".");
    }
    Serial.println("\n[Sensor] Warm-up complete.");
  }

  warmupDone = true;
  digitalWrite(LED_PIN, HIGH);  // solid ON = ready

  Serial.println("[Setup] Ready.  Starting measurement loop.");
  Serial.printf("  Dashboard URL : %s\n", DASHBOARD_URL);
  Serial.printf("  Send interval : %lu s\n\n", SEND_INTERVAL_MS / 1000);
}

// ─────────────────────────────────────────────────────────────────────
// loop
// ─────────────────────────────────────────────────────────────────────
void loop() {
  // Reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Connection lost — reconnecting…");
    connectWiFi();
  }

  unsigned long now = millis();
  if (now - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = now;

    Serial.println("─────────────────────────────────");

    // Read sensor
    ZPHS01B_Data data = readSensor();

    if (!data.valid) {
      Serial.println("[Sensor] Invalid reading — skipping.");
      return;
    }

    // Print to Serial monitor
    Serial.printf("[Reading]\n");
    Serial.printf("  PM1.0  : %6.1f  µg/m³\n", data.pm1);
    Serial.printf("  PM2.5  : %6.1f  µg/m³\n", data.pm25);
    Serial.printf("  PM10   : %6.1f  µg/m³\n", data.pm10);
    Serial.printf("  CO2    : %6.0f  ppm\n",    data.co2);
    Serial.printf("  O3     : %6.2f  ppb\n",    data.o3);
    Serial.printf("  CO     : %6.3f  mg/m³\n",  data.co);
    Serial.printf("  NO2    : %6.2f  ppb\n",    data.no2);
    Serial.printf("  HCHO   : %6.4f  mg/m³\n",  data.ch2o);
    Serial.printf("  Temp   : %5.1f  °C\n",     data.temp);
    Serial.printf("  Hum    : %5.1f  %%RH\n",   data.hum);
    Serial.printf("  VOC    : %d (1-5)\n",       data.voc);

    // POST to dashboard
    Serial.println("[HTTP] Sending to dashboard…");
    bool ok = postToDashboard(data);
    if (ok) {
      // Quick blink on success
      digitalWrite(LED_PIN, LOW);
      delay(80);
      digitalWrite(LED_PIN, HIGH);
    }
  }
}
