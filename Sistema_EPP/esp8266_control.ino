// ============================================================
//  ESP8266 Control EPP — LEDs + Botón para Sistema de EPP
//  Conexión WiFi → Flask API en servidor
// ============================================================
//  CONEXIONES:
//
//  LED Rojo (FALTA casco/chaleco):
//    Ánodo ─── GPIO5 (D1)  →  resistencia 220Ω → GND
//
//  LED Amarillo (WiFi):
//    Ánodo ─── GPIO13 (D7) →  resistencia 220Ω → GND
//
//  Pushbutton (INICIAR VERIFICACIÓN):
//    Pin 1 ─── GPIO4 (D2)
//    Pin 2 ─── GND
//    (usa pull-up interno, presionado = LOW)
//
//  LED Verde (BIENVENIDO):
//    Ánodo ─── GPIO2 (D4)  →  resistencia 220Ω → GND
//    ⚠ GPIO2 debe estar HIGH durante boot (no forzar a GND)
//
//  Alimentación:
//    ESP8266 VCC ─── 3.3V
//    ESP8266 GND ─── GND (compartido con LEDs y botón)
// ============================================================

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

// ── WiFi ──
const char* ssid     = "perroLobo";
const char* password = "perroloba123";

// ── Servidor Flask ──
const char* serverHost  = "192.168.100.6";
const uint16_t serverPort = 5000;

// ── Pines GPIO ──
#define PIN_LED_ROJO    5   // GPIO5  / D1  — alerta: falta EPP
#define PIN_BOTON       4   // GPIO4  / D2  — iniciar verificación
#define PIN_LED_VERDE   2   // GPIO2  / D4  — BIENVENIDO
#define PIN_LED_WIFI   13   // GPIO13 / D7  — estado WiFi

// ── Tiempos ──
const unsigned long POLL_INTERVAL   = 2000;   // consultar estado cada 2s
const unsigned long DEBOUNCE_DELAY  = 50;     // antirrebote botón
const unsigned long RECONNECT_DELAY = 3000;   // esperar entre reconexiones

// ── Variables globales ──
WiFiClient wifiClient;
HTTPClient http;

bool wifiConnected   = false;
bool lastButtonState = HIGH;
unsigned long tPoll     = 0;
unsigned long tLed      = 0;
unsigned long tButton   = 0;
bool parpadeoState = false;

// Estado actual desde el servidor
struct {
  bool   helmet_ok   = false;
  bool   vest_ok     = false;
  bool   both_ok     = false;
  bool   active      = false;
  String ver_state   = "idle";    // idle | verifying | done
  bool   ver_allowed = false;     // true solo cuando BIENVENIDO
} estado;

// ── FUNCIONES AUXILIARES ──

String getURL(const char* path) {
  return String("http://") + serverHost + ":" + serverPort + path;
}

bool consultarEstado() {
  String url = getURL("/api/esp/estado");
  http.begin(wifiClient, url);
  http.setTimeout(2000);
  int code = http.GET();
  if (code != 200) { http.end(); return false; }
  String body = http.getString();
  http.end();

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) return false;

  estado.helmet_ok   = doc["helmet_ok"]   | false;
  estado.vest_ok     = doc["vest_ok"]     | false;
  estado.both_ok     = doc["both_ok"]     | false;
  estado.active      = doc["active"]      | false;
  estado.ver_state   = doc["ver_state"]   | "idle";
  estado.ver_allowed = doc["ver_allowed"] | false;
  return true;
}

bool enviarVerificar() {
  String url = getURL("/api/esp/verificar");
  http.begin(wifiClient, url);
  http.setTimeout(2000);
  int code = http.POST("");
  http.end();
  return (code == 200);
}

void actualizarLEDs() {
  unsigned long now = millis();

  // ── LED WiFi (Amarillo) ──
  if (WiFi.status() == WL_CONNECTED && wifiConnected) {
    digitalWrite(PIN_LED_WIFI, HIGH);   // fijo = conectado
  } else {
    if (now - tLed >= 250) {
      tLed = now;
      parpadeoState = !parpadeoState;
      digitalWrite(PIN_LED_WIFI, parpadeoState ? HIGH : LOW);
    }
  }

  // ── LED Verde (BIENVENIDO) ──
  // Se enciende solo cuando la verificación terminó y fue exitosa
  if (estado.ver_state == "done" && estado.ver_allowed) {
    digitalWrite(PIN_LED_VERDE, HIGH);
  } else {
    digitalWrite(PIN_LED_VERDE, LOW);
  }

  // ── LED Rojo (falta EPP) ──
  // Se enciende cuando falta casco o chaleco en el feed activo
  if (wifiConnected && !estado.both_ok && estado.active) {
    digitalWrite(PIN_LED_ROJO, HIGH);
  } else {
    digitalWrite(PIN_LED_ROJO, LOW);
  }
}

void leerBoton() {
  bool actual = digitalRead(PIN_BOTON);
  if (actual == LOW && lastButtonState == HIGH && millis() - tButton > DEBOUNCE_DELAY) {
    tButton = millis();
    Serial.println("[BOTON] -> INICIAR VERIFICACION");
    if (enviarVerificar()) {
      Serial.println("[BOTON] OK");
    } else {
      Serial.println("[BOTON] FALLO");
    }
  }
  lastButtonState = actual;
}

// ── SETUP ──
void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("=== ESP8266 Control EPP ===");

  pinMode(PIN_LED_ROJO,  OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_WIFI,  OUTPUT);
  pinMode(PIN_BOTON,     INPUT_PULLUP);

  digitalWrite(PIN_LED_ROJO,  LOW);
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_WIFI,  LOW);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Conectando WiFi...");
}

// ── LOOP ──
void loop() {
  unsigned long now = millis();

  leerBoton();
  actualizarLEDs();

  if (WiFi.status() == WL_CONNECTED) {
    if (!wifiConnected) {
      wifiConnected = true;
      Serial.println();
      Serial.print("IP: ");
      Serial.println(WiFi.localIP());
    }

    if (now - tPoll >= POLL_INTERVAL) {
      tPoll = now;
      if (consultarEstado()) {
        Serial.print("casco=");
        Serial.print(estado.helmet_ok ? "OK" : "FALTA");
        Serial.print(" chaleco=");
        Serial.print(estado.vest_ok ? "OK" : "FALTA");
        Serial.print(" ver=");
        Serial.print(estado.ver_state.c_str());
        Serial.print(" bienvenido=");
        Serial.print(estado.ver_allowed ? "SI" : "NO");
        Serial.println();
      }
    }
  } else {
    if (wifiConnected) {
      wifiConnected = false;
      Serial.println("\nWiFi PERDIDO");
    }
  }
}
