# Sistema de Detección de EPP con Reconocimiento Facial

Sistema local de control de acceso que verifica uso obligatorio de casco y chaleco mediante visión por computadora, con reconocimiento facial y control por hardware (ESP8266).

---

## Herramientas y Lenguajes Usados

| Herramienta / Lenguaje | Versión | Propósito |
|---|---|---|
| **Python** | 3.10+ | Lenguaje principal del backend |
| **Flask** | 3.x | Framework web (servidor HTTP + API REST) |
| **OpenCV** | 4.x | Captura y procesamiento de video en tiempo real |
| **Ultralytics YOLOv8** | 8.x | Detección de objetos (casco, chaleco) |
| **InsightFace** | 0.7.x | Reconocimiento facial (extracción y comparación de plantillas) |
| **MobileNet (torchvision)** | - | Clasificador auxiliar de chaleco |
| **SQLite 3** | - | Base de datos local |
| **NumPy** | 1.x | Manipulación de arreglos y matrices |
| **HTML / CSS / JavaScript** | - | Interfaz web (Jinja2 templates) |
| **C++ (Arduino)** | - | Firmware del ESP8266 |
| **ESP8266** | - | Microcontrolador WiFi con LEDs y botón |
| **ArduinoJson** | 6.x | Parseo de JSON en el ESP8266 |

---

## Arquitectura del Sistema

```
┌─────────────────────────────┐       ┌──────────────────┐
│   Navegador Web (UI)        │       │  ESP8266          │
│   http://192.168.100.6:5000 │       │  ─ LEDs (rojo,    │
│                             │       │    verde, WiFi)   │
│                             │       │  ─ Botón físico   │
└──────────┬──────────────────┘       └────────┬─────────┘
           │ HTTP                              │ HTTP
           ▼                                    ▼
┌──────────────────────────────────────────────────────┐
│               Flask Server (app.py)                   │
│   ┌──────────┬───────────┬──────────┬─────────────┐  │
│   │ Cámaras  │ Detección │ Facial   │ API REST    │  │
│   │ (hilos)  │ YOLO+HSV  │ FaceNet  │ /api/esp/*  │  │
│   └──────────┴───────────┴──────────┴─────────────┘  │
│               │                                       │
│               ▼                                       │
│   ┌──────────────────────────────────────────────┐   │
│   │          Base de datos SQLite                │   │
│   │   (workers, cameras, violations, events)     │   │
│   └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## Detección de Casco

**Archivo:** `detector.py` — método `_detect_helmet_in_roi()`

### Proceso
1. YOLO se ejecuta sobre el **fotograma completo** (no sobre un recorte)
2. Se filtran detecciones redundantes por **IoU** (superposición > 0.3)
3. La caja del casco se posiciona **encima del rostro** (shift vertical según altura de la cara)
4. Se agrega **40% de padding** a la bounding box
5. Se valida el color con filtro HSV

### Validación de Color (`_helmet_color_ok()`)

Colores permitidos: **azul, amarillo, rojo, blanco** — cualquier otro color (ej. rosa) se rechaza.

| Color | Rango H | Rango S | Rango V |
|-------|---------|---------|---------|
| Azul | 100–130 | 50–255 | 50–255 |
| Amarillo | 20–35 | 50–255 | 50–255 |
| Rojo | 0–10 ó 170–180 | 50–255 | 50–255 |
| Blanco | 0–180 | 0–30 | 200–255 |

Si ≥**15%** del ROI está dentro de algún rango → casco válido. Si no → `no-helmet`.

### Clasificador MobileNet desactivado

El modelo `modelo_casco` en `clasificador_api.py` fue **deshabilitado** (`modelo_casco = None`) porque tenía solo **4 imágenes de `no_casco`** (sombreros) vs 22 de `casco` — sobreajuste total. La detección se hace exclusivamente con YOLO + color.

---

## Detección de Chaleco

**No modificada.** Sigue usando el clasificador MobileNet entrenado.
- Archivo: `chaleco_api.py`
- Método: `_detect_vest_in_roi()`
- Funciona "super excelente" — sin cambios.

---

## Control por Hardware (ESP8266)

**Archivo:** `esp8266_control.ino`

### Conexión WiFi
- **SSID:** `perroLobo`
- **Password:** `perroloba123`
- **Servidor:** `192.168.100.6:5000`

### Pines

| Pin | GPIO | Componente | Resistencia |
|-----|------|------------|-------------|
| D1 | GPIO5 | LED rojo | 220Ω |
| D2 | GPIO4 | Botón pulsador (pull-up) | - |
| D4 | GPIO2 | LED verde | 220Ω |
| D7 | GPIO13 | LED WiFi | 220Ω |

### Lógica de Funcionamiento
- Cada **2 segundos** hace `GET /api/esp/estado` y actualiza LEDs
- **LED rojo (D1):** encendido cuando el feed está activo y `both_ok == false`
- **LED verde (D4):** encendido solo cuando `ver_state == "done"` y `ver_allowed == true` (BIENVENIDO)
- **LED WiFi (D7):** sólido = conectado, parpadea cada 500ms = desconectado/conectando
- **Botón (D2):** envía `POST /api/esp/verificar` → inicia verificación manual en la cámara activa por defecto

---

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/esp/estado` | JSON con `helmet_ok`, `vest_ok`, `both_ok`, `active`, `ver_state` (idle/verifying/done), `ver_allowed` |
| POST | `/api/esp/verificar` | Inicia verificación manual (`session.start_manual()`) |
| POST | `/workers/<id>/delete` | Elimina trabajador (soft-delete, active=0) |
| GET | `/workers` (ó `/personas`) | Lista + formulario de registro de personal |
| POST | `/api/rostros/subir` | Sube foto de rostro para registro biométrico |
| POST | `/api/rostros/reconocer` | Reconoce rostro contra base de datos |

---

## Base de Datos

**Archivo:** `data/epp_events.db` (SQLite)

### Tablas

| Tabla | Propósito |
|-------|-----------|
| `workers` | Personas registradas con plantilla facial (`active=0` = eliminado) |
| `cameras` | Cámaras configuradas |
| `violations` | Infracciones detectadas (sin casco, sin chaleco) |
| `access_events` | Eventos de acceso (permitido/denegado) |

### Funciones principales (`database.py`)
- `add_worker()`, `list_workers()`, `get_worker()`, `delete_worker()`, `save_worker_face()`
- `add_camera()`, `delete_camera()`, `list_cameras()`
- `log_violation()`, `log_access_event()`
- `count_by_type()`, `list_violations()`, `list_access_events()`

---

## Modelo YOLO (`Casco.pt`)

- **Clases:** 4 — `casco`, `no-casco`, `chaleco`, `no-chaleco`
- Un solo modelo cubre tanto casco como chaleco
- Entrenado con imágenes de obras en construcción
- Se ejecuta sobre el fotograma completo a resolución 640×360
- Se procesa cada **12 cuadros** para reducir uso de CPU

---

## Interfaz Web

- Templates en `templates/` con **Jinja2**
- Páginas: control de acceso, gestión de personas, dashboard de violaciones
- Captura de rostro en vivo con countdown de 5 segundos
- Subida de foto para registro biométrico
- Botón **ELIMINAR** en gestión de personas (con confirmación, soft-delete)

---

## Cómo Ejecutar

### Servidor Flask
```bash
cd Sistema_EPP
python app.py
```
Luego abrir en navegador: `http://127.0.0.1:5000`

### ESP8266
1. Abrir `esp8266_control.ino` en Arduino IDE
2. Instalar board ESP8266 y librería ArduinoJson
3. Programar y cablear según tabla de pines
4. Alimentar con 5V

### Cámaras
- Cámara local: `0` ó `1` (índice USB)
- Cámara IP: `192.168.x.x:8080` → se convierte automáticamente a `http://192.168.x.x:8080/video`
- Para probar: ejecutar `probar_camara.bat`

---

## Flujo de Uso Correcto

1. Ir a **Personas** → registrar datos de la persona
2. Subir foto o capturar rostro en vivo
3. Ir a **Control de acceso** → seleccionar cámara
4. Presionar **Verificar acceso**
5. El sistema verifica: **rostro registrado + casco + chaleco**
6. Si todo ok → acceso permitido (BIENVENIDO); si no → acceso denegado

---

## Rendimiento

- Cámaras en hilos compartidos (evita abrir同一 cámara varias veces)
- Video procesado a **640×360**
- Detección cada **12 cuadros**
- JPEG con compresión moderada
- Un solo modelo YOLO por ciclo (casco + chaleco juntos)
