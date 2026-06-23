# AGENTS.md — Estado del Proyecto

## Repositorio
- **URL:** https://github.com/oscar999ml/control-acceso-epp
- **Rama:** `master` (por defecto)
- **Commits:** 16, organizados por función
- **Descripción:** *Sistema de control de acceso con deteccion de EPP (casco y chaleco), reconocimiento facial y control ESP8266*

## Lo que ya está subido
- Código fuente completo (Python, HTML, CSS, JS)
- Pesos YOLO (`Casco.pt`, `Chaleco.pt`, `yolov8n-pose.pt`, `casco_clf.pt`)
- Dataset de entrenamiento completo (imágenes + labels + checkpoints)
- Firmware ESP8266 (`esp8266_control.ino`)
- README.md con documentación completa
- `.gitignore` bien configurado

## Para que alguien clone y ejecute
```
git clone https://github.com/oscar999ml/control-acceso-epp.git
cd control-acceso-epp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd Sistema_EPP
python app.py
```
Sin necesidad de entrenar nada.

## Pendiente / Mejora posible
- Subir `data/rostros/` con las fotos de personas registradas (solo si se quiere, son datos personales)
- Crear `setup.bat` que automatice venv + pip install
- El calendario de GitHub tardará ~24h en mostrar los 16 commits
- Renombrar rama `master` a `main` si se prefiere el estándar moderno
- Agregar soporte para WebSocket en tiempo real
