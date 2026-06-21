from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

import audio_api
from body_api import BodyDetector
from camera_runtime import CameraManager
from database import (
    DATA_DIR,
    add_camera,
    add_worker,
    connect,
    count_by_type,
    delete_camera,
    init_db,
    list_access_events,
    list_cameras,
    list_violations,
    list_workers,
    log_access_event,
    delete_worker,
)
from clasificador_api import ClasificadorEPP
from detector import EPPDetector
from face_api import FaceRecognitionAPI


app = Flask(__name__)

clasificador = ClasificadorEPP()
clasificador.cargar_modelos()

detector = EPPDetector(confidence=0.25, log_cooldown_seconds=20, fast_mode=False, clasificador=clasificador)
body_detector = BodyDetector()
face_api = FaceRecognitionAPI()
camera_manager = CameraManager()
latest_status: dict[int, dict] = {}

last_access_decision: dict[int, dict] = {}


# ── SISTEMA DE VERIFICACIÓN CONTINUA ──────────────────────────

@dataclass
class VerificationDecision:
    allowed: bool
    person_name: str
    reason: str
    stats: dict = None


PREPARE_SECONDS = 0.5         # apenas un momento quieto
COUNTDOWN_SECONDS = 0         # sin cuenta regresiva
VERIFY_FRAMES = 6             # ~0.6s a 10fps
REQUIRED_RATIO = 0.65         # 65% de frames deben cumplir para aprobar
DONE_DISPLAY_SECONDS = 3      # mostrar resultado breve
COOLDOWN_SECONDS = 2          # reinicio rápido


class VerificationSession:
    def __init__(self, camera_id: int):
        self.cam_id = camera_id
        self.state = "idle"
        self.ready_since = 0.0
        self.countdown_start = 0.0
        self.buffer: list[dict] = []
        self.result: VerificationDecision = None
        self.decision_made_at = 0.0
        self.last_frame_gray = None
        self.movement_threshold = 0.25
        self.manual_trigger = False
        self._audio_played = False

    def reset(self):
        self.state = "idle"
        self.ready_since = 0.0
        self.countdown_start = 0.0
        self.buffer = []
        self.result = None
        self.last_frame_gray = None
        self.manual_trigger = False
        self._audio_played = False

    def cancel(self):
        self.reset()

    def update(self, frame, face, summary, now) -> dict:
        out = {"overlay": "", "overlay_color": None, "decision_changed": False, "decision": None}

        if self.state == "idle":
            if self.result and (now - self.decision_made_at < DONE_DISPLAY_SECONDS):
                return self._done_overlay(out)
            if self.result and (now - self.decision_made_at >= DONE_DISPLAY_SECONDS):
                self.result = None
                self.decision_made_at = 0.0
                self._audio_played = False

            if self.manual_trigger:
                self.manual_trigger = False
                self.state = "verifying"
                self.buffer = []
                self._audio_played = False
                audio_api.play_start()
                return out  # sin overlay

            return out  # video limpio

        if self.state == "verifying":
            ok = bool(face and face.ok)
            h_ok = bool(summary and summary.get("helmet_ok", False))
            v_ok = bool(summary and summary.get("vest_ok", False))

            self.buffer.append({
                "identified": ok,
                "helmet_ok": h_ok,
                "vest_ok": v_ok,
                "label": face.label if face else "Desconocido",
            })

            if len(self.buffer) >= VERIFY_FRAMES:
                self._decide()
                self.state = "done"
                self.decision_made_at = now
                out["decision_changed"] = True
                out["decision"] = self.result
                if not self._audio_played:
                    self._audio_played = True
                    if self.result.allowed:
                        audio_api.play_ok()
                    else:
                        audio_api.play_fail()
                return self._done_overlay(out)

            return out  # sin overlay durante verificación

        if self.state == "done":
            if now - self.decision_made_at < DONE_DISPLAY_SECONDS:
                return self._done_overlay(out)
            self._auto_return()
            return out

        return out

    def _done_overlay(self, out):
        if self.result and self.result.allowed:
            out["overlay"] = f"✔ BIENVENIDO {self.result.person_name.upper()}"
            out["overlay_color"] = (0, 200, 0)
        elif self.result:
            out["overlay"] = f"✖ {self.result.reason}"
            out["overlay_color"] = (0, 0, 200)
        return out

    def _auto_return(self):
        self.state = "idle"
        self.ready_since = 0.0
        self.buffer = []
        self.result = None
        self.last_frame_gray = None

    def _check_readiness(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.last_frame_gray is not None:
            diff = cv2.absdiff(gray, self.last_frame_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            change = np.count_nonzero(thresh) / thresh.size
            moving = change > self.movement_threshold
        else:
            moving = False
        self.last_frame_gray = gray
        return not moving

    def _decide(self):
        if not self.buffer:
            self.result = VerificationDecision(False, "—", "Sin datos")
            return
        total = len(self.buffer)
        id_ok = sum(1 for f in self.buffer if f["identified"]) / total
        h_ok = sum(1 for f in self.buffer if f["helmet_ok"]) / total
        v_ok = sum(1 for f in self.buffer if f["vest_ok"]) / total
        labels = [f["label"] for f in self.buffer]
        main_label = Counter(labels).most_common(1)[0][0]

        if id_ok >= REQUIRED_RATIO and h_ok >= REQUIRED_RATIO and v_ok >= REQUIRED_RATIO:
            self.result = VerificationDecision(True, main_label, "Acceso permitido")
        else:
            reasons = []
            if id_ok < REQUIRED_RATIO:
                reasons.append("No registrado")
            if h_ok < REQUIRED_RATIO:
                reasons.append("Falta casco")
            if v_ok < REQUIRED_RATIO:
                reasons.append("Falta chaleco")
            self.result = VerificationDecision(False, main_label, " | ".join(reasons))

    def start_manual(self):
        if self.state == "idle" and not self.manual_trigger and self.result is None:
            self.manual_trigger = True
            self.ready_since = 0.0
            return True
        return False

    def get_public_state(self) -> dict:
        d = {
            "state": self.state,
            "progress": round(len(self.buffer) / max(VERIFY_FRAMES, 1), 2),
        }
        if self.result:
            d["allowed"] = self.result.allowed
            d["person"] = self.result.person_name
            d["reason"] = self.result.reason
        return d


# sesiones de verificación por cámara
sessions: dict[int, VerificationSession] = {}


def get_session(camera_id: int) -> VerificationSession:
    if camera_id not in sessions:
        sessions[camera_id] = VerificationSession(camera_id)
    return sessions[camera_id]


# ── PROCESADOR GLOBAL ─────────────────────────────────────────

def global_processor(frame):
    try:
        body = body_detector.detect(frame)
        if body.has_person:
            detections = detector.detect_with_regions(frame, body.head_xyxy, body.torso_xyxy)
        else:
            detections = detector.detect(frame)
        face = face_api.recognize(frame)
        summary = detector.summarize(detections)
        return detections, face, summary
    except Exception as e:
        print(f"Error en global_processor: {e}")
        return [], None, {"helmet_ok": True, "vest_ok": True, "text": "Error", "active": False}


camera_manager.set_processor(global_processor)

# --- CACHÉ DE CONFIGURACIÓN ---
_cached_cameras = []
_last_camera_update = 0.0


def get_active_cameras():
    global _cached_cameras, _last_camera_update
    now = time.time()
    if now - _last_camera_update > 5.0:
        _cached_cameras = list_cameras()
        _last_camera_update = now
    return _cached_cameras


def make_status_frame(message: str):
    frame = np.full((360, 640, 3), 255, dtype=np.uint8)
    cv2.putText(frame, message, (24, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 180), 2)
    return frame


def build_status_payload():
    cameras = []
    for cam in get_active_cameras():
        status = latest_status.get(cam["id"])
        summary = status["summary"] if status else None
        face = status["face"] if status else None
        session = get_session(cam["id"])
        
        is_identified = bool(face and face.ok)
        helmet_ok = bool(summary and summary.get("helmet_ok", False))
        vest_ok = bool(summary and summary.get("vest_ok", False))
        
        access_allowed = False
        if is_identified and helmet_ok and vest_ok:
            access_allowed = True
        
        status_text = "Área despejada"
        if summary and summary.get("active"):
            status_text = summary["text"]
            if not is_identified:
                status_text = "PERSONA NO REGISTRADA"
            elif not helmet_ok and not vest_ok:
                status_text = "FALTA CASCO Y CHALECO"
            elif not helmet_ok:
                status_text = "FALTA CASCO DE SEGURIDAD"
            elif not vest_ok:
                status_text = "FALTA CHALECO REFLECTANTE"

        # ── estado de verificación ──
        ver = session.get_public_state()
        if session.state == "done" and ver.get("allowed"):
            status_text = f"✅ BIENVENIDO {ver['person']}"
        elif session.state == "done" and not ver.get("allowed"):
            status_text = f"❌ {ver['reason']}"

        item = {
            "id": cam["id"],
            "nombre": cam["name"],
            "tiene_lectura": status is not None,
            "casco_correcto": helmet_ok,
            "chaleco_correcto": vest_ok,
            "persona": face.label if face else "Desconocido",
            "persona_registrada": is_identified,
            "texto": status_text,
            "alerta": not access_allowed if (summary and summary.get("active")) else False,
            "acceso_permitido": access_allowed,
            "es_punto_acceso": bool(cam["is_access_control"]) if "is_access_control" in cam.keys() else False,
            "verificacion": ver,
        }
        cameras.append(item)
    return {"camaras": cameras}


@app.route("/data/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(DATA_DIR / "snapshots", filename)


@app.route("/fotos")
def evidence():
    try:
        snapshot_dir = DATA_DIR / "snapshots"
        if not snapshot_dir.exists():
            snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        photos = []
        for p in snapshot_dir.glob("*.jpg"):
            try:
                photos.append({
                    "name": p.name,
                    "url": url_for('serve_snapshot', filename=p.name),
                    "mtime": p.stat().st_mtime
                })
            except Exception:
                continue
        
        photos.sort(key=lambda x: x["mtime"], reverse=True)
        return render_template("evidence.html", photos=photos[:50])
    except Exception as e:
        print(f"Error en fotos: {e}")
        return f"Error en galería: {e}", 500


@app.route("/api/verificacion/iniciar/<int:camera_id>", methods=["POST"])
def api_start_verification(camera_id: int):
    session = get_session(camera_id)
    ok = session.start_manual()
    return jsonify({"ok": ok, "mensaje": "Verificación iniciada" if ok else "Ya hay una verificación activa"})


@app.route("/api/verificacion/detener/<int:camera_id>", methods=["POST"])
def api_stop_verification(camera_id: int):
    session = get_session(camera_id)
    session.cancel()
    return jsonify({"ok": True, "mensaje": "Verificación detenida"})


@app.route("/api/verificacion/estado/<int:camera_id>")
def api_verification_status(camera_id: int):
    session = get_session(camera_id)
    return jsonify(session.get_public_state())


# ── API PARA ESP8266 ──────────────────────────────────────────

_DEFAULT_CAMERA_ID: int | None = None

def _get_esp_camera_id() -> int:
    global _DEFAULT_CAMERA_ID
    if _DEFAULT_CAMERA_ID is None:
        cams = get_active_cameras()
        if cams:
            _DEFAULT_CAMERA_ID = cams[0]["id"]
        else:
            _DEFAULT_CAMERA_ID = 0
    return _DEFAULT_CAMERA_ID


@app.route("/api/esp/estado")
def api_esp_estado():
    cid = _get_esp_camera_id()
    st = latest_status.get(cid, {})
    s = st.get("summary") or {}
    helmet_ok = bool(s.get("helmet_ok", False))
    vest_ok = bool(s.get("vest_ok", False))
    active = bool(s.get("active", False))
    session = get_session(cid)
    sv = session.get_public_state()
    return jsonify({
        "helmet_ok": helmet_ok,
        "vest_ok": vest_ok,
        "both_ok": helmet_ok and vest_ok,
        "active": active,
        "ver_state": sv.get("state", "idle"),
        "ver_allowed": sv.get("allowed") if sv.get("state") == "done" else None,
    })


@app.route("/api/esp/verificar", methods=["POST"])
def api_esp_verificar():
    cid = _get_esp_camera_id()
    session = get_session(cid)
    ok = session.start_manual()
    return jsonify({"ok": ok})


@app.route("/access", methods=["GET", "POST"])
@app.route("/control-acceso", methods=["GET", "POST"])
def access_control():
    try:
        if request.method == "POST":
            camera_id = int(request.form.get("camera_id", "0") or "0")
            with connect() as conn:
                conn.execute("UPDATE cameras SET is_access_control = 0")
                if camera_id > 0:
                    conn.execute("UPDATE cameras SET is_access_control = 1 WHERE id = ?", (camera_id,))
            return redirect(url_for("access_control"))

        cameras = list_cameras(active_only=True)
        events = list_access_events(50)
        return render_template("access.html", cameras=cameras, events=events)
    except Exception as e:
        print(f"Error en configuración de acceso: {e}")
        return f"Error en configuración: {e}", 500


@app.route("/")
def index():
    cameras = get_active_cameras()
    return render_template("index.html", cameras=cameras, latest_status=latest_status)


@app.route("/admin")
@app.route("/configuracion")
def admin():
    cameras = get_active_cameras()
    violations = list_violations(30)
    totals = count_by_type()
    return render_template("admin.html", cameras=cameras, violations=violations, totals=totals)


@app.route("/video/<int:camera_id>")
def video(camera_id: int):
    camera = next((cam for cam in get_active_cameras() if cam["id"] == camera_id), None)
    if camera is None:
        return "Cámara no encontrada", 404
    return Response(
        frame_stream(camera["id"], camera["name"], camera["source"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def frame_stream(camera_id: int, camera_name: str, source: str):
    stream = camera_manager.get(camera_id, source)
    session = get_session(camera_id)
    while True:
        camera_frame = stream.read()
        if not camera_frame.ok or camera_frame.frame is None:
            frame = make_status_frame(camera_frame.message)
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(0.5)
            continue

        frame = camera_frame.frame
        detections = camera_frame.detections or []
        face = camera_frame.face
        summary = camera_frame.summary

        latest_status[camera_id] = {
            "camera_name": camera_name,
            "detections": detections,
            "summary": summary or {"helmet_ok": True, "vest_ok": True, "text": "Iniciando...", "active": False},
            "face": face,
        }

        # ── Verificación continua ──
        now = time.time()
        ver = session.update(frame.copy(), face, summary, now)

        # ── overlay en el video ──
        h, w = frame.shape[:2]
        overlay_color = ver.get("overlay_color")
        overlay_text = ver.get("overlay", "")

        if overlay_text:
            # barra semitransparente arriba
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            if overlay_color:
                color = (int(overlay_color[0]), int(overlay_color[1]), int(overlay_color[2]))
                cv2.putText(frame, overlay_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 3)

        # ── resultado grande al centro ──
        if ver.get("decision_changed") and ver.get("decision"):
            dec = ver["decision"]
            big_text = f"BIENVENIDO {dec.person_name.upper()}" if dec.allowed else f"DENEGADO"
            big_color = (0, 200, 0) if dec.allowed else (0, 0, 200)
            # fondo semitransparente
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (w//2 - 250, h//2 - 90), (w//2 + 250, h//2 + 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, big_text, (w//2 - 220, h//2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, big_color, 4)

        # ── dibujado existente ──
        person_label = face.label if face else "Desconocido"
        detector.draw_and_log(frame, detections, camera_id, camera_name, person_label)
        face_api.draw_face(frame)
        if face and face.ok:
            cv2.putText(frame, f"ID: {face.label}", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.04)


# ── Stream solo para registro facial (sin EPP) ──

def frame_stream_face(camera_id: int, source: str):
    """Igual que frame_stream pero SOLO dibuja el rostro, sin YOLO ni verify."""
    stream = camera_manager.get(camera_id, source)
    while True:
        cf = stream.read()
        if not cf.ok or cf.frame is None:
            f = make_status_frame(cf.message)
            ok, jpeg = cv2.imencode(".jpg", f)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            time.sleep(0.5)
            continue

        frame = cf.frame.copy()
        # Solo dibujar el rostro
        face_api.draw_face(frame)
        h, w = frame.shape[:2]
        cv2.putText(frame, "REGISTRO FACIAL", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.04)


@app.route("/video-face/<int:camera_id>")
def video_face(camera_id: int):
    camera = next((cam for cam in get_active_cameras() if cam["id"] == camera_id), None)
    if camera is None:
        return "Cámara no encontrada", 404
    return Response(
        frame_stream_face(camera["id"], camera["source"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "app": "Sistema EPP", "version": "1.0.0"})

@app.route("/estado.json")
def status_json():
    return jsonify(build_status_payload())


@app.post("/cameras")
def create_camera():
    name = request.form.get("name", "").strip()
    source = request.form.get("source", "").strip()
    if name and source:
        add_camera(name, source)
    return redirect(url_for("admin"))


@app.post("/cameras/<int:camera_id>/delete")
def remove_camera(camera_id: int):
    delete_camera(camera_id)
    camera_manager.remove(camera_id)
    return redirect(url_for("admin"))


@app.route("/workers", methods=["GET", "POST"])
@app.route("/personas", methods=["GET", "POST"])
def workers():
    if request.method == "POST":
        add_worker(
            None,
            request.form.get("full_name", "").strip(),
            request.form.get("area", "").strip(),
            request.form.get("person_type", "Personal")
        )
        return redirect(url_for("workers"))
    return render_template("workers.html", workers=list_workers(), cameras=get_active_cameras())


@app.post("/workers/<int:worker_id>/delete")
def remove_worker(worker_id: int):
    delete_worker(worker_id)
    return redirect(url_for("workers"))


@app.post("/api/rostros/subir")
def api_upload_face():
    worker_id = int(request.form.get("worker_id", "0") or "0")
    file = request.files.get("photo")
    if not file:
        return jsonify({"correcto": False, "mensaje": "Sin imagen"}), 400
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"correcto": False, "mensaje": "Imagen inválida"}), 400
    result = face_api.register_face(worker_id, frame)
    return jsonify({"correcto": result.ok, "mensaje": result.message})


@app.post("/api/rostros/registrar")
def api_register_face():
    worker_id = int(request.form.get("worker_id", "0") or "0")
    camera_id = int(request.form.get("camera_id", "0") or "0")
    camera = next((cam for cam in get_active_cameras() if cam["id"] == camera_id), None)
    if not camera:
        return jsonify({"correcto": False, "mensaje": "Cámara no encontrada"}), 404

    stream = camera_manager.get(camera["id"], camera["source"])

    # Esperar hasta obtener frames válidos (hasta 5 segundos)
    best_frame = None
    deadline = time.time() + 5.0
    capture_count = 0
    while time.time() < deadline:
        result = stream.read()
        if result.ok and result.frame is not None:
            capture_count += 1
            # Verificar si hay un rostro detectable
            gray = cv2.cvtColor(result.frame, cv2.COLOR_BGR2GRAY)
            test_faces = face_api._detect_faces(result.frame)
            if len(test_faces) > 0:
                # Usar frame con rostro detectado
                best_frame = result.frame
                break
            # Guardar el primer frame válido por si no aparece rostro
            if best_frame is None:
                best_frame = result.frame
        time.sleep(0.1)

    if best_frame is None:
        return jsonify({"correcto": False, "mensaje": "No se pudo capturar imagen de la cámara"}), 400

    result = face_api.register_face(worker_id, best_frame)
    return jsonify({"correcto": result.ok, "mensaje": result.message})


# ── Captura para entrenamiento de modelos ──

TRAIN_DIR = DATA_DIR / "entrenamiento"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/entrenamiento/capturar")
def api_capture_train():
    camera_id = int(request.form.get("camera_id", "0") or "0")
    angle = request.form.get("angle", "frente").strip()
    camera = next((cam for cam in get_active_cameras() if cam["id"] == camera_id), None)
    if not camera:
        return jsonify({"ok": False, "mensaje": "Cámara no encontrada"}), 404

    stream = camera_manager.get(camera["id"], camera["source"])
    best_frame = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        r = stream.read()
        if r.ok and r.frame is not None:
            best_frame = r.frame
            break
        time.sleep(0.1)

    if best_frame is None:
        return jsonify({"ok": False, "mensaje": "No se pudo capturar"}), 400

    img_dir = TRAIN_DIR / "chaleco" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    count = len([f for f in img_dir.glob("captura_*.png")]) + 1
    fname = f"captura_{count:02d}_{angle}.png"
    cv2.imwrite(str(img_dir / fname), best_frame)
    return jsonify({"ok": True, "archivo": fname, "mensaje": f"Capturado: {fname}"})


@app.route("/entrenamiento")
def entrenamiento():
    img_dir = TRAIN_DIR / "chaleco" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    fotos = sorted(img_dir.glob("captura_*.png"))
    return render_template("entrenamiento.html", cameras=get_active_cameras(), fotos=fotos)


@app.route("/entrenamiento/fotos/<path:filename>")
def entrenamiento_fotos(filename):
    from flask import send_from_directory
    return send_from_directory(str(TRAIN_DIR / "chaleco" / "images"), filename)


import subprocess
import sys

TRAIN_SCRIPT = Path(__file__).parent / "train_chaleco.py"

@app.post("/api/entrenamiento/entrenar")
def api_train_model():
    try:
        # Auto-anotar las nuevas capturas
        _annotate_captures()
        # Ejecutar entrenamiento
        result = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(Path(__file__).parent), capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            return jsonify({"ok": False, "mensaje": f"Error: {result.stderr[-200:]}"}), 500
        return jsonify({"ok": True, "mensaje": "Modelo entrenado correctamente"})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "mensaje": "Entrenamiento tardó demasiado"}), 500
    except Exception as e:
        return jsonify({"ok": False, "mensaje": str(e)}), 500


def _annotate_captures():
    """Genera labels YOLO para las capturas usando segmentación por color."""
    import cv2
    import numpy as np

    img_dir = TRAIN_DIR / "chaleco" / "images"
    lbl_dir = TRAIN_DIR / "chaleco" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for fname in sorted(img_dir.glob("captura_*.png")):
        label_path = lbl_dir / f"{fname.stem}.txt"
        if label_path.exists():
            continue

        img = cv2.imread(str(fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Rojo/naranja
        lr1 = np.array([0, 40, 40])
        ur1 = np.array([15, 255, 255])
        lr2 = np.array([160, 40, 40])
        ur2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lr1, ur1) | cv2.inRange(hsv, lr2, ur2)

        # Gris/plomo
        lg = np.array([0, 0, 80])
        ug = np.array([180, 40, 200])
        mask_gray = cv2.inRange(hsv, lg, ug)

        # Amarillo/naranja brillante
        ly = np.array([15, 80, 80])
        uy = np.array([45, 255, 255])
        mask_yellow = cv2.inRange(hsv, ly, uy)

        mask = mask_red | mask_gray | mask_yellow
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 0.01 * h * w:
            continue

        x, y, bw, bh = cv2.boundingRect(largest)
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        with open(str(label_path), "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")


@app.route("/violations")
@app.route("/infracciones")
def violations():
    return render_template("violations.html", violations=list_violations(200))


import pandas as pd
import io

@app.route("/reporte/excel")
def download_excel():
    try:
        events = list_access_events(1000)
        df = pd.DataFrame(events)
        
        # Traducir columnas para el reporte
        df = df.rename(columns={
            "created_at": "Fecha y Hora",
            "camera_name": "Cámara",
            "person_label": "Persona Identificada",
            "person_type": "Categoría",
            "helmet_ok": "Casco (1=Sí)",
            "vest_ok": "Chaleco (1=Sí)",
            "decision": "Resultado",
            "reason": "Motivo"
        })
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Accesos EPP')
        
        output.seek(0)
        return Response(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment;filename=Reporte_Seguridad_EPP.xlsx"}
        )
    except Exception as e:
        print(f"Error Excel: {e}")
        return str(e), 500


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, threaded=True)
