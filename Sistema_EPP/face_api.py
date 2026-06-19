from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from database import ROOT, get_worker, list_workers, save_worker_face


FACE_DIR = ROOT / "data" / "rostros"


@dataclass
class FaceMatch:
    worker_id: int | None
    label: str
    score: float
    ok: bool
    message: str


class FaceRecognitionAPI:

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(320, 320))
        self._worker_cache: list[dict] = []
        self._last_cache_update = 0.0

    def _update_cache(self):
        now = time.time()
        if now - self._last_cache_update < 10.0 and self._worker_cache:
            return

        workers = list_workers()
        new_cache = []
        for w in workers:
            encoded = w["face_template"] if "face_template" in w.keys() else None
            if encoded:
                emb = self._decode_template(encoded)
                if emb is not None and len(emb) == 512:
                    new_cache.append({
                        "id": w["id"],
                        "full_name": w["full_name"],
                        "embedding": emb,
                    })
        self._worker_cache = new_cache
        self._last_cache_update = now

    def register_face(self, worker_id: int, frame) -> FaceMatch:
        worker = get_worker(worker_id)
        if worker is None:
            return FaceMatch(None, "Persona no registrada", 0.0, False,
                             "Primero registra los datos de la persona")

        faces = self._app.get(frame)
        if len(faces) == 0:
            return FaceMatch(worker_id, worker["full_name"], 0.0, False,
                             "No se detectó un rostro claro")

        face = faces[0]
        embedding = face.embedding.copy()

        FACE_DIR.mkdir(parents=True, exist_ok=True)
        image_path = FACE_DIR / f"persona_{worker_id}.jpg"
        cv2.imwrite(str(image_path), frame)

        save_worker_face(worker_id, self._encode_template(embedding), str(image_path))
        self._last_cache_update = 0.0

        return FaceMatch(worker_id, worker["full_name"], 1.0, True,
                         "Rostro registrado correctamente")

    def recognize(self, frame) -> FaceMatch:
        faces = self._app.get(frame)
        if len(faces) == 0:
            return FaceMatch(None, "No identificado", 0.0, False,
                             "No se detectó rostro")

        embedding = faces[0].embedding.copy()
        self._update_cache()

        if not self._worker_cache:
            return FaceMatch(None, "No identificado", 0.0, False,
                             "No existen rostros registrados")

        best_id = None
        best_label = "No identificado"
        best_score = -1.0

        for w in self._worker_cache:
            score = float(np.dot(embedding, w["embedding"]))
            if score > best_score:
                best_score = score
                best_id = w["id"]
                best_label = w["full_name"]

        ok = best_score >= self.threshold
        message = "Persona reconocida" if ok else "Rostro no coincide"
        return FaceMatch(best_id if ok else None,
                         best_label if ok else "No identificado",
                         best_score, ok, message)

    def draw_face(self, frame):
        faces = self._app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[:4]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(frame, "Rostro", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
            lmks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
            if lmks is not None:
                for pt in lmks[0::20]:
                    cx, cy = int(pt[0]), int(pt[1])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
        return frame

    def _detect_faces(self, frame):
        faces = self._app.get(frame)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[:4]
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                result.append((x1, y1, w, h))
        return result

    def is_facing_camera(self, frame) -> tuple[bool, float]:
        faces = self._app.get(frame)
        if len(faces) == 0:
            return False, 0.0

        face = faces[0]
        lmks = face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None
        if lmks is None or len(lmks) < 10:
            return False, 0.0

        ojo_izq = lmks[33]
        ojo_der = lmks[68]
        nariz = lmks[54]
        boca = lmks[88]

        dy_eyes = ojo_der[1] - ojo_izq[1]
        dx_eyes = ojo_der[0] - ojo_izq[0]
        angle = abs(np.degrees(np.arctan2(dy_eyes, dx_eyes)))
        if angle > 20:
            return False, max(0.0, 1.0 - angle / 45.0)

        left_ratio = (nariz[0] - ojo_izq[0]) / max(dx_eyes, 1)
        right_ratio = (ojo_der[0] - nariz[0]) / max(dx_eyes, 1)
        symmetry = 1.0 - abs(left_ratio - right_ratio) / (left_ratio + right_ratio + 1e-6)

        return symmetry >= 0.5, symmetry

    def has_glasses(self, frame) -> tuple[bool, float]:
        """Detecta posibles lentes/obstrucción en región ocular usando bordes."""
        faces = self._app.get(frame)
        if len(faces) == 0:
            return False, 0.0

        face = faces[0]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[:4]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0

        h, w = roi.shape
        eye_region = roi[int(h*0.2):int(h*0.55), int(w*0.05):int(w*0.95)]
        if eye_region.size == 0:
            return False, 0.0

        edges = cv2.Canny(eye_region, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=8)

        if lines is None:
            return False, 0.0

        h_lines = 0
        total = len(lines)
        for line in lines:
            x1l, y1l, x2l, y2l = line[0]
            angle = abs(np.degrees(np.arctan2(y2l - y1l, x2l - x1l)))
            if angle < 20 or angle > 160:
                h_lines += 1

        score = h_lines / max(total, 1)
        return score > 0.35, score

    @staticmethod
    def _encode_template(embedding: np.ndarray) -> str:
        return base64.b64encode(embedding.astype("float32").tobytes()).decode("ascii")

    @staticmethod
    def _decode_template(encoded: str) -> np.ndarray | None:
        try:
            arr = np.frombuffer(base64.b64decode(encoded.encode("ascii")), dtype="float32")
            return arr if len(arr) > 0 else None
        except Exception:
            return None

    # Mantener compatibilidad con el formato de template anterior
    _template = _encode_template
    _similarity = lambda self, a, b: float(np.dot(a, b))
