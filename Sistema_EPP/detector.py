from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from database import ROOT, log_violation


MODEL_DIR = ROOT.parent
CASCO_MODEL = MODEL_DIR / "Casco.pt"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"

DETECTOR_VERSION = "2.0.0"

SPANISH_LABELS = {
    "helmet": "Casco",
    "no-helmet": "Sin Casco",
    "vest": "Chaleco",
    "no-vest": "Sin Chaleco",
}


@dataclass
class Detection:
    label: str
    display_label: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def x1(self) -> int:
        return self.xyxy[0]

    @property
    def y1(self) -> int:
        return self.xyxy[1]

    @property
    def x2(self) -> int:
        return self.xyxy[2]

    @property
    def y2(self) -> int:
        return self.xyxy[3]


class EPPDetector:
    def __init__(self, confidence: float = 0.5, log_cooldown_seconds: int = 30, fast_mode: bool = False, clasificador=None):
        self.confidence = confidence
        self.log_cooldown_seconds = log_cooldown_seconds
        self.fast_mode = fast_mode
        self.clasificador = clasificador
        self._helmet_model = None
        self._last_logged: dict[tuple[int | None, str], float] = {}

    def _load_models(self) -> None:
        if self._helmet_model is not None:
            return
        from ultralytics import YOLO
        self._helmet_model = YOLO(str(CASCO_MODEL))

    @staticmethod
    def _helmet_color_ok(roi_bgr: np.ndarray, min_pct: float = 15.0) -> bool:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        masks = []
        # Azul: H 100–130
        masks.append(cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255])))
        # Amarillo: H 20–35
        masks.append(cv2.inRange(hsv, np.array([20, 50, 40]), np.array([35, 255, 255])))
        # Rojo: H 0–10 y 160–180
        masks.append(cv2.inRange(hsv, np.array([0, 60, 40]), np.array([10, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([160, 60, 40]), np.array([180, 255, 255])))
        # Blanco: baja saturación, alto brillo
        masks.append(cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 35, 255])))
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for m in masks:
            combined |= m
        pct = float(np.sum(combined > 0)) / max(1, hsv.shape[0] * hsv.shape[1]) * 100.0
        return pct >= min_pct

    @staticmethod
    def _vest_color_pct(roi_hsv: np.ndarray) -> float:
        # Chaleco plomo (gris): baja saturación, amplio valor
        grey_lower = np.array([0, 3, 40])
        grey_upper = np.array([180, 35, 200])
        # Chaleco rojo (burgundy): H 150-180, saturación media-alta
        red_lower = np.array([150, 70, 60])
        red_upper = np.array([180, 255, 255])

        mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
        mask |= cv2.inRange(roi_hsv, grey_lower, grey_upper)
        mask |= cv2.inRange(roi_hsv, red_lower, red_upper)

        non_dark = roi_hsv[:, :, 2] > 30
        valid = cv2.bitwise_and(mask, mask, mask=non_dark.astype(np.uint8))
        total = float(np.sum(valid > 0))
        denom = max(1, np.sum(non_dark))
        return total / denom * 100.0

    def _detect_helmet_in_roi(self, frame, roi_box: tuple[int, int, int, int]) -> list[Detection]:
        x1, y1, x2, y2 = roi_box

        # YOLO sobre frame completo (como ZIP original), filtrar por región
        imgsz = 640 if not self.fast_mode else 416
        results = self._helmet_model.predict(frame, conf=self.confidence, imgsz=imgsz, verbose=False)
        dets = self._extract(results, {"helmet", "no-helmet"})

        roi_dets = []
        for d in dets:
            ix1, iy1 = max(d.x1, x1), max(d.y1, y1)
            ix2, iy2 = min(d.x2, x2), min(d.y2, y2)
            if ix1 < ix2 and iy1 < iy2:
                overlap = (ix2 - ix1) * (iy2 - iy1)
                det_area = (d.x2 - d.x1) * (d.y2 - d.y1)
                if overlap / max(det_area, 1) > 0.2:
                    # Validar color del casco si YOLO dice "helmet"
                    if d.label == "helmet":
                        crop = frame[d.y1:d.y2, d.x1:d.x2]
                        if crop.size > 0 and not self._helmet_color_ok(crop):
                            d.label = "no-helmet"
                            d.display_label = SPANISH_LABELS["no-helmet"]
                    dw = int((d.x2 - d.x1) * 0.4)
                    dh = int((d.y2 - d.y1) * 0.4)
                    d.xyxy = (
                        max(0, d.x1 - dw),
                        max(0, d.y1 - dh),
                        d.x2 + dw,
                        d.y2 + dh,
                    )
                    roi_dets.append(d)

        return roi_dets

    def _detect_vest_in_roi(self, frame, roi_box: tuple[int, int, int, int]) -> list[Detection]:
        x1, y1, x2, y2 = roi_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        h_crop, w_crop = crop.shape[:2]
        if h_crop < 20 or w_crop < 20:
            return []

        # Use classifier if available
        if self.clasificador and self.clasificador.modelo_chaleco is not None:
            label, conf = self.clasificador.predecir_chaleco(crop)
            return [Detection(
                label=label,
                display_label=SPANISH_LABELS.get(label, label),
                confidence=round(conf, 3),
                xyxy=roi_box,
            )]

        # Fallback to color
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = self._vest_color_pct(hsv)

        if total > 10.0:
            label, conf = "vest", min(0.95, total / 80.0)
        else:
            label, conf = "no-vest", max(0.5, 1.0 - total / 30.0)

        return [Detection(
            label=label,
            display_label=SPANISH_LABELS.get(label, label),
            confidence=round(conf, 3),
            xyxy=(x1, y1, x2, y2),
        )]

    def detect(self, frame) -> list[Detection]:
        self._load_models()
        imgsz = 640 if not self.fast_mode else 416
        results = self._helmet_model.predict(frame, conf=self.confidence, imgsz=imgsz, verbose=False)
        return self._extract(results, {"helmet", "no-helmet"})

    def detect_with_regions(self, frame, head_box: tuple | None, torso_box: tuple | None) -> list[Detection]:
        self._load_models()
        detections: list[Detection] = []

        if head_box:
            # Desplazar la región hacia arriba para detectar el casco (no la cara)
            x1, y1, x2, y2 = head_box
            h_h = y2 - y1
            helmet_y1 = max(0, y1 - int(h_h * 0.35))
            helmet_y2 = y1 + int(h_h * 0.5)
            helmet_box = (x1, helmet_y1, x2, helmet_y2)

            helmet_dets = self._detect_helmet_in_roi(frame, helmet_box)
            if not helmet_dets:
                helmet_dets.append(Detection(
                    label="no-helmet",
                    display_label=SPANISH_LABELS["no-helmet"],
                    confidence=0.6,
                    xyxy=helmet_box,
                ))
            detections.extend(helmet_dets)

        if not self.fast_mode and torso_box:
            detections.extend(self._detect_vest_in_roi(frame, torso_box))

        return detections

    def summarize(self, detections: list[Detection]) -> dict[str, bool | str]:
        labels = {det.label for det in detections}

        helmet_missing = "no-helmet" in labels
        vest_missing = "no-vest" in labels

        helmet_ok = not helmet_missing
        vest_ok = not vest_missing

        for det in detections:
            if det.label in SPANISH_LABELS:
                det.display_label = SPANISH_LABELS[det.label]

        if not labels:
            return {
                "helmet_ok": True,
                "vest_ok": True,
                "text": "Área despejada",
                "active": False,
            }

        return {
            "helmet_ok": helmet_ok,
            "vest_ok": vest_ok,
            "text": self.status_text(helmet_ok, vest_ok),
            "active": True,
        }

    @staticmethod
    def status_text(helmet_ok: bool, vest_ok: bool) -> str:
        missing = []
        if not helmet_ok:
            missing.append("casco")
        if not vest_ok:
            missing.append("chaleco")
        return "EPP completo" if not missing else "Falta " + " y ".join(missing)

    def draw_and_log(self, frame, detections: list[Detection], camera_id: int | None, camera_name: str, person_label: str = "Desconocido"):
        for det in detections:
            is_violation = det.label in {"no-helmet", "no-vest"}
            color = (0, 0, 255) if is_violation else (0, 180, 0)
            x1, y1, x2, y2 = det.xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{det.display_label} {det.confidence:.2f}"
            cv2.putText(frame, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_violation:
                self._log_if_needed(frame, det, camera_id, camera_name, person_label)

        return frame

    @staticmethod
    def _extract(results, allowed_labels: set[str]) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, cls_id))
                if label not in allowed_labels:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        label=label,
                        display_label=SPANISH_LABELS.get(label, label),
                        confidence=conf,
                        xyxy=(x1, y1, x2, y2),
                    )
                )
        return detections

    def _log_if_needed(self, frame, detection: Detection, camera_id: int | None, camera_name: str, person_label: str) -> None:
        key = (camera_id, detection.label)
        now = monotonic()
        if now - self._last_logged.get(key, 0) < self.log_cooldown_seconds:
            return

        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        filename_label = detection.display_label.lower().replace(" ", "_")
        safe_person = person_label.replace(" ", "_")
        filename = f"INFRACTOR_{safe_person}_{filename_label}_{int(now * 1000)}.jpg"
        path = SNAPSHOT_DIR / filename
        cv2.imwrite(str(path), frame)

        log_violation(
            camera_id=camera_id,
            camera_name=camera_name,
            violation_type=detection.display_label,
            confidence=detection.confidence,
            snapshot_path=str(path),
            worker_label=person_label,
        )
        self._last_logged[key] = now
