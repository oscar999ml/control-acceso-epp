from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BodyRegions:
    head_xyxy: tuple[int, int, int, int] | None
    torso_xyxy: tuple[int, int, int, int] | None
    has_person: bool


POSE_KPT = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


class BodyDetector:
    def __init__(self, conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO("yolov8n-pose.pt")

    def detect(self, frame: np.ndarray) -> BodyRegions:
        self._lazy_load()
        h, w = frame.shape[:2]
        results = self._model.predict(frame, conf=self.conf_threshold, imgsz=640, verbose=False)

        if len(results) == 0 or len(results[0].keypoints) == 0:
            return BodyRegions(None, None, False)

        kp = results[0].keypoints[0].xy[0].cpu().numpy() if hasattr(results[0].keypoints[0].xy, 'cpu') else results[0].keypoints[0].xy[0]
        confs = results[0].keypoints[0].conf[0].cpu().numpy() if hasattr(results[0].keypoints[0].conf, 'cpu') else results[0].keypoints[0].conf[0]

        def kv(idx: int) -> tuple[float, float, float]:
            return float(kp[idx][0]), float(kp[idx][1]), float(confs[idx]) if confs is not None else 1.0

        nose = kv(POSE_KPT["nose"])
        le = kv(POSE_KPT["left_ear"])
        re = kv(POSE_KPT["right_ear"])
        ls = kv(POSE_KPT["left_shoulder"])
        rs = kv(POSE_KPT["right_shoulder"])
        lh = kv(POSE_KPT["left_hip"])
        rh = kv(POSE_KPT["right_hip"])

        ear_xs = [p[0] for p in [le, re] if p[2] > 0.3]
        ear_ys = [p[1] for p in [le, re] if p[2] > 0.3]
        if nose[2] > 0.3:
            ear_xs.append(nose[0])
            ear_ys.append(nose[1])

        if not ear_xs:
            return BodyRegions(None, None, False)

        cx = sum(ear_xs) / len(ear_xs)
        cy = sum(ear_ys) / len(ear_ys)
        head_w = max(ear_xs) - min(ear_xs)
        head_h = head_w * 0.9

        pad = head_w * 0.2
        hx1 = int(max(0, cx - head_w * 0.6 - pad))
        hy1 = int(max(0, cy - head_h * 0.5 - pad))
        hx2 = int(min(w, cx + head_w * 0.6 + pad))
        hy2 = int(min(h, cy + head_h * 0.6 + pad))

        if hx2 <= hx1 or hy2 <= hy1:
            head_box = None
        else:
            head_box = (hx1, hy1, hx2, hy2)

        shoulder_kpts = [ls, rs]
        hip_kpts = [lh, rh]
        valid_sh = [p for p in shoulder_kpts if p[2] > 0.3]
        valid_hi = [p for p in hip_kpts if p[2] > 0.3]

        if len(valid_sh) < 2 and len(valid_hi) < 2:
            torso_box = None
        else:
            all_x = [p[0] for p in valid_sh + valid_hi]
            all_y = [p[1] for p in valid_sh + valid_hi]
            if not all_x:
                torso_box = None
            else:
                torso_cx = sum(all_x) / len(all_x)
                sh_y = min((p[1] for p in valid_sh), default=cy + 20)
                hi_y = max((p[1] for p in valid_hi), default=sh_y + 50)
                torso_w = (max(all_x) - min(all_x)) * 1.2
                torso_h = (hi_y - sh_y) * 1.1

                tx1 = int(max(0, torso_cx - torso_w / 2))
                ty1 = int(min(h, sh_y))
                tx2 = int(min(w, torso_cx + torso_w / 2))
                ty2 = int(min(h, hi_y + torso_h * 0.3))

                if tx2 <= tx1 or ty2 <= ty1:
                    torso_box = None
                else:
                    torso_box = (tx1, ty1, tx2, ty2)

        return BodyRegions(head_box, torso_box, True)

    def draw(self, frame: np.ndarray, body: BodyRegions) -> np.ndarray:
        if body.head_xyxy:
            x1, y1, x2, y2 = body.head_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(frame, "Cabeza", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
        if body.torso_xyxy:
            x1, y1, x2, y2 = body.torso_xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(frame, "Torso", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)
        return frame
