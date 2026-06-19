from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass

import cv2


def parse_source(source: str):
    source = source.strip()
    if source == "0" or source == "1":
        return int(source)
    if source.isdigit():
        return int(source)
    if source.startswith(("rtsp://", "http://", "https://")):
        return source
    if "." in source:
        return f"http://{source}/video"
    return source


@dataclass
class CameraFrame:
    frame: object | None
    ok: bool
    message: str
    updated_at: float
    detections: list = None
    face: object = None
    summary: dict = None


class CameraStream:
    def __init__(self, source: str, width: int = 640, height: int = 360, processor=None, backend=None):
        self.source = source
        self.parsed_source = parse_source(source)
        self.width = width
        self.height = height
        self.processor = processor
        self.backend = backend
        
        self._frame = None
        self._ok = False
        self._message = "Iniciando..."
        self._updated_at = 0.0
        self._detections = []
        self._face = None
        self._summary = None
        
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
        # Cola para procesamiento (max 1 frame para evitar lag)
        self._proc_queue = queue.Queue(maxsize=1)
        
        # Hilo de captura
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        # Hilo de procesamiento
        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self._cap_thread.start()
        self._proc_thread.start()

    def read(self) -> CameraFrame:
        with self._lock:
            # Solo copiar si es necesario
            frame_copy = self._frame.copy() if self._frame is not None else None
            return CameraFrame(
                frame_copy,
                self._ok,
                self._message,
                self._updated_at,
                self._detections,
                self._face,
                self._summary
            )

    def stop(self) -> None:
        self._stop.set()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            kwargs = {}
            if self.backend is not None:
                kwargs["apiPreference"] = self.backend
            cap = cv2.VideoCapture(self.parsed_source, **kwargs)
            
            if isinstance(self.parsed_source, int):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, 30)

            if not cap.isOpened():
                self._update_status(None, False, "Cámara fuera de línea")
                cap.release()
                time.sleep(5)
                continue

            self._update_status(None, True, "Conectado, esperando video...")
            
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                
                if frame.shape[0] > 0:
                    frame = cv2.resize(frame, (self.width, self.height))
                    self._update_status(frame, True, "Cámara activa")
                    
                    # Intentar poner en cola de procesamiento sin bloquear
                    try:
                        if self.processor:
                            # Si la cola está llena, sacar el viejo y poner el nuevo (siempre procesar lo más reciente)
                            if self._proc_queue.full():
                                try: self._proc_queue.get_nowait()
                                except queue.Empty: pass
                            self._proc_queue.put_nowait(frame.copy())
                    except Exception:
                        pass
                
                # Pequeño respiro para no saturar el bus
                time.sleep(0.01)

            cap.release()
            self._update_status(None, False, "Reconectando...")
            time.sleep(2)

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._proc_queue.get(timeout=1.0)
                if self.processor:
                    detections, face, summary = self.processor(frame)
                    with self._lock:
                        self._detections = detections
                        self._face = face
                        self._summary = summary
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en hilo de procesamiento: {e}")

    def _update_status(self, frame, ok: bool, message: str) -> None:
        with self._lock:
            self._frame = frame
            self._ok = ok
            self._message = message
            self._updated_at = time.time()


class CameraManager:
    def __init__(self):
        self._streams: dict[tuple[int, str], CameraStream] = {}
        self._lock = threading.Lock()
        self.processor = None

    def set_processor(self, processor):
        self.processor = processor

    def get(self, camera_id: int, source: str) -> CameraStream:
        key = (camera_id, source)
        with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                # Detectar si es cámara local (índice numérico) y usar DShow
                import cv2
                backend = None
                src = source.strip()
                if src.isdigit():
                    # Probar DShow primero para OBS Virtual Cam y webcams en Windows
                    try:
                        test = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
                        if test.isOpened():
                            test.release()
                            backend = cv2.CAP_DSHOW
                    except Exception:
                        pass
                stream = CameraStream(source, processor=self.processor, backend=backend)
                self._streams[key] = stream
            return stream

    def remove(self, camera_id: int) -> None:
        with self._lock:
            keys = [key for key in self._streams if key[0] == camera_id]
            for key in keys:
                self._streams.pop(key).stop()

