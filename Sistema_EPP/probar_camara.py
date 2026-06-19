from __future__ import annotations

import argparse

import cv2

from camera_runtime import parse_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba si una camara puede abrirse con OpenCV.")
    parser.add_argument("source", help="Ejemplo: 0, 1, 192.168.1.50:8080, http://IP:PUERTO/video o rtsp://...")
    args = parser.parse_args()

    source = parse_source(args.source)
    print(f"Probando camara: {source}")

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("NO se pudo abrir la camara.")
        print("Si usas IP Webcam en celular, prueba con: http://IP:PUERTO/video")
        print("Ejemplo: http://192.168.1.50:8080/video")
        return

    print("Camara OK.")
    print(f"Resolucion detectada: {frame.shape[1]}x{frame.shape[0]}")


if __name__ == "__main__":
    main()
