from __future__ import annotations

import threading
import winsound


def _beep(freq: int, duration: int):
    threading.Thread(target=lambda: winsound.Beep(freq, duration), daemon=True).start()


def play_start():
    """Atención: verification starting"""
    _beep(660, 150)


def play_ok():
    """Acceso permitido — tono ascendente"""
    _beep(523, 120)
    threading.Thread(target=lambda: (
        winsound.Beep(659, 120),
        winsound.Beep(784, 200),
    ), daemon=True).start()


def play_fail():
    """Acceso denegado — tono descendente / buzzer"""
    _beep(400, 250)
    threading.Thread(target=lambda: (
        winsound.Beep(300, 250),
    ), daemon=True).start()


def play_wait():
    """Esperando — toque sutil"""
    _beep(500, 80)
