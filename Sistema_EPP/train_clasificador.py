from clasificador_api import ClasificadorEPP

print("=== ENTRENANDO CLASIFICADOR DE CASCO ===")
clf = ClasificadorEPP()
clf.entrenar_casco(epochs=80, lr=0.001)
print()

print("=== ENTRENANDO CLASIFICADOR DE CHALECO ===")
clf.entrenar_chaleco(epochs=80, lr=0.001)
print()

print("=== VERIFICACIÓN ===")
clf.cargar_modelos()

import cv2
from pathlib import Path

data = Path(__file__).parent / "data" / "entrenamiento"

print("\n--- Test casco ---")
for clase in ["casco", "no_casco"]:
    carpeta = data / "casco_clf" / clase
    for f in sorted(carpeta.glob("*"))[:3]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        label, conf = clf.predecir_casco(img)
        print(f"  {f.name}: pred={label} conf={conf:.3f}")

print("\n--- Test chaleco ---")
for clase in ["chaleco"]:
    carpeta = data / "chaleco_clf" / clase
    for f in sorted(carpeta.glob("*"))[:3]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        label, conf = clf.predecir_chaleco(img)
        print(f"  {f.name}: pred={label} conf={conf:.3f}")

print("\n[OK] Entrenamiento completo")
