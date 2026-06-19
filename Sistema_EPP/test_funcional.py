from __future__ import annotations

import time

import numpy as np

from app import app, detector, VerificationSession, VERIFY_FRAMES

client = app.test_client()

PASS, FAIL, TOTAL = 0, 0, 0

def test(name: str):
    global TOTAL
    TOTAL += 1
    def wrapper(ok: bool, detail: str = ""):
        global PASS, FAIL
        if ok:
            PASS += 1
            print(f"  [OK] {name}")
        else:
            FAIL += 1
            print(f"  [FAIL] {name} - {detail}")
    return wrapper


# ==============================================================
#  HELPERS
# ==============================================================

def frame(h=360, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

class DFace:
    def __init__(self, ok=True, label="Juan"):
        self.ok = ok; self.label = label

class DSum:
    def __init__(self, h=True, v=True, act=True):
        self.helmet_ok = h; self.vest_ok = v; self.active = act
    def get(self, k, d=None):
        return getattr(self, k, d)


# ==============================================================
#  1. PAGINAS PRINCIPALES
# ==============================================================

print("\n=== 1. PAGINAS PRINCIPALES ===\n")

ok = test("GET / regresa 200")
ok(client.get("/").status_code == 200)

ok = test("GET /estado.json -> JSON con 'camaras'")
data = client.get("/estado.json").get_json()
ok(data and "camaras" in data)

ok = test("cada camara tiene 'verificacion'")
ok(all("verificacion" in c for c in data["camaras"]))

for url in ["/configuracion", "/personas", "/control-acceso", "/fotos"]:
    ok = test(f"GET {url} -> 200")
    ok(client.get(url).status_code == 200)


# ==============================================================
#  2. API VERIFICACION
# ==============================================================

print("\n=== 2. API VERIFICACION ===\n")

ok = test("GET estado/999 -> idle")
st = client.get("/api/verificacion/estado/999").get_json()
ok(st["state"] == "idle" and st["progress"] == 0.0)

ok = test("POST iniciar/999 -> ok=true")
ini = client.post("/api/verificacion/iniciar/999").get_json()
ok(ini.get("ok") is True)

ok = test("POST repetido -> ok=false")
ini2 = client.post("/api/verificacion/iniciar/999").get_json()
ok(ini2.get("ok") is False)


# ==============================================================
#  3. FLUJO COMPLETO VERIFICATION SESSION
# ==============================================================

print("\n=== 3. FLUJO VS ===\n")

vs = VerificationSession(888)
t = time.time()

# -- Estado inicial --
ok = test("state = idle al inicio")
ok(vs.state == "idle")

# -- Activar manual (nuevo: va directo a verifying) --
vs.manual_trigger = True
out = vs.update(frame(), DFace(ok=False), DSum(), t)

ok = test("con trigger -> verifying directo, overlay vacio")
ok(vs.state == "verifying" and out.get("overlay","") == "",
   f"state={vs.state}, overlay='{out.get('overlay','')}'")

ok = test("primer frame en verifying -> buffer empieza vacio")
ok(len(vs.buffer) == 0, f"buffer={len(vs.buffer)}")

# Llenar buffer con VERIFY_FRAMES frames positivos
for i in range(VERIFY_FRAMES):
    out = vs.update(frame(), DFace(ok=True, label="Juan Perez"),
                    DSum(h=True, v=True), t + i * 0.1)

ok = test(f"despues de {VERIFY_FRAMES} frames -> done, PERMITIDO")
ok(vs.state == "done" and vs.result and vs.result.allowed,
   f"state={vs.state}, allowed={vs.result.allowed if vs.result else None}")

ok = test("overlay contiene BIENVENIDO")
ok("BIENVENIDO" in out.get("overlay","").upper())

ok = test("get_public_state refleja resultado")
ps = vs.get_public_state()
ok(ps.get("allowed") is True and "Juan" in ps.get("person",""))

# -- Caso: sin chaleco --
vs2 = VerificationSession(889)
t = time.time()
vs2.manual_trigger = True
out = vs2.update(frame(), DFace(ok=True, label="Pedro"), DSum(h=True, v=False), t)
for i in range(VERIFY_FRAMES):
    out = vs2.update(frame(), DFace(ok=True, label="Pedro"),
                     DSum(h=True, v=False), t + i * 0.1)

ok = test("sin chaleco -> done, DENEGADO")
ok(vs2.state == "done" and vs2.result and not vs2.result.allowed,
   f"state={vs2.state}, allowed={vs2.result.allowed if vs2.result else None}")

ok = test("motivo contiene 'chaleco'")
ok("chaleco" in (vs2.result.reason or "").lower(),
   f"reason={vs2.result.reason}")

# -- Cancel manual --
vs3 = VerificationSession(890)
vs3.manual_trigger = True
out = vs3.update(frame(), DFace(ok=False), DSum(), t)
ok = test("cancel() -> idle y result=None")
vs3.cancel()
ok(vs3.state == "idle" and vs3.result is None)

# -- API stop --
ok = test("POST /api/verificacion/detener/999 -> ok")
r = client.post("/api/verificacion/detener/999")
ok(r.status_code == 200 and r.get_json().get("ok") is True)

# -- Auto-reset --
ok = test("7s despues -> reset a idle")
vs2.decision_made_at = t - 7.0
out = vs2.update(frame(), DFace(ok=False), DSum(), t)
ok(vs2.state == "idle" and vs2.result is None)


# ==============================================================
#  4. DETECTOR
# ==============================================================

print("\n=== 4. DETECTOR ===\n")

ok = test("confidence = 0.25")
ok(detector.confidence == 0.25, f"conf={detector.confidence}")


# ==============================================================
#  5. REGISTRO PERSONAS
# ==============================================================

print("\n=== 5. REGISTRO PERSONAS ===\n")

ok = test("POST /personas -> 302")
ok(client.post("/personas", data={
    "full_name": "Test User", "area": "QA", "person_type": "Personal"
}).status_code == 302)

ok = test("GET /personas -> muestra el nombre")
r = client.get("/personas")
ok("Test User" in r.get_data(as_text=True))


# ==============================================================
#  6. FACE API
# ==============================================================

print("\n=== 6. FACE API ===\n")

from app import face_api
ok = test("is_facing_camera(frame vacio) -> False")
fc, sc = face_api.is_facing_camera(frame())
ok(not fc and sc == 0.0)


# ==============================================================
#  7. REPORTE EXCEL
# ==============================================================

print("\n=== 7. REPORTE EXCEL ===\n")

ok = test("GET /reporte/excel -> 200 + content-type Excel")
r = client.get("/reporte/excel")
ok(r.status_code == 200 and "spreadsheetml" in r.content_type)


# ==============================================================
#  SUMMARY
# ==============================================================

line = "=" * 50
print(f"\n{line}")
print(f"RESULTADOS: {PASS}/{TOTAL} pruebas pasaron")
print(f"{line}\n")
if FAIL > 0:
    exit(1)
