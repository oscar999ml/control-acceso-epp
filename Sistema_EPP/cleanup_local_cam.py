import sqlite3
import os

db_path = 'data/epp_events.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute('DELETE FROM cameras WHERE source = "0"')
    conn.commit()
    conn.close()
    print("Cámara local eliminada con éxito.")
else:
    print("Base de datos no encontrada.")
