@echo off
cd /d "%~dp0"
set /p SOURCE=Escribe la camara (0, 1, IP:PUERTO o URL completa): 
.venv\Scripts\python.exe probar_camara.py %SOURCE%
pause
