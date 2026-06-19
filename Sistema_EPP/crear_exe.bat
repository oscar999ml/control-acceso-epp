@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller --noconfirm --onedir --name Sistema_EPP --add-data "templates;templates" --add-data "static;static" --add-data "data;data" app.py
echo.
echo EXE creado en: dist\Sistema_EPP\Sistema_EPP.exe
echo Nota: para esta version conviene ejecutar el EXE desde esta carpeta o mantener Casco.pt y Chaleco.pt en Modelo_de_EPP.
pause
