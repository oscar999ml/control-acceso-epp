@echo off
echo ==========================================
echo Creando Ejecutable del Sistema EPP con IA
echo ==========================================
echo.
echo Instalando PyInstaller si no existe...
.venv\Scripts\pip install pyinstaller --quiet

echo.
echo Generando ejecutable (esto puede tardar unos minutos)...
.venv\Scripts\pyinstaller Sistema_EPP.spec --noconfirm

echo.
echo ==========================================
echo PROCESO FINALIZADO
echo El ejecutable se encuentra en: dist\Sistema_EPP\Sistema_EPP.exe
echo.
echo RECUERDA: Debes copiar toda la carpeta 'dist\Sistema_EPP' 
echo para que funcione en otra computadora.
echo ==========================================
pause
