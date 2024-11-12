@echo off
SET PYTHONWARNINGS=ignore
SET USE_VENV=1
SET VENV_DIR="%~dp0%venv"
SET LANGUAGE=en-US

if ["%1"] == ["pt-BR"] (
    SET LANGUAGE=%1%
)

if ["%USE_VENV%"] == ["0"] goto :skip_venv

echo Activating python venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
call %VENV_DIR%\Scripts\activate.bat

:skip_venv
if ["%LANGUAGE%"] == ["pt-BR"] goto :pt_BR
if ["%LANGUAGE%"] == ["en-US"] goto :en_US

:en_US
python app.py
goto :end

:pt_BR
python app_pt-br.py
goto :end

:end

