@echo off

SET USE_VENV=1
SET VENV_DIR=%~dp0%venv

if ["%1"] == ["T"] (
    echo Cleaning virtual env folder.     
    rmdir /s /q venv >stdout.txt 2>stderr.txt    
)

if ["%2"] == ["F"] (
    SET USE_VENV=0
    echo You chose not to use venv...    
)

if not defined PYTHON (set PYTHON=python)

dir "%VENV_DIR%\Scripts\Python.exe" >stdout.txt 2>stderr.txt
if %ERRORLEVEL% == 0 goto :end

if ["%USE_VENV%"] == ["0"] goto :skip_venv

for /f "delims=" %%i in ('CALL %PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_FULLNAME="%%i"
echo Creating python venv dir %VENV_DIR% in using python %PYTHON_FULLNAME%
%PYTHON_FULLNAME% -m venv %VENV_DIR% >stdout.txt 2>stderr.txt

echo Activating python venv
set PYTHON="%VENV_DIR%\Scripts\Python.exe"
echo venv %PYTHON%
call %VENV_DIR%\Scripts\activate.bat

:skip_venv
python.exe -m pip install --upgrade pip
echo Installing requirements. This could take a few minutes...
pip install -r requirements.txt
goto:end

if %ERRORLEVEL% == 0 goto :end
echo Cannot activate python venv, aborting... %VENV_DIR%
goto :show_stdout_stderr

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type stderr.txt

:end
echo.
echo All done!. Launch 'start.bat'.
pause

