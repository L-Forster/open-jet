@echo off
setlocal

cd /d "%~dp0"

set "OPENJET_REPO_URL=https://github.com/l-forster/open-jet.git"
if defined LOCALAPPDATA (
    set "OPENJET_BASE=%LOCALAPPDATA%\OpenJet"
) else (
    set "OPENJET_BASE=%USERPROFILE%\.openjet"
)
set "OPENJET_SOURCE=%OPENJET_BASE%\source"
set "OPENJET_BIN=%OPENJET_BASE%\bin"

set "PYTHON_EXE="
set "PYTHON_ARGS="

if defined PYTHON (
    "%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo error: PYTHON does not point to Python 3.10 or newer.
        exit /b 1
    )
    set "PYTHON_EXE=%PYTHON%"
) else (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    )
)

if not defined PYTHON_EXE (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo error: Python 3.10 or newer was not found.
    echo Install Python from https://www.python.org/downloads/windows/ and enable "Add python.exe to PATH".
    exit /b 1
)

if exist "pyproject.toml" if exist "src\cli.py" (
    set "OPENJET_SOURCE=%CD%"
) else (
    git --version >nul 2>nul
    if errorlevel 1 (
        echo error: git was not found.
        echo Install Git for Windows from https://git-scm.com/download/win and rerun this installer.
        exit /b 1
    )
    if not exist "%OPENJET_BASE%" mkdir "%OPENJET_BASE%"
    if errorlevel 1 exit /b 1
    if exist "%OPENJET_SOURCE%\.git" (
        git -C "%OPENJET_SOURCE%" pull --ff-only
        if errorlevel 1 exit /b 1
    ) else (
        if exist "%OPENJET_SOURCE%" (
            echo error: %OPENJET_SOURCE% exists but is not an OpenJet git checkout.
            exit /b 1
        )
        git clone "%OPENJET_REPO_URL%" "%OPENJET_SOURCE%"
        if errorlevel 1 exit /b 1
    )
    cd /d "%OPENJET_SOURCE%"
)

if exist "src\*.pyd" del /q "src\*.pyd"

if not exist ".venv\Scripts\python.exe" (
    "%PYTHON_EXE%" %PYTHON_ARGS% -m venv .venv
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip "setuptools>=68,<80" wheel
if errorlevel 1 exit /b 1

set "OPENJET_BUILD_EXTENSIONS=0"
".venv\Scripts\python.exe" -m pip install --no-build-isolation -e .
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check hf_transfer "huggingface_hub>=0.25"
if errorlevel 1 echo warning: could not install hf_transfer; downloads will fall back to single-stream

".venv\Scripts\python.exe" -c "from src.observation.processors import provision_default_faster_whisper_model; raise SystemExit(0 if provision_default_faster_whisper_model() else 1)"
if errorlevel 1 echo warning: could not pre-provision local transcription assets; OpenJet will retry on first microphone use

if not exist "%OPENJET_BIN%" mkdir "%OPENJET_BIN%"
if errorlevel 1 exit /b 1

>"%OPENJET_BIN%\openjet.cmd" echo @echo off
>>"%OPENJET_BIN%\openjet.cmd" echo "%CD%\.venv\Scripts\openjet.exe" %%*
>"%OPENJET_BIN%\open-jet.cmd" echo @echo off
>>"%OPENJET_BIN%\open-jet.cmd" echo "%CD%\.venv\Scripts\open-jet.exe" %%*

echo %PATH% | find /I "%OPENJET_BIN%" >nul
if errorlevel 1 (
    set "OLD_PATH=%PATH%"
    set "PATH=%OPENJET_BIN%;%PATH%"
    setx PATH "%OPENJET_BIN%;%OLD_PATH%" >nul
)

echo Installed open-jet from %CD%
call "%OPENJET_BIN%\openjet.cmd" setup
exit /b %ERRORLEVEL%
