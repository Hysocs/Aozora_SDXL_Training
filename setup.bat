@ECHO OFF
SETLOCAL

:: Keep window open by relaunching
IF "%~1"=="" (
    cmd /k "%~f0" KEEPALIVE
    EXIT /b
)

:: =================================================================================
::              SDXL Training Environment Setup Script (Fixed)
:: =================================================================================

:: --- CONFIGURATION ---
SET VENV_DIR=portable_Venv
SET PYTHON_EXE=py -3.12
SET WHEELS_DIR=%VENV_DIR%\Wheels

:: PyTorch 2.9.1 + CUDA 12.8 Stack
SET TORCH_VERSION=2.9.1
SET TORCHVISION_VERSION=0.24.1
SET TORCHAUDIO_VERSION=2.9.1
SET XFORMERS_VERSION=0.0.33.post2

:: Flash Attention Wheel (for PyTorch 2.9.1 + CUDA 12.8 + Python 3.12)
SET FLASH_ATTN_FILENAME=flash_attn-2.8.2+cu128torch2.9.1-cp312-none-win_amd64.whl
SET FLASH_ATTN_URL=https://github.com/kingbri1/flash-attention/releases/download/v2.8.2/%FLASH_ATTN_FILENAME%

:: --- MENU ---
:MENU
CLS
ECHO.
ECHO === SDXL Training Environment Installer ===
ECHO [1] Install WITH Flash Attention (RTX 30xx/40xx)
ECHO [2] Install WITHOUT Flash Attention (AMD/Older NVIDIA)
ECHO.
ECHO Python 3.12, PyTorch %TORCH_VERSION%, CUDA 12.8
ECHO.
CHOICE /C 12 /N /M "Enter choice [1 or 2]: "

IF ERRORLEVEL 2 SET INSTALL_MODE=NO_FLASH & GOTO START_CHECKS
IF ERRORLEVEL 1 SET INSTALL_MODE=FLASH & GOTO START_CHECKS

:START_CHECKS
ECHO.
ECHO === Starting Setup ===
ECHO.

:: --- CUDA Check ---
where nvidia-smi >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [ERROR] nvidia-smi not found. Install NVIDIA drivers from nvidia.com/drivers
    GOTO FATAL_ERROR
)
ECHO [OK] CUDA detected.
ECHO.

:: --- Python Check ---
where %PYTHON_EXE% >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO [ERROR] Python 3.12 not found. Install from python.org/downloads/
    GOTO FATAL_ERROR
)
%PYTHON_EXE% --version
ECHO [OK] Python 3.12+ found.
ECHO.

:: --- Requirements.txt Check ---
IF NOT EXIST requirements.txt (
    ECHO [ERROR] requirements.txt not found in: %CD%
    GOTO FATAL_ERROR
)
ECHO [OK] requirements.txt found.
ECHO.

:: --- Virtual Environment ---
IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    ECHO [INFO] Creating virtual environment...
    %PYTHON_EXE% -m venv %VENV_DIR%
    IF ERRORLEVEL 1 GOTO FATAL_ERROR
    ECHO [OK] venv created.
) ELSE (
    ECHO [OK] venv exists.
)
ECHO.

:: --- Activate venv ---
CALL "%VENV_DIR%\Scripts\activate.bat"
IF ERRORLEVEL 1 GOTO FATAL_ERROR
ECHO [OK] venv activated.
ECHO.

:: --- Set Encoding ---
SET PYTHONIOENCODING=utf-8

:: --- Core ML Stack (PyTorch + Vision + Audio + Xformers) ---
ECHO [INFO] Installing PyTorch %TORCH_VERSION% stack...
ECHO [INFO] This may take 10-20 minutes depending on your internet speed...
python -m pip install --upgrade pip
python -m pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% xformers==%XFORMERS_VERSION% --index-url https://download.pytorch.org/whl/cu128
IF ERRORLEVEL 1 (
    ECHO [ERROR] PyTorch installation failed. Check your internet/PyPI access.
    GOTO FATAL_ERROR
)
ECHO [OK] PyTorch + Xformers installed.
ECHO.

:: --- Flash Attention ---
IF "%INSTALL_MODE%"=="FLASH" (
    ECHO [INFO] Installing Flash Attention v2.8.2...
    IF NOT EXIST "%WHEELS_DIR%" MKDIR "%WHEELS_DIR%"
    
    SET FLASH_PATH=%WHEELS_DIR%\%FLASH_ATTN_FILENAME%
    IF NOT EXIST "%FLASH_PATH%" (
        ECHO [INFO] Downloading from GitHub...
        curl -L --progress-bar -o "%FLASH_PATH%" %FLASH_ATTN_URL%
        IF ERRORLEVEL 1 (
            ECHO [ERROR] Download failed. Check URL or internet.
            GOTO FATAL_ERROR
        )
        ECHO [OK] Downloaded.
    ) ELSE (
        ECHO [INFO] Using cached wheel.
    )
    
    ECHO [INFO] Installing wheel...
    python -m pip install "%FLASH_PATH%"
    IF ERRORLEVEL 1 GOTO FATAL_ERROR
    ECHO [OK] Flash Attention installed.
    ECHO.
)

:: --- Dependencies ---
ECHO [INFO] Installing remaining packages from requirements.txt...
python -m pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO [ERROR] Package installation failed. Check requirements.txt
    GOTO FATAL_ERROR
)
ECHO [OK] All packages installed.
ECHO.

:: --- Final Verification ---
ECHO [INFO] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); import torchvision; print(f'Vision: {torchvision.__version__}'); import xformers; print(f'Xformers: {xformers.__version__}')"
IF "%INSTALL_MODE%"=="FLASH" (
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>nul || ECHO [WARN] Flash import test failed.
)
ECHO [OK] Verification complete.
ECHO.

:: --- Success ---
ECHO.
ECHO ========================================
ECHO    INSTALLATION COMPLETE!
ECHO ========================================
ECHO.
ECHO To activate manually: CALL "%VENV_DIR%\Scripts\activate.bat"
PAUSE
GOTO END

:FATAL_ERROR
ECHO.
ECHO ========================================
ECHO    INSTALLATION FAILED
ECHO ========================================
ECHO.
ECHO Common issues:
ECHO 1. Python 3.12 not installed or 'py' launcher not in PATH
ECHO 2. No internet connection to PyPI
ECHO 3. requirements.txt contains invalid packages
ECHO 4. Insufficient disk space
ECHO 5. Antivirus blocking pip/curl
ECHO.
PAUSE

:END
EXIT /b