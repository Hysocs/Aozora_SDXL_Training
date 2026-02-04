@ECHO OFF
SETLOCAL

:: Keep window open by relaunching
IF "%~1"=="" (
    cmd /k "%~f0" KEEPALIVE
    EXIT /b
)

:: =================================================================================
::              SDXL Training Environment Setup Script (Final Fix)
:: =================================================================================

:: --- CONFIGURATION ---
SET VENV_DIR=portable_Venv
SET PYTHON_EXE=py -3.11
SET WHEELS_DIR=%VENV_DIR%\Wheels

:: PyTorch 2.9.1 + CUDA 12.8 Stack
SET TORCH_VERSION=2.9.1
SET TORCHVISION_VERSION=0.24.1
SET TORCHAUDIO_VERSION=2.9.1
SET XFORMERS_VERSION=0.0.33.post2

:: Flash Attention Wheel (Validated for PyTorch 2.9.1 + CUDA 12.8 + Python 3.11)
SET FLASH_ATTN_FILENAME=flash_attn-2.8.3+cu128torch2.9-cp311-cp311-win_amd64.whl
SET FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/%FLASH_ATTN_FILENAME%  

:: --- MENU ---
:MENU
CLS
ECHO.
ECHO === SDXL Training Environment Installer ===
ECHO [1] Install WITH Flash Attention 2 (Recommended for RTX 30xx/40xx)
ECHO [2] Install WITH xformers (Alternative - No Flash Attention)
ECHO.
ECHO PyTorch %TORCH_VERSION% + CUDA 12.8
ECHO.
CHOICE /C 12 /N /M "Enter choice [1 or 2]: "

IF ERRORLEVEL 2 SET INSTALL_MODE=XFORMERS & GOTO START_CHECKS
IF ERRORLEVEL 1 SET INSTALL_MODE=FLASH & GOTO START_CHECKS

:START_CHECKS
ECHO.
ECHO === Starting Setup [%INSTALL_MODE% mode] ===
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
    ECHO [ERROR] Python 3.11 not found. Install from python.org/downloads/
    GOTO FATAL_ERROR
)
%PYTHON_EXE% --version
ECHO [OK] Python 3.11+ found.
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

:: --- Upgrade pip ---
ECHO [INFO] Upgrading pip...
python -m pip install --upgrade pip
ECHO.

:: --- Core PyTorch Stack (torch + torchvision + torchaudio) ---
ECHO [INFO] Installing PyTorch %TORCH_VERSION% core stack...
python -m pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/cu128  
IF ERRORLEVEL 1 (
    ECHO [ERROR] PyTorch core installation failed.
    GOTO FATAL_ERROR
)
ECHO [OK] PyTorch core installed.
ECHO.

:: --- Attention Backend (Mutual Exclusive) ---
IF "%INSTALL_MODE%"=="FLASH" (
    ECHO [INFO] Installing Flash Attention 2.8.3 - mutual exclusive, no xformers...
    IF NOT EXIST "%WHEELS_DIR%" MKDIR "%WHEELS_DIR%"
    
    SET FLASH_PATH=%WHEELS_DIR%\%FLASH_ATTN_FILENAME%
    IF NOT EXIST "%FLASH_PATH%" (
        ECHO [INFO] Downloading Flash Attention wheel...
        curl -L --progress-bar -o "%FLASH_PATH%" %FLASH_ATTN_URL%
        IF ERRORLEVEL 1 (
            ECHO [ERROR] Download failed. Check internet or URL validity.
            GOTO FATAL_ERROR
        )
        ECHO [OK] Flash Attention wheel downloaded.
    ) ELSE (
        ECHO [INFO] Using cached Flash Attention wheel.
    )
    
    ECHO [INFO] Installing Flash Attention...
    python -m pip install "%FLASH_PATH%"
    IF ERRORLEVEL 1 GOTO FATAL_ERROR
    ECHO [OK] Flash Attention 2.8.3 installed.
    ECHO.
) ELSE (
    ECHO [INFO] Installing xformers %XFORMERS_VERSION% - mutual exclusive, no Flash Attention...
    python -m pip install xformers==%XFORMERS_VERSION% --index-url https://download.pytorch.org/whl/cu128  
    IF ERRORLEVEL 1 (
        ECHO [ERROR] xformers installation failed.
        GOTO FATAL_ERROR
    )
    ECHO [OK] xformers installed.
    ECHO.
)

:: --- Remaining Dependencies ---
ECHO [INFO] Installing packages from requirements.txt...
python -m pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO [ERROR] requirements.txt installation failed.
    GOTO FATAL_ERROR
)
ECHO [OK] All dependencies installed.
ECHO.

:: --- Final Verification ---
ECHO [INFO] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); import torchvision; print(f'Torchvision: {torchvision.__version__}')"

IF "%INSTALL_MODE%"=="FLASH" (
    :: Removed parentheses in ECHO below to fix crash
    python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>nul || ECHO [WARN] Flash Attention import failed - check installation.
) ELSE (
    python -c "import xformers; print(f'xformers: {xformers.__version__}')" 2>nul || ECHO [WARN] xformers import failed - check installation.
)
ECHO [OK] Verification complete.
ECHO.

:: --- Success ---
ECHO.
ECHO ========================================
ECHO    INSTALLATION COMPLETE! (%INSTALL_MODE%)
ECHO ========================================
ECHO.
ECHO Recommended config setting:
IF "%INSTALL_MODE%"=="FLASH" (
    :: Removed parentheses below to fix crash
    ECHO   "MEMORY_EFFICIENT_ATTENTION": "flash_attn"   - or "sdpa"/"cudnn" for max speed
) ELSE (
    ECHO   "MEMORY_EFFICIENT_ATTENTION": "xformers"
)
ECHO.
ECHO To activate venv: CALL "%VENV_DIR%\Scripts\activate.bat", or run the start_gui bat
PAUSE
GOTO END

:FATAL_ERROR
ECHO.
ECHO ========================================
ECHO    INSTALLATION FAILED
ECHO ========================================
ECHO.
ECHO Common fixes:
ECHO 1. Python 3.11 not in PATH or 'py' launcher missing
ECHO 2. No/stable internet [PyPI/GitHub]
ECHO 3. Antivirus blocking curl/pip
ECHO 4. Disk space / permissions
ECHO.
PAUSE

:END
EXIT /b