CLS
SETLOCAL ENABLEDELAYEDEXPANSION

chcp 65001 >nul

:: Setup for ANSI colors in the console.
FOR /F "tokens=1,2 delims=#" %%a IN ('"PROMPT #$E# & FOR %%b IN (1) DO REM"') DO SET "ESC=%%a"

:: Define color variables for easier use.
SET "COLOR_RESET=!ESC![0m"
SET "COLOR_RED=!ESC![91m"
SET "COLOR_GREEN=!ESC![92m"
SET "COLOR_YELLOW=!ESC![93m"
SET "COLOR_CYAN=!ESC![96m"
SET "COLOR_WHITE=!ESC![97m"

:: =================================================================================
::              SDXL Training Environment Setup Script (Fully Automated)
:: =================================================================================

:: --- ( STEP 1: CONFIGURE YOUR PATHS AND URLS HERE ) ---
SET VENV_DIR=portable_Venv
SET PYTHON_EXE=py -3.10
SET VENV_PATH=.\!VENV_DIR!\Scripts
SET WHEELS_DIR=.\!VENV_DIR!\Wheels


SET FLASH_ATTN_URL="https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%%2Bcu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl"
SET FLASH_ATTN_FILENAME=flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
SET FLASH_ATTN_WHL_PATH=!WHEELS_DIR!\!FLASH_ATTN_FILENAME!

:: --- ( END OF CONFIGURATION ) ---

:MENU
CLS
ECHO !COLOR_CYAN!====================================================!COLOR_RESET!
ECHO !COLOR_CYAN!        SDXL Training Environment Installer         !COLOR_RESET!
ECHO !COLOR_CYAN!====================================================!COLOR_RESET!
ECHO.
ECHO !COLOR_WHITE!Please choose your installation type:!COLOR_RESET!
ECHO.
ECHO   !COLOR_CYAN![1]!COLOR_RESET! Install !COLOR_YELLOW!WITH!COLOR_RESET! Flash Attention (Recommended for NVIDIA RTX cards)
ECHO   !COLOR_CYAN![2]!COLOR_RESET! Install !COLOR_YELLOW!WITHOUT!COLOR_RESET! Flash Attention (For AMD or older NVIDIA cards)
ECHO.
ECHO !COLOR_YELLOW!Note: The installation process downloads many files and may take a while.!COLOR_RESET!
ECHO.

CHOICE /C 12 /N /M "Enter your choice [1 or 2]: "

IF ERRORLEVEL 2 GOTO InstallWithoutFlash
IF ERRORLEVEL 1 GOTO InstallWithFlash
GOTO MENU

:InstallWithFlash
ECHO.
ECHO !COLOR_GREEN![INFO] Proceeding with Flash Attention installation.!COLOR_RESET!
SET "INSTALL_MODE=FLASH"
GOTO:START_CHECKS

:InstallWithoutFlash
ECHO.
ECHO !COLOR_GREEN![INFO] Proceeding without Flash Attention installation.!COLOR_RESET!
SET "INSTALL_MODE=NO_FLASH"
GOTO:START_CHECKS

:START_CHECKS
ECHO.
ECHO !COLOR_CYAN!===================================================!COLOR_RESET!
ECHO !COLOR_CYAN!  Starting Automated Environment Setup             !COLOR_RESET!
ECHO !COLOR_CYAN!===================================================!COLOR_RESET!
ECHO.

:: Pre-flight Checks
IF NOT EXIST "requirements.txt" (
    ECHO !COLOR_RED![ERROR] requirements.txt not found! Please save the corrected version.!COLOR_RESET!
    GOTO:FATAL_ERROR
)
where %PYTHON_EXE% >nul 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] Python command '%PYTHON_EXE%' not found.!COLOR_RESET!
    GOTO:FATAL_ERROR
)
IF NOT EXIST "!VENV_PATH!\activate.bat" (
    ECHO !COLOR_WHITE![INFO] Virtual environment not found. Creating it now...!COLOR_RESET!
    %PYTHON_EXE% -m venv %VENV_DIR%
    IF ERRORLEVEL 1 (ECHO !COLOR_RED![ERROR] Failed to create venv.!COLOR_RESET! & GOTO:FATAL_ERROR)
    ECHO !COLOR_GREEN![SUCCESS] Virtual environment created.!COLOR_RESET!
) ELSE (
    ECHO !COLOR_GREEN![INFO] Virtual environment found.!COLOR_RESET!
)
ECHO.

ECHO !COLOR_WHITE![INFO] Activating virtual environment...!COLOR_RESET!
CALL "!VENV_PATH!\activate.bat"
IF ERRORLEVEL 1 (ECHO !COLOR_RED![ERROR] Failed to activate venv.!COLOR_RESET! & GOTO:FATAL_ERROR)
ECHO !COLOR_GREEN![SUCCESS] Virtual environment is active.!COLOR_RESET! & python --version & ECHO.

SET PYTHONIOENCODING=utf-8
ECHO !COLOR_WHITE![INFO] Upgrading pip...!COLOR_RESET!
python -m pip install --upgrade pip
ECHO.

:: Check/download Flash Attention if selected
IF "!INSTALL_MODE!"=="FLASH" (
    IF NOT EXIST "!FLASH_ATTN_WHL_PATH!" (
        ECHO !COLOR_WHITE![INFO] Flash Attention .whl file not found. Attempting to download...!COLOR_RESET!
        IF NOT EXIST "!WHEELS_DIR!" MKDIR "!WHEELS_DIR!"
        ECHO !COLOR_CYAN![COMMAND] curl -L -o "!FLASH_ATTN_WHL_PATH!" !FLASH_ATTN_URL!!COLOR_RESET!
        curl -L -o "!FLASH_ATTN_WHL_PATH!" !FLASH_ATTN_URL!
        IF ERRORLEVEL 1 (ECHO !COLOR_RED![ERROR] Download failed.!COLOR_RESET! & DEL "!FLASH_ATTN_WHL_PATH!" >nul 2>nul & GOTO:FATAL_ERROR)
        ECHO !COLOR_GREEN![SUCCESS] Download complete.!COLOR_RESET!
    ) ELSE (
        ECHO !COLOR_GREEN![INFO] Flash Attention wheel found locally.!COLOR_RESET!
    )
    ECHO.

    ECHO !COLOR_WHITE![INFO] Installing Flash Attention from local .whl file...!COLOR_RESET!
    python -m pip install "!FLASH_ATTN_WHL_PATH!"
    IF ERRORLEVEL 1 (ECHO !COLOR_RED![ERROR] Failed to install Flash Attention wheel.!COLOR_RESET! & GOTO:FATAL_ERROR)
    ECHO !COLOR_GREEN![SUCCESS] Flash Attention installed.!COLOR_RESET! & ECHO.
)

ECHO !COLOR_WHITE![INFO] Installing remaining dependencies from requirements.txt...!COLOR_RESET!
python -m pip install -r requirements.txt
IF ERRORLEVEL 1 (ECHO !COLOR_RED![ERROR] Failed to install from requirements.txt.!COLOR_RESET! & GOTO:FATAL_ERROR)
ECHO !COLOR_GREEN![SUCCESS] All other dependencies installed.!COLOR_RESET! & ECHO.

:: [MODIFIED] Install the correct PyTorch stack FIRST. This is crucial.
:: We are using the CUDA 12.1 builds as they are a stable standard.
ECHO !COLOR_WHITE![INFO] Installing PyTorch, Torchvision, and Torchaudio... This is the most important step!!COLOR_RESET!
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
IF ERRORLEVEL 1 (ECHO !COLOR_RED![ERROR] Failed to install PyTorch.!COLOR_RESET! & GOTO:FATAL_ERROR)
ECHO !COLOR_GREEN![SUCCESS] PyTorch stack installed.!COLOR_RESET! & ECHO.


ECHO.
ECHO !COLOR_GREEN!============================================================!COLOR_RESET!
ECHO !COLOR_GREEN!      SETUP COMPLETE! All packages are installed.           !COLOR_RESET!
ECHO !COLOR_GREEN!============================================================!COLOR_RESET!
ECHO.
GOTO:END

:FATAL_ERROR
ECHO.
ECHO !COLOR_RED!============================================================!COLOR_RESET!
ECHO !COLOR_RED!      SETUP FAILED. Please review the errors above.         !COLOR_RESET!
ECHO !COLOR_RED!============================================================!COLOR_RESET!
ECHO.

:END
PAUSE
EXIT