@ECHO OFF
CLS
SETLOCAL ENABLEDELAYEDEXPANSION

chcp 65001 >nul 2>&1

:: Setup for ANSI colors in the console.
FOR /F "tokens=1,2 delims=#" %%a IN ('"PROMPT #$E# & FOR %%b IN (1) DO REM"') DO SET "ESC=%%a"

:: Define color variables for easier use.
SET "COLOR_RESET=!ESC![0m"
SET "COLOR_RED=!ESC![91m"
SET "COLOR_GREEN=!ESC![92m"
SET "COLOR_YELLOW=!ESC![93m"
SET "COLOR_CYAN=!ESC![96m"
SET "COLOR_WHITE=!ESC![97m"
SET "COLOR_BLUE=!ESC![94m"

:: =================================================================================
::              SDXL Training Environment Setup Script (Fully Automated)
:: =================================================================================

:: --- ( STEP 1: CONFIGURE YOUR PATHS AND URLS HERE ) ---
SET VENV_DIR=portable_Venv
SET PYTHON_EXE=py -3.11
SET VENV_PATH=.\!VENV_DIR!\Scripts
SET WHEELS_DIR=.\!VENV_DIR!\Wheels

:: PyTorch 2.8.0 + CUDA 12.8 Wheels
SET FLASH_ATTN_URL=https://github.com/kingbri1/flash-attention/releases/download/v2.8.2/flash_attn-2.8.2+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
SET FLASH_ATTN_FILENAME=flash_attn-2.8.2+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl

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

:: --- CUDA Detection ---
ECHO !COLOR_WHITE![INFO] Checking for NVIDIA CUDA...!COLOR_RESET!
where nvidia-smi >nul 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] CUDA not detected! nvidia-smi not found.!COLOR_RESET!
    ECHO !COLOR_YELLOW![WARNING] This installer requires an NVIDIA GPU with CUDA support.!COLOR_RESET!
    ECHO !COLOR_YELLOW!          Please install NVIDIA drivers and CUDA toolkit.!COLOR_RESET!
    GOTO:FATAL_ERROR
)

nvidia-smi >nul 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] CUDA check failed! No NVIDIA GPU detected.!COLOR_RESET!
    GOTO:FATAL_ERROR
)

ECHO !COLOR_GREEN![SUCCESS] CUDA detected successfully.!COLOR_RESET!
ECHO.

:: --- Pre-flight Checks ---
IF NOT EXIST "requirements.txt" (
    ECHO !COLOR_RED![ERROR] requirements.txt not found!COLOR_RESET!
    GOTO:FATAL_ERROR
)

where %PYTHON_EXE% >nul 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] Python 3.11 not found!COLOR_RESET!
    ECHO !COLOR_YELLOW!        Please install Python 3.11 and ensure it's in your PATH.!COLOR_RESET!
    GOTO:FATAL_ERROR
)

:: --- Virtual Environment Setup ---
IF NOT EXIST "!VENV_PATH!\activate.bat" (
    ECHO !COLOR_WHITE![INFO] Creating virtual environment...!COLOR_RESET!
    %PYTHON_EXE% -m venv %VENV_DIR% >nul 2>nul
    IF ERRORLEVEL 1 (
        ECHO !COLOR_RED![ERROR] Failed to create virtual environment.!COLOR_RESET!
        GOTO:FATAL_ERROR
    )
    ECHO !COLOR_GREEN![SUCCESS] Virtual environment created.!COLOR_RESET!
) ELSE (
    ECHO !COLOR_GREEN![INFO] Virtual environment found.!COLOR_RESET!
)
ECHO.

ECHO !COLOR_WHITE![INFO] Activating virtual environment...!COLOR_RESET!
CALL "!VENV_PATH!\activate.bat"
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] Failed to activate virtual environment.!COLOR_RESET!
    GOTO:FATAL_ERROR
)
ECHO !COLOR_GREEN![SUCCESS] Virtual environment activated.!COLOR_RESET!
ECHO.

SET PYTHONIOENCODING=utf-8

:: --- Pip Upgrade ---
ECHO !COLOR_WHITE![INFO] Upgrading pip...!COLOR_RESET!
python -m pip install --upgrade pip --quiet --no-warn-script-location 2>nul
ECHO !COLOR_GREEN![SUCCESS] Pip upgraded.!COLOR_RESET!
ECHO.

:: --- PyTorch Installation ---
ECHO !COLOR_WHITE![INFO] Installing PyTorch 2.8.0 + CUDA 12.8...!COLOR_RESET!
ECHO !COLOR_BLUE!       This may take several minutes, please wait...!COLOR_RESET!
python -m pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet --no-warn-script-location 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] PyTorch installation failed.!COLOR_RESET!
    ECHO !COLOR_YELLOW![INFO] Opening detailed output in new window...!COLOR_RESET!
    START "PyTorch Installation - Details" cmd /k "python -m pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    GOTO:FATAL_ERROR
)
ECHO !COLOR_GREEN![SUCCESS] PyTorch stack installed.!COLOR_RESET!
ECHO.

:: --- Xformers Installation ---
ECHO !COLOR_WHITE![INFO] Installing Xformers for PyTorch 2.8.0 + CUDA 12.8...!COLOR_RESET!
ECHO !COLOR_BLUE!       This may take several minutes, please wait...!COLOR_RESET!
python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu128 --quiet --no-warn-script-location 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] Xformers installation failed.!COLOR_RESET!
    ECHO !COLOR_YELLOW![INFO] Opening detailed output in new window...!COLOR_RESET!
    START "Xformers Installation - Details" cmd /k "python -m pip install -U xformers --index-url https://download.pytorch.org/whl/cu128"
    GOTO:FATAL_ERROR
)
ECHO !COLOR_GREEN![SUCCESS] Xformers installed.!COLOR_RESET!
ECHO.

:: --- Flash Attention Installation (Optional) ---
IF "!INSTALL_MODE!"=="FLASH" (
    ECHO !COLOR_WHITE![INFO] Installing Flash Attention...!COLOR_RESET!
    
    IF NOT EXIST "!WHEELS_DIR!" MKDIR "!WHEELS_DIR!" >nul 2>nul
    
    IF NOT EXIST "!FLASH_ATTN_WHL_PATH!" (
        ECHO !COLOR_BLUE!       Downloading flash-attention wheel...!COLOR_RESET!
        curl -L --progress-bar -o "!FLASH_ATTN_WHL_PATH!" !FLASH_ATTN_URL! 2>nul
        IF ERRORLEVEL 1 (
            ECHO !COLOR_RED![ERROR] Flash Attention download failed.!COLOR_RESET!
            DEL "!FLASH_ATTN_WHL_PATH!" >nul 2>nul
            GOTO:FATAL_ERROR
        )
        ECHO !COLOR_GREEN![SUCCESS] Flash Attention downloaded.!COLOR_RESET!
    ) ELSE (
        ECHO !COLOR_GREEN![INFO] Flash Attention wheel found locally.!COLOR_RESET!
    )
    
    python -m pip install "!FLASH_ATTN_WHL_PATH!" --quiet --no-warn-script-location 2>nul
    IF ERRORLEVEL 1 (
        ECHO !COLOR_RED![ERROR] Flash Attention installation failed.!COLOR_RESET!
        ECHO !COLOR_YELLOW![INFO] Opening detailed output in new window...!COLOR_RESET!
        START "Flash Attention Installation - Details" cmd /k "python -m pip install "!FLASH_ATTN_WHL_PATH!""
        GOTO:FATAL_ERROR
    )
    ECHO !COLOR_GREEN![SUCCESS] Flash Attention installed.!COLOR_RESET!
    ECHO.
)

:: --- Requirements.txt Installation ---
ECHO !COLOR_WHITE![INFO] Installing remaining dependencies...!COLOR_RESET!
ECHO !COLOR_BLUE!       This may take several minutes, please wait...!COLOR_RESET!
python -m pip install -r requirements.txt --quiet --no-warn-script-location 2>nul
IF ERRORLEVEL 1 (
    ECHO !COLOR_RED![ERROR] Failed to install dependencies from requirements.txt!COLOR_RESET!
    ECHO !COLOR_YELLOW![INFO] Opening detailed output in new window...!COLOR_RESET!
    START "Requirements Installation - Details" cmd /k "python -m pip install -r requirements.txt"
    GOTO:FATAL_ERROR
)
ECHO !COLOR_GREEN![SUCCESS] All dependencies installed.!COLOR_RESET!
ECHO.

:: --- Success ---
ECHO.
ECHO !COLOR_GREEN!============================================================!COLOR_RESET!
ECHO !COLOR_GREEN!      INSTALLATION COMPLETE!                                !COLOR_RESET!
ECHO !COLOR_GREEN!============================================================!COLOR_RESET!
ECHO.
ECHO !COLOR_CYAN!Your SDXL training environment is ready to use.!COLOR_RESET!
ECHO.
GOTO:END

:FATAL_ERROR
ECHO.
ECHO !COLOR_RED!============================================================!COLOR_RESET!
ECHO !COLOR_RED!      INSTALLATION FAILED                                   !COLOR_RESET!
ECHO !COLOR_RED!============================================================!COLOR_RESET!
ECHO.
ECHO !COLOR_YELLOW!Please check the error messages above and try again.!COLOR_RESET!
ECHO.

:END
PAUSE
EXIT