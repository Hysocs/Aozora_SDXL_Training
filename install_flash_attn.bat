@ECHO OFF
SETLOCAL

:: Standalone Flash Attention installer for the portable venv.
:: Mirrors setup.bat option 1, with extra diagnostics.
:: Pass NOPAUSE when calling this from another batch script.

SET VENV_DIR=portable_Venv
SET WHEELS_DIR=%VENV_DIR%\Wheels
SET FLASH_ATTN_FILENAME=flash_attn-2.8.3+cu128torch2.9-cp311-cp311-win_amd64.whl
SET FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/%FLASH_ATTN_FILENAME%
SET FLASH_PATH=%WHEELS_DIR%\%FLASH_ATTN_FILENAME%
SET NO_PAUSE=0
IF /I "%~1"=="NOPAUSE" SET NO_PAUSE=1

ECHO.
ECHO === Flash Attention Standalone Installer ===
ECHO Wheel: %FLASH_ATTN_FILENAME%
ECHO URL:   %FLASH_ATTN_URL%
ECHO.

IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    ECHO [ERROR] Venv not found: %VENV_DIR%\Scripts\activate.bat
    ECHO Run setup.bat first so portable_Venv exists.
    GOTO FATAL_ERROR
)

CALL "%VENV_DIR%\Scripts\activate.bat"
IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to activate venv.
    GOTO FATAL_ERROR
)

ECHO [OK] Venv activated.
ECHO.

ECHO [INFO] Python:
python --version
ECHO.

ECHO [INFO] Pip:
python -m pip --version
ECHO.

ECHO [INFO] Torch check:
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
IF ERRORLEVEL 1 (
    ECHO [ERROR] Torch check failed. Flash Attention wheel requires torch 2.9.x + CUDA 12.8 in this script.
    GOTO FATAL_ERROR
)
ECHO.

IF NOT EXIST "%WHEELS_DIR%" (
    ECHO [INFO] Creating wheel cache: %WHEELS_DIR%
    MKDIR "%WHEELS_DIR%"
    IF ERRORLEVEL 1 (
        ECHO [ERROR] Failed to create wheel cache.
        GOTO FATAL_ERROR
    )
)

IF NOT EXIST "%FLASH_PATH%" (
    ECHO [INFO] Downloading Flash Attention wheel...
    curl -L --fail --show-error --progress-bar -o "%FLASH_PATH%" "%FLASH_ATTN_URL%"
    IF ERRORLEVEL 1 (
        ECHO [ERROR] Download failed.
        ECHO Check internet access, GitHub access, antivirus, or whether the release URL changed.
        GOTO FATAL_ERROR
    )
) ELSE (
    ECHO [INFO] Using cached wheel: %FLASH_PATH%
)
ECHO.

ECHO [INFO] Cached wheel details:
DIR "%FLASH_PATH%"
ECHO.

ECHO [INFO] Installing Flash Attention from wheel...
python -m pip install --force-reinstall --no-deps "%FLASH_PATH%"
IF ERRORLEVEL 1 (
    ECHO [ERROR] pip install failed.
    GOTO FATAL_ERROR
)
ECHO.

ECHO [INFO] Verifying package metadata:
python -m pip show flash-attn
IF ERRORLEVEL 1 (
    ECHO [ERROR] pip cannot find flash-attn after install.
    GOTO FATAL_ERROR
)
ECHO.

ECHO [INFO] Verifying import:
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__); print('path:', flash_attn.__file__)"
IF ERRORLEVEL 1 (
    ECHO [ERROR] flash_attn import failed after install.
    GOTO FATAL_ERROR
)

ECHO.
ECHO ========================================
ECHO    FLASH ATTENTION INSTALL COMPLETE
ECHO ========================================
ECHO.
IF "%NO_PAUSE%"=="0" PAUSE
GOTO END

:FATAL_ERROR
ECHO.
ECHO ========================================
ECHO    FLASH ATTENTION INSTALL FAILED
ECHO ========================================
ECHO.
IF "%NO_PAUSE%"=="0" PAUSE
EXIT /b 1

:END
EXIT /b 0
