@ECHO OFF
CLS

:: Path to your virtual environment's Scripts folder
SET VENV_PATH=.\portable_Venv\Scripts

ECHO [INFO] Activating virtual environment...
CALL "%VENV_PATH%\activate.bat"
IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to activate the virtual environment.
    ECHO         Make sure you have run setup.bat first!
    PAUSE
    EXIT /B
)

ECHO [INFO] Starting the Training GUI...
python gui.py

ECHO [INFO] GUI closed.
PAUSE