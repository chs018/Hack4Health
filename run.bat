@echo off
REM Quick Start Script for Tamil Poetry Emotion-Rasa Classifier
REM Windows Batch File

echo ================================================================================
echo TAMIL POETRY EMOTION-RASA CLASSIFICATION SYSTEM
echo Hackathon-Ready AI Project
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo [SETUP] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if packages are installed
echo [SETUP] Checking dependencies...
pip list | findstr "transformers" >nul
if errorlevel 1 (
    echo [SETUP] Installing required packages...
    pip install -r requirements.txt
    echo.
) else (
    echo [SETUP] Dependencies already installed
    echo.
)

REM Run setup test
echo [TEST] Running setup verification...
python test_setup.py
if errorlevel 1 (
    echo.
    echo [ERROR] Setup test failed! Please check the errors above.
    pause
    exit /b 1
)
echo.

REM Check if model exists
if not exist "models\emotion_model" (
    echo [TRAIN] Model not found. Starting training...
    echo This will take 10-30 minutes depending on your hardware.
    echo.
    set /p train="Do you want to train the model now? (y/n): "
    if /i "%train%"=="y" (
        python backend\train.py
        echo.
    ) else (
        echo.
        echo [INFO] Skipping training. Run 'python backend\train.py' manually later.
        echo.
    )
) else (
    echo [INFO] Model already trained
    echo.
)

echo ================================================================================
echo READY TO RUN!
echo ================================================================================
echo.
echo Choose an option:
echo   1. Start API Server (backend)
echo   2. Launch Dashboard (frontend)
echo   3. Train Model
echo   4. Run Tests
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting API Server...
    echo Access at: http://localhost:8000
    echo API Docs: http://localhost:8000/docs
    echo.
    python backend\app.py
) else if "%choice%"=="2" (
    echo.
    echo Launching Dashboard...
    echo Opening in browser: http://localhost:8501
    echo.
    streamlit run frontend\dashboard.py
) else if "%choice%"=="3" (
    echo.
    echo Training Model...
    python backend\train.py
) else if "%choice%"=="4" (
    echo.
    echo Running Tests...
    python test_setup.py
) else if "%choice%"=="5" (
    echo.
    echo Goodbye!
    exit /b 0
) else (
    echo.
    echo Invalid choice!
)

echo.
pause
