@echo off
echo ================================================================================
echo Tamil Poetry Emotion-Rasa Classification System
echo Complete Application Startup
echo ================================================================================
echo.

echo [1/3] Starting Backend API Server...
start "Backend API" cmd /k "cd /d %~dp0 && python -c "import sys; sys.path.append('.'); from backend import app; app.start_server()""
timeout /t 5 /nobreak >nul

echo [2/3] Initializing Database...
echo Database will be created automatically by the backend...
timeout /t 2 /nobreak >nul

echo [3/3] Starting Frontend Dashboard...
start "Frontend Dashboard" cmd /k "cd /d %~dp0 && streamlit run frontend/dashboard.py --server.port 8501"
timeout /t 3 /nobreak >nul

echo.
echo ================================================================================
echo APPLICATION READY!
echo ================================================================================
echo.
echo Backend API:        http://localhost:8000
echo API Documentation:  http://localhost:8000/docs
echo Frontend Dashboard: http://localhost:8501
echo.
echo Opening frontend in browser...
timeout /t 3 /nobreak >nul
start http://localhost:8501
echo.
echo Press any key to stop all services...
pause >nul

echo.
echo Stopping services...
taskkill /FI "WINDOWTITLE eq Backend API*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Frontend Dashboard*" /T /F >nul 2>&1
echo Services stopped.
pause
