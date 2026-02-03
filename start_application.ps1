# Tamil Poetry Emotion-Rasa Classification System
# Complete Application Startup Script (PowerShell)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Tamil Poetry Emotion-Rasa Classification System" -ForegroundColor Yellow
Write-Host "Complete Application Startup" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "[1/3] Starting Backend API Server..." -ForegroundColor Cyan
$BackendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python -c "import sys; sys.path.append('.'); from backend import app; app.start_server()"
} -ArgumentList $ScriptDir

Start-Sleep -Seconds 5

Write-Host "[2/3] Database will be initialized automatically..." -ForegroundColor Cyan
Start-Sleep -Seconds 2

Write-Host "[3/3] Starting Frontend Dashboard..." -ForegroundColor Cyan
$FrontendJob = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    streamlit run frontend/dashboard.py --server.port 8501
} -ArgumentList $ScriptDir

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "APPLICATION READY!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend API:        " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000" -ForegroundColor Yellow
Write-Host "API Documentation:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "Frontend Dashboard: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Database:           SQLite (emotion_rasa.db)" -ForegroundColor Gray
Write-Host ""

Write-Host "Opening frontend in browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor Red
Write-Host ""

# Keep script running
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job -Job $BackendJob -ErrorAction SilentlyContinue
    Stop-Job -Job $FrontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $BackendJob -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $FrontendJob -Force -ErrorAction SilentlyContinue
    Write-Host "Services stopped." -ForegroundColor Green
}
