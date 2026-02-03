# Tamil Poetry Emotion-Rasa Classification System
# Quick Start Script

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "TAMIL POETRY EMOTION-RASA CLASSIFICATION SYSTEM" -ForegroundColor Yellow
Write-Host "Hackathon-Ready AI Project" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/5] Project structure:" -ForegroundColor Cyan
Write-Host "  emotion-rasa-ai/" -ForegroundColor White
Write-Host "  ├── backend/         (FastAPI + ML models)" -ForegroundColor Gray
Write-Host "  ├── frontend/        (Streamlit dashboard)" -ForegroundColor Gray
Write-Host "  ├── data/            (Tamil poetry dataset)" -ForegroundColor Gray
Write-Host "  ├── models/          (Saved model files)" -ForegroundColor Gray
Write-Host "  └── requirements.txt (Dependencies)" -ForegroundColor Gray

Write-Host ""
Write-Host "[3/5] Quick Start Commands:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Install dependencies:" -ForegroundColor Yellow
Write-Host "     cd emotion-rasa-ai" -ForegroundColor White
Write-Host "     pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "  2. Train the model:" -ForegroundColor Yellow
Write-Host "     python backend/train.py" -ForegroundColor White
Write-Host "     (Takes 10-30 min, requires GPU recommended)" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Start the API server:" -ForegroundColor Yellow
Write-Host "     python backend/app.py" -ForegroundColor White
Write-Host "     (Access at http://localhost:8000)" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Launch the dashboard:" -ForegroundColor Yellow
Write-Host "     streamlit run frontend/dashboard.py" -ForegroundColor White
Write-Host "     (Opens in browser at http://localhost:8501)" -ForegroundColor Gray

Write-Host ""
Write-Host "[4/5] Features:" -ForegroundColor Cyan
Write-Host "  [OK] IndicBERT fine-tuned on Tamil poetry" -ForegroundColor Green
Write-Host "  [OK] 10 Navarasa + Bhakti categories" -ForegroundColor Green
Write-Host "  [OK] Interactive dashboard with visualizations" -ForegroundColor Green
Write-Host "  [OK] REST API for integration" -ForegroundColor Green
Write-Host "  [OK] Attention heatmaps and explainability" -ForegroundColor Green
Write-Host "  [OK] Confusion matrix and metrics" -ForegroundColor Green

Write-Host ""
Write-Host "[5/5] Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Navigate to project: cd emotion-rasa-ai" -ForegroundColor White
Write-Host "  2. Install packages: pip install -r requirements.txt" -ForegroundColor White
Write-Host "  3. Train model: python backend/train.py" -ForegroundColor White
Write-Host "  4. Run dashboard: streamlit run frontend/dashboard.py" -ForegroundColor White

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "Ready to classify Tamil poetry emotions! 🎭" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host ""

Write-Host "For detailed documentation, see: README.md" -ForegroundColor Gray
Write-Host ""
