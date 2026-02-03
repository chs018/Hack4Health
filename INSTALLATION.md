# Installation & Setup Guide

## 🚀 Complete Setup Instructions

### Step 1: System Requirements

**Minimum Requirements:**
- Python 3.11 or higher
- 8GB RAM
- 5GB free disk space

**Recommended:**
- Python 3.11+
- 16GB RAM
- NVIDIA GPU with CUDA support (for faster training)
- 10GB free disk space

---

### Step 2: Install Python Dependencies

#### Option A: Using pip (Recommended)

```bash
# Navigate to project directory
cd emotion-rasa-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows CMD:
venv\Scripts\activate.bat

# On Linux/Mac:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

#### Option B: Using conda

```bash
# Create conda environment
conda create -n emotion-rasa python=3.11

# Activate environment
conda activate emotion-rasa

# Install PyTorch with CUDA (if you have GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

### Step 3: Verify Installation

Run the test script to verify everything is installed correctly:

```bash
python test_setup.py
```

**Expected Output:**
```
[1/6] Testing module imports...
✓ All modules imported successfully

[2/6] Testing Tamil text preprocessing...
✓ Preprocessing works

[3/6] Testing Rasa mapping...
✓ Rasa mapping works

[4/6] Testing utility functions...
✓ Utilities work

[5/6] Testing explainability functions...
✓ Explainability works

[6/6] Checking data file...
✓ Data file exists and is valid

✓ ALL TESTS PASSED!
```

---

### Step 4: Train the Model

```bash
python backend/train.py
```

**Training Progress:**
1. Loads dataset (40 samples)
2. Preprocesses Tamil text
3. Downloads IndicBERT model (~500MB)
4. Fine-tunes on emotion classification
5. Evaluates and saves model
6. Generates confusion matrix

**Time Required:**
- With GPU: 10-15 minutes
- With CPU: 30-60 minutes

**Output Files:**
- `models/emotion_model/` - Trained model
- `models/emotion_model/label_encoder.pkl` - Label encoder
- `models/emotion_model/confusion_matrix.png` - Confusion matrix
- `models/emotion_model/class_distribution.png` - Data distribution

---

### Step 5: Start the API Server

```bash
python backend/app.py
```

**API Endpoints:**
- Base: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

**Test API:**
```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"காதல் என்பது உயிரினும் இனிது\"}"

# Using Python
python -c "
import requests
response = requests.post(
    'http://localhost:8000/predict',
    json={'text': 'காதல் என்பது உயிரினும் இனிது'}
)
print(response.json())
"
```

---

### Step 6: Launch the Dashboard

Open a **new terminal** (keep API running) and execute:

```bash
# Activate virtual environment again
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Start Streamlit
streamlit run frontend/dashboard.py
```

**Dashboard URL:** http://localhost:8501

The dashboard will automatically open in your default browser.

---

## 🔧 Troubleshooting

### Issue: "No module named 'transformers'"

**Solution:**
```bash
pip install transformers torch
```

### Issue: "Model not found"

**Solution:**
Make sure you completed Step 4 (training). Check if `models/emotion_model/` exists.

### Issue: "CUDA out of memory"

**Solution:**
Edit `backend/train.py` and reduce batch size:
```python
train_model(batch_size=4)  # or even 2
```

### Issue: "Tamil text not displaying"

**Solution:**
- Install Tamil Unicode fonts
- Windows: Latha, Nirmala UI
- Mac: Install Tamil fonts from Font Book
- Linux: `sudo apt-get install fonts-tamil`

### Issue: Port 8000 or 8501 already in use

**Solution:**
```bash
# For API (change port in backend/app.py)
python backend/app.py --port 8001

# For Dashboard
streamlit run frontend/dashboard.py --server.port 8502
```

### Issue: Import errors in dashboard

**Solution:**
Make sure you're running from the correct directory:
```bash
cd emotion-rasa-ai
streamlit run frontend/dashboard.py
```

---

## 📦 Package Versions (Tested)

```
torch==2.1.0
transformers==4.35.0
fastapi==0.104.1
streamlit==1.28.1
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
plotly==5.18.0
```

---

## 🐳 Docker Setup (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["python", "backend/app.py"]
```

Build and run:
```bash
docker build -t emotion-rasa-ai .
docker run -p 8000:8000 -p 8501:8501 emotion-rasa-ai
```

---

## 🌐 Cloud Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Set main file: `frontend/dashboard.py`
5. Deploy!

### Deploy API to Heroku

```bash
# Create Procfile
echo "web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
heroku create emotion-rasa-api
git push heroku main
```

---

## 📱 Testing with Sample Data

```python
# Test with various emotions
test_samples = {
    "Love": "காதல் என்பது உயிரினும் இனிது",
    "Joy": "மகிழ்ச்சி பெருகுது நெஞ்சில்",
    "Sorrow": "துயரம் நிறைந்த என் உள்ளம்",
    "Devotion": "பக்தி பெருகுகிறது உள்ளத்தில்"
}

from backend.model import EmotionPredictor

predictor = EmotionPredictor()

for emotion, text in test_samples.items():
    result = predictor.predict(text)
    print(f"{emotion}: {result['predicted_emotion']} ({result['confidence']:.2%})")
```

---

## ✅ Final Checklist

- [ ] Python 3.11+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Test script passed (`python test_setup.py`)
- [ ] Model trained (`python backend/train.py`)
- [ ] API server running (`python backend/app.py`)
- [ ] Dashboard accessible (http://localhost:8501)
- [ ] Tested with sample Tamil verses

---

## 🎉 You're All Set!

Your Tamil Poetry Emotion-Rasa Classification system is ready to use!

**Next Steps:**
1. Try different Tamil verses in the dashboard
2. Explore the API documentation at http://localhost:8000/docs
3. Check model performance metrics
4. Customize and extend the system

**For more information, see README.md**
