# TROUBLESHOOTING: Backend API Not Running / Model Not Loaded

## 🔴 Current Issue
**Error**: "Backend API is not running or model is not loaded"

## 🔍 Root Cause
The model training process is failing because it cannot download the IndicBERT model from HuggingFace due to:
- Network timeout/connectivity issues
- Firewall or proxy blocking huggingface.co
- Slow internet connection

## ✅ Solutions (Choose One)

### Option 1: Fix Network and Retry Training (Recommended)
```powershell
# Check internet connection
Test-NetConnection huggingface.co -Port 443

# If blocked, try with VPN or different network

# Then retrain:
cd c:\health4HACK\emotion-rasa-ai
python backend\train.py
```

### Option 2: Use Pre-cached Model (If Available)
If you've downloaded models before, they may be cached:
```powershell
# Check HuggingFace cache
dir $env:USERPROFILE\.cache\huggingface\hub

# If ai4bharat--indic-bert exists, training should work offline
```

### Option 3: Manual Model Download
1. Go to: https://huggingface.co/ai4bharat/indic-bert
2. Click "Files and versions"
3. Download these files to `models\emotion_model\`:
   - config.json
   - pytorch_model.bin
   - tokenizer_config.json
   - vocab.txt
   - special_tokens_map.json

4. Place the downloaded `label_encoder.pkl` (already exists in models\emotion_model\)

### Option 4: Use Smaller Demo Model
If network issues persist, modify `backend\train.py`:

Change line 40 from:
```python
model_name = "ai4bharat/indic-bert"
```

To:
```python
model_name = "bert-base-multilingual-cased"  # Smaller, more likely to download
```

Then retry training.

### Option 5: Train on Google Colab (Network-free locally)
1. Upload your data to Google Colab
2. Train there (free GPU!)
3. Download the trained model files
4. Place them in `c:\health4HACK\emotion-rasa-ai\models\emotion_model\`

## 📋 Required Files for Model to Work

Your `models\emotion_model\` directory needs:
```
emotion_model/
├── config.json              ❌ MISSING
├── pytorch_model.bin        ❌ MISSING  
├── tokenizer_config.json    ❌ MISSING
├── vocab.txt                ❌ MISSING
├── special_tokens_map.json  ❌ MISSING
├── label_encoder.pkl        ✅ EXISTS
└── class_distribution.png   ✅ EXISTS
```

## 🚀 Quick Test Once Model is Ready

After fixing, restart everything:
```powershell
cd c:\health4HACK\emotion-rasa-ai
.\start_application.ps1
```

## 📞 Current Status

✅ Backend API is running on http://localhost:8000
✅ Frontend is running on http://localhost:8501
✅ Database is initialized
❌ Model files are incomplete - training failed due to network timeout

## 💡 Recommended Action

1. **Check your internet connection**
2. **Try downloading manually from HuggingFace** (Option 3 above)
3. **Or wait for better network and retry training**

The application architecture is working correctly - we just need the model files!
