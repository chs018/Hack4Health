# Project Architecture

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐      ┌──────────────────────┐      │
│  │  Streamlit Dashboard│      │   REST API Clients   │      │
│  │  (dashboard.py)     │      │   (External Apps)    │      │
│  └──────────┬──────────┘      └──────────┬───────────┘      │
│             │                             │                   │
└─────────────┼─────────────────────────────┼──────────────────┘
              │                             │
              ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    API / SERVICE LAYER                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │          FastAPI Backend (app.py)                  │     │
│  │  ┌──────────────────────────────────────────────┐  │     │
│  │  │  Endpoints:                                   │  │     │
│  │  │  • POST /predict   - Emotion classification  │  │     │
│  │  │  • GET /health     - Health check            │  │     │
│  │  │  • GET /emotions   - List emotions           │  │     │
│  │  │  • GET /rasas      - List Navarasa           │  │     │
│  │  └──────────────────────────────────────────────┘  │     │
│  └─────────────────────┬──────────────────────────────┘     │
│                        │                                      │
└────────────────────────┼──────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   BUSINESS LOGIC LAYER                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐    │
│  │  Model Inference│  │  Rasa Mapping   │  │ Explain  │    │
│  │  (model.py)     │  │  (rasa_mapper)  │  │(explain) │    │
│  │                 │  │                 │  │          │    │
│  │  • Load model   │  │  • Emotion→Rasa │  │• Attention│   │
│  │  • Predict      │  │  • Get colors   │  │• Heatmaps │   │
│  │  • Get attention│  │  • Descriptions │  │• Top tokens│  │
│  └────────┬────────┘  └────────┬────────┘  └────┬─────┘    │
│           │                     │                 │          │
│           └─────────────────────┴─────────────────┘          │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │         IndicBERT Fine-tuned Model                 │     │
│  │  ┌──────────────────────────────────────────────┐  │     │
│  │  │  Architecture:                                │  │     │
│  │  │  • Input: Tamil text tokens                  │  │     │
│  │  │  • Encoder: 12-layer Transformer             │  │     │
│  │  │  • Output: 10-class emotion logits           │  │     │
│  │  │  • Attention: Multi-head self-attention      │  │     │
│  │  └──────────────────────────────────────────────┘  │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA PROCESSING LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐    │
│  │  Preprocessing  │  │  Tokenization   │  │  Utils   │    │
│  │  (preprocess.py)│  │  (Tokenizer)    │  │(utils.py)│    │
│  │                 │  │                 │  │          │    │
│  │  • Clean text   │  │  • IndicBERT    │  │• Helpers │    │
│  │  • Normalize    │  │    tokenizer    │  │• I/O     │    │
│  │  • Unicode      │  │  • Subword      │  │• Metrics │    │
│  └────────┬────────┘  └────────┬────────┘  └────┬─────┘    │
│           │                     │                 │          │
└───────────┼─────────────────────┼─────────────────┼──────────┘
            │                     │                 │
            ▼                     ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────┐  ┌──────────────────────────┐       │
│  │ Training Dataset   │  │  Trained Model Files     │       │
│  │ primary_emotions.csv│  │  models/emotion_model/   │       │
│  │                    │  │                          │       │
│  │ • Tamil verses     │  │  • pytorch_model.bin     │       │
│  │ • Emotion labels   │  │  • config.json           │       │
│  │ • 40 samples       │  │  • tokenizer files       │       │
│  │                    │  │  • label_encoder.pkl     │       │
│  └────────────────────┘  └──────────────────────────┘       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### Training Pipeline

```
1. Load CSV Data
   ↓
2. Preprocess Tamil Text
   (Clean, Normalize, Remove punctuation)
   ↓
3. Encode Labels
   (LabelEncoder: Emotion → Integer)
   ↓
4. Train/Test Split (80/20)
   ↓
5. Tokenize with IndicBERT
   (Text → Token IDs)
   ↓
6. Fine-tune Model
   (10 epochs, early stopping)
   ↓
7. Evaluate & Save
   (Metrics, Confusion Matrix, Model files)
```

### Prediction Pipeline

```
User Input (Tamil Verse)
   ↓
1. Preprocess Text
   (clean_text, normalize)
   ↓
2. Tokenize
   (IndicBERT tokenizer)
   ↓
3. Model Forward Pass
   (Get logits & attention weights)
   ↓
4. Softmax → Probabilities
   ↓
5. Get Predicted Class
   (argmax of probabilities)
   ↓
6. Map to Rasa
   (Emotion → Navarasa category)
   ↓
7. Extract Attention
   (Token importance weights)
   ↓
8. Generate Explanation
   (Top tokens, highlighted text)
   ↓
Return Results to User
```

## 📦 Component Details

### Backend Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `app.py` | FastAPI REST API | `predict_emotion()`, `health_check()` |
| `train.py` | Model training pipeline | `train_model()`, `plot_confusion_matrix()` |
| `model.py` | Inference engine | `EmotionPredictor.predict()` |
| `preprocess.py` | Text cleaning | `normalize_text()`, `clean_text()` |
| `rasa_mapper.py` | Emotion mapping | `map_to_rasa()`, `get_rasa_color()` |
| `explain.py` | Explainability | `create_highlighted_tokens()` |
| `utils.py` | Helper functions | `save_pickle()`, `get_model_path()` |

### Frontend Components

| Component | Purpose | Features |
|-----------|---------|----------|
| `dashboard.py` | Streamlit UI | Input, visualization, explanation |
| Input Section | Text entry | Samples, custom input |
| Results Section | Prediction display | Emotion, Rasa, confidence |
| Visualization | Charts | Gauge, bars, heatmap |
| Explanation | Model reasoning | Top tokens, probabilities |

## 🎯 ML Model Architecture

```
Input: Tamil Text
   ↓
Tokenizer (IndicBERT)
   ↓
[CLS] token_1 token_2 ... token_n [SEP]
   ↓
┌─────────────────────────────────┐
│   IndicBERT Encoder (12 layers) │
│                                 │
│  Layer 1:  Multi-Head Attention │
│           ↓                     │
│           Feed Forward          │
│           ↓                     │
│  Layer 2:  Multi-Head Attention │
│           ↓                     │
│           ...                   │
│           ↓                     │
│  Layer 12: Multi-Head Attention │
│           ↓                     │
│           Feed Forward          │
└────────────┬────────────────────┘
             ↓
    [CLS] embedding (768-dim)
             ↓
    Classification Head
    (Linear: 768 → 10)
             ↓
    Softmax
             ↓
    10 Emotion Probabilities
```

## 🗂️ File Structure with Descriptions

```
emotion-rasa-ai/
│
├── backend/                      # Backend Python modules
│   ├── __init__.py               # Package initialization
│   ├── app.py                    # FastAPI REST API (189 lines)
│   ├── train.py                  # Training pipeline (316 lines)
│   ├── model.py                  # Inference engine (179 lines)
│   ├── preprocess.py             # Text preprocessing (89 lines)
│   ├── rasa_mapper.py            # Emotion-Rasa mapping (102 lines)
│   ├── explain.py                # Explainability (215 lines)
│   └── utils.py                  # Utilities (130 lines)
│
├── frontend/                     # Frontend Streamlit app
│   ├── __init__.py               # Package initialization
│   └── dashboard.py              # Interactive dashboard (468 lines)
│
├── data/                         # Dataset directory
│   └── primary_emotions.csv      # Tamil poetry dataset (40 samples)
│
├── models/                       # Model storage (created after training)
│   └── emotion_model/            # Trained model files
│       ├── pytorch_model.bin     # Model weights (~500MB)
│       ├── config.json           # Model configuration
│       ├── tokenizer_config.json # Tokenizer settings
│       ├── vocab.txt             # Vocabulary
│       ├── label_encoder.pkl     # Label encoder
│       ├── confusion_matrix.png  # Performance visualization
│       └── class_distribution.png# Data distribution plot
│
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
├── INSTALLATION.md               # Setup guide
├── ARCHITECTURE.md               # This file
├── config.json                   # Configuration settings
├── .gitignore                    # Git ignore rules
├── test_setup.py                 # Setup verification script
├── run.bat                       # Windows launcher
└── START_HERE.ps1                # PowerShell info script
```

## 🔐 Security Considerations

1. **Input Validation**: All text inputs are validated and sanitized
2. **Model Safety**: No code execution in model predictions
3. **API Security**: CORS enabled, rate limiting recommended
4. **Data Privacy**: No data is stored permanently

## 🚀 Performance Optimization

1. **Model Caching**: Model loaded once and cached
2. **Batch Processing**: Support for multiple predictions
3. **GPU Acceleration**: Automatic GPU detection and usage
4. **Efficient Tokenization**: IndicBERT optimized tokenizer

## 📊 Monitoring & Logging

- API request/response logging
- Model prediction confidence tracking
- Error handling and reporting
- Performance metrics collection

## 🔄 Future Architecture Enhancements

1. **Microservices**: Separate training and inference services
2. **Database**: Add PostgreSQL for data persistence
3. **Caching**: Redis for frequent predictions
4. **Load Balancing**: Multiple API instances
5. **Monitoring**: Prometheus + Grafana dashboard
6. **CI/CD**: Automated testing and deployment
7. **Containerization**: Docker + Kubernetes deployment
