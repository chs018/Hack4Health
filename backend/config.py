"""
Configuration Module
Centralized configuration for the application
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/emotion_rasa.db")

# API configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")

# Frontend configuration
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

# Model configuration
MODEL_PATH = MODELS_DIR / "emotion_model"
MAX_LENGTH = 128
BATCH_SIZE = 16

# Session configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# CORS configuration
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:8501",
    "http://localhost:3000",
    "*"  # Allow all for development
]

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
