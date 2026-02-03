"""
FastAPI Backend for Tamil Poetry Emotion Classification
REST API for real-time emotion prediction and Rasa mapping
"""

import os
import sys
import time
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model import EmotionPredictor
from backend.rasa_mapper import map_to_rasa, get_rasa_description, get_rasa_color
from backend.explain import create_highlighted_tokens, explain_prediction
from backend.utils import calculate_confidence_level
from backend.database import (
    init_db, get_db_session, PredictionHistory, UserSession, UserFeedback,
    get_prediction_stats, get_emotion_distribution, get_rasa_distribution,
    get_recent_predictions
)
from backend.config import API_HOST, API_PORT, CORS_ORIGINS


# Initialize FastAPI app
app = FastAPI(
    title="Tamil Poetry Emotion-Rasa Classifier API",
    description="NLP API for classifying emotions in Tamil poetry and mapping to Navarasa categories",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "காதல் என்பது உயிரினும் இனிது",
                "session_id": "optional-session-id"
            }
        }


class PredictionResponse(BaseModel):
    prediction_id: int
    session_id: str
    text: str
    cleaned_text: str
    emotion: str
    rasa: str
    rasa_description: str
    rasa_color: str
    confidence: float
    confidence_level: str
    probabilities: Dict[str, float]
    highlighted_tokens: List[Dict]
    explanation: str
    top_tokens: List[tuple]
    processing_time: float


class FeedbackRequest(BaseModel):
    prediction_id: int
    session_id: str
    rating: Optional[int] = None
    correct_emotion: Optional[str] = None
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Load model and initialize database on startup"""
    global predictor
    
    # Initialize database
    try:
        print("Initializing database...")
        init_db()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Database initialization warning: {str(e)}")
    
    # Load model
    try:
        print("Loading emotion classification model...")
        predictor = EmotionPredictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Make sure to train the model first: python backend/train.py")
        predictor = None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "online",
        "message": "Tamil Poetry Emotion-Rasa Classifier API",
        "model_loaded": predictor is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if predictor is None:
        return {
            "status": "error",
            "message": "Model not loaded. Please train the model first.",
            "model_loaded": False
        }
    
    return {
        "status": "healthy",
        "message": "API is running and model is loaded",
        "model_loaded": True
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest, db: Session = Depends(get_db_session)):
    """
    Predict emotion for Tamil poetry text
    
    Args:
        request: PredictionRequest with text field
        db: Database session
        
    Returns:
        PredictionResponse with emotion, rasa, and explainability data
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using: python backend/train.py"
        )
    
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text field cannot be empty"
        )
    
    try:
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Update or create session
        session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        if not session:
            session = UserSession(session_id=session_id)
            db.add(session)
        session.last_active = datetime.utcnow()
        session.total_predictions += 1
        
        # Start timing
        start_time = time.time()
        
        # Get prediction
        result = predictor.predict(request.text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Map to Rasa
        emotion = result['predicted_emotion']
        rasa = map_to_rasa(emotion)
        rasa_description = get_rasa_description(rasa)
        rasa_color = get_rasa_color(rasa)
        
        # Get confidence level
        confidence_level = calculate_confidence_level(result['confidence'])
        
        # Create highlighted tokens
        highlighted_tokens = []
        if result['attention_weights'] and result['tokens']:
            highlighted_tokens = create_highlighted_tokens(
                result['tokens'],
                result['attention_weights']
            )
        
        # Generate explanation
        explanation_data = explain_prediction(
            result['tokens'],
            result['attention_weights'] or [],
            emotion,
            result['confidence']
        )
        
        # Store prediction in database
        prediction = PredictionHistory(
            session_id=session_id,
            text=request.text,
            predicted_emotion=emotion,
            predicted_rasa=rasa,
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            processing_time=processing_time
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return {
            "prediction_id": prediction.id,
            "session_id": session_id,
            "text": result['text'],
            "cleaned_text": result['cleaned_text'],
            "emotion": emotion,
            "rasa": rasa,
            "rasa_description": rasa_description,
            "rasa_color": rasa_color,
            "confidence": result['confidence'],
            "confidence_level": confidence_level,
            "probabilities": result['probabilities'],
            "highlighted_tokens": highlighted_tokens,
            "explanation": explanation_data['explanation'],
            "top_tokens": explanation_data['top_tokens'],
            "processing_time": processing_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/emotions")
async def get_emotions():
    """Get list of all emotion labels"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "emotions": list(predictor.label_encoder.classes_)
    }


@app.get("/rasas")
async def get_rasas():
    """Get list of all Rasa categories"""
    from backend.rasa_mapper import RASA_DESCRIPTIONS, RASA_COLORS
    
    rasas = []
    for rasa, description in RASA_DESCRIPTIONS.items():
        rasas.append({
            "name": rasa,
            "description": description,
            "color": RASA_COLORS[rasa]
        })
    
    return {"rasas": rasas}


@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "IndicBERT",
        "num_classes": len(predictor.label_encoder.classes_),
        "classes": list(predictor.label_encoder.classes_),
        "device": str(predictor.device)
    }


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db_session)):
    """Submit feedback for a prediction"""
    try:
        user_feedback = UserFeedback(
            prediction_id=feedback.prediction_id,
            session_id=feedback.session_id,
            rating=feedback.rating,
            correct_emotion=feedback.correct_emotion,
            comment=feedback.comment
        )
        db.add(user_feedback)
        db.commit()
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 20, db: Session = Depends(get_db_session)):
    """Get prediction history for a session"""
    try:
        predictions = db.query(PredictionHistory).filter(
            PredictionHistory.session_id == session_id
        ).order_by(PredictionHistory.timestamp.desc()).limit(limit).all()
        
        return {
            "session_id": session_id,
            "count": len(predictions),
            "predictions": [
                {
                    "id": p.id,
                    "text": p.text,
                    "emotion": p.predicted_emotion,
                    "rasa": p.predicted_rasa,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp.isoformat()
                }
                for p in predictions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.get("/analytics/stats")
async def get_stats(db: Session = Depends(get_db_session)):
    """Get overall statistics"""
    try:
        stats = get_prediction_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/analytics/emotions")
async def get_emotion_stats(db: Session = Depends(get_db_session)):
    """Get emotion distribution"""
    try:
        distribution = get_emotion_distribution(db)
        return {"distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get emotion stats: {str(e)}")


@app.get("/analytics/rasas")
async def get_rasa_stats(db: Session = Depends(get_db_session)):
    """Get rasa distribution"""
    try:
        distribution = get_rasa_distribution(db)
        return {"distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rasa stats: {str(e)}")


@app.get("/analytics/recent")
async def get_recent(limit: int = 10, db: Session = Depends(get_db_session)):
    """Get recent predictions across all sessions"""
    try:
        predictions = get_recent_predictions(db, limit)
        return {
            "count": len(predictions),
            "predictions": [
                {
                    "id": p.id,
                    "text": p.text[:50] + "..." if len(p.text) > 50 else p.text,
                    "emotion": p.predicted_emotion,
                    "rasa": p.predicted_rasa,
                    "confidence": p.confidence,
                    "timestamp": p.timestamp.isoformat()
                }
                for p in predictions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent predictions: {str(e)}")


def start_server(host=None, port=None):
    """Start the FastAPI server"""
    if host is None:
        host = API_HOST
    if port is None:
        port = API_PORT
    
    print("=" * 80)
    print("TAMIL POETRY EMOTION-RASA CLASSIFIER API")
    print("=" * 80)
    print(f"\nStarting server on http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print(f"Analytics: http://{host}:{port}/analytics/stats")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
