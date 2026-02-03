"""
Database Models and Configuration
SQLAlchemy models for storing predictions, sessions, and user feedback
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emotion_rasa.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Database Models
class PredictionHistory(Base):
    """Store prediction history"""
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True)
    text = Column(Text, nullable=False)
    predicted_emotion = Column(String(50), nullable=False)
    predicted_rasa = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)  # Store all emotion probabilities
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Float)  # Time taken for prediction in seconds


class UserSession(Base):
    """Track user sessions"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    total_predictions = Column(Integer, default=0)


class UserFeedback(Base):
    """Store user feedback on predictions"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, index=True)
    session_id = Column(String(100), index=True)
    rating = Column(Integer)  # 1-5 rating
    correct_emotion = Column(String(50))  # User's correction if prediction was wrong
    comment = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SystemMetrics(Base):
    """Store system performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_predictions = Column(Integer, default=0)
    average_confidence = Column(Float)
    average_processing_time = Column(Float)
    most_common_emotion = Column(String(50))
    most_common_rasa = Column(String(50))


# Database utility functions
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")


@contextmanager
def get_db():
    """Get database session with context manager"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get database session (for FastAPI dependency injection)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Analytics functions
def get_prediction_stats(db):
    """Get prediction statistics"""
    from sqlalchemy import func
    
    stats = db.query(
        func.count(PredictionHistory.id).label('total_predictions'),
        func.avg(PredictionHistory.confidence).label('avg_confidence'),
        func.avg(PredictionHistory.processing_time).label('avg_processing_time'),
    ).first()
    
    return {
        'total_predictions': stats.total_predictions or 0,
        'avg_confidence': round(stats.avg_confidence or 0, 4),
        'avg_processing_time': round(stats.avg_processing_time or 0, 4),
    }


def get_emotion_distribution(db, limit=10):
    """Get emotion distribution"""
    from sqlalchemy import func
    
    emotions = db.query(
        PredictionHistory.predicted_emotion,
        func.count(PredictionHistory.id).label('count')
    ).group_by(
        PredictionHistory.predicted_emotion
    ).order_by(
        func.count(PredictionHistory.id).desc()
    ).limit(limit).all()
    
    return [{'emotion': e[0], 'count': e[1]} for e in emotions]


def get_rasa_distribution(db, limit=10):
    """Get rasa distribution"""
    from sqlalchemy import func
    
    rasas = db.query(
        PredictionHistory.predicted_rasa,
        func.count(PredictionHistory.id).label('count')
    ).group_by(
        PredictionHistory.predicted_rasa
    ).order_by(
        func.count(PredictionHistory.id).desc()
    ).limit(limit).all()
    
    return [{'rasa': r[0], 'count': r[1]} for r in rasas]


def get_recent_predictions(db, limit=20):
    """Get recent predictions"""
    predictions = db.query(PredictionHistory).order_by(
        PredictionHistory.timestamp.desc()
    ).limit(limit).all()
    
    return predictions


if __name__ == "__main__":
    # Initialize database when run directly
    print("Initializing database...")
    init_db()
    print("Database ready!")
