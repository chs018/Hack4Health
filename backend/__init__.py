"""
Backend package for Tamil Poetry Emotion-Rasa Classification
"""

__version__ = "1.0.0"
__author__ = "AI + Full Stack Engineer"

# Import key components for easy access
from .model import EmotionPredictor, load_model
from .rasa_mapper import map_to_rasa, get_rasa_description
from .preprocess import normalize_text, clean_text

__all__ = [
    'EmotionPredictor',
    'load_model',
    'map_to_rasa',
    'get_rasa_description',
    'normalize_text',
    'clean_text'
]
