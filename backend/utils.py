"""
Utility Functions for Emotion-Rasa Classification System
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path


def ensure_dir(directory):
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory (str): Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_pickle(obj, filepath):
    """
    Save object as pickle file
    
    Args:
        obj: Object to save
        filepath (str): Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved pickle to: {filepath}")


def load_pickle(filepath):
    """
    Load object from pickle file
    
    Args:
        filepath (str): Path to pickle file
        
    Returns:
        object: Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj, filepath):
    """
    Save object as JSON file
    
    Args:
        obj: Object to save (must be JSON serializable)
        filepath (str): Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to: {filepath}")


def load_json(filepath):
    """
    Load object from JSON file
    
    Args:
        filepath (str): Path to JSON file
        
    Returns:
        object: Loaded object
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_prediction_output(emotion, rasa, confidence, probabilities):
    """
    Format prediction results for display
    
    Args:
        emotion (str): Predicted emotion
        rasa (str): Mapped rasa
        confidence (float): Confidence score
        probabilities (dict): All class probabilities
        
    Returns:
        dict: Formatted output
    """
    return {
        'emotion': emotion,
        'rasa': rasa,
        'confidence': float(confidence),
        'probabilities': {k: float(v) for k, v in probabilities.items()},
    }


def get_top_k_predictions(probabilities, labels, k=3):
    """
    Get top K predictions with probabilities
    
    Args:
        probabilities (array): Probability array
        labels (list): List of label names
        k (int): Number of top predictions
        
    Returns:
        list: List of (label, probability) tuples
    """
    top_k_idx = np.argsort(probabilities)[-k:][::-1]
    return [(labels[idx], probabilities[idx]) for idx in top_k_idx]


def calculate_confidence_level(probability):
    """
    Categorize confidence level
    
    Args:
        probability (float): Probability score
        
    Returns:
        str: Confidence level (High/Medium/Low)
    """
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def get_model_path(model_name="emotion_model"):
    """
    Get full path to model directory
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: Full path to model
    """
    base_dir = Path(__file__).parent.parent
    return os.path.join(base_dir, 'models', model_name)


def get_data_path(filename):
    """
    Get full path to data file
    
    Args:
        filename (str): Name of data file
        
    Returns:
        str: Full path to data file
    """
    base_dir = Path(__file__).parent.parent
    return os.path.join(base_dir, 'data', filename)


if __name__ == "__main__":
    # Test utilities
    print("Model path:", get_model_path())
    print("Data path:", get_data_path("primary_emotions.csv"))
    
    # Test confidence levels
    for prob in [0.95, 0.65, 0.45, 0.25]:
        print(f"Probability {prob:.2f} → Confidence: {calculate_confidence_level(prob)}")
