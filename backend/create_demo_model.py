"""
Quick Demo Model Setup
Creates a minimal working model for testing the application
"""

import os
import sys
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_demo_model():
    """Create a minimal demo model for testing"""
    print("Creating demo model for testing...")
    
    # Model directory
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'emotion_model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Try to use a small multilingual model
    model_name = "bert-base-multilingual-cased"
    
    print(f"Downloading {model_name}...")
    print("This may take a few minutes on first run...")
    
    try:
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create model with 19 emotion classes
        emotions = ['Anger', 'Betrayal', 'Calmness', 'Caution', 'Clarity', 'Confidence',
                   'Contentment', 'Courage', 'Devotion', 'Disgust', 'Fear', 'Gratitude',
                   'Joy', 'Love', 'Pride', 'Reverence', 'Sorrow', 'Wisdom', 'Wonder']
        
        num_labels = len(emotions)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # Create and save label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(emotions)
        
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"✓ Demo model created successfully at: {model_dir}")
        print(f"✓ Model: {model_name}")
        print(f"✓ Number of emotions: {num_labels}")
        print("\nNote: This is an untrained demo model.")
        print("Predictions will be random until you train with your data.")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to create demo model: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try running again (downloads may timeout)")
        print("3. Consider using a VPN if HuggingFace is blocked")
        return False


if __name__ == "__main__":
    success = create_demo_model()
    if success:
        print("\n" + "="*80)
        print("READY TO USE!")
        print("="*80)
        print("\nRestart your backend server:")
        print("  python -c \"import sys; sys.path.append('.'); from backend import app; app.start_server()\"")
    else:
        print("\nPlease resolve the issues above and try again.")
