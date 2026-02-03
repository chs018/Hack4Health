"""
Model Inference Module
Loads trained model and makes predictions with attention weights
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocess import normalize_text
from backend.utils import load_pickle, get_model_path


class EmotionPredictor:
    """Emotion prediction class with attention weights extraction"""
    
    def __init__(self, model_path=None):
        """
        Initialize predictor
        
        Args:
            model_path (str): Path to saved model directory
        """
        if model_path is None:
            model_path = get_model_path('emotion_model')
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            output_attentions=True  # Enable attention outputs
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
        self.label_encoder = load_pickle(label_encoder_path)
        
        print(f"Model loaded successfully!")
        print(f"Classes: {list(self.label_encoder.classes_)}")
    
    def predict(self, text):
        """
        Predict emotion for given text
        
        Args:
            text (str): Input Tamil text
            
        Returns:
            dict: Prediction results with probabilities and attention weights
        """
        # Preprocess text
        cleaned_text = normalize_text(text)
        
        if not cleaned_text:
            return {
                'error': 'Text is empty after preprocessing',
                'original_text': text
            }
        
        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get tokens for attention visualization
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get probabilities
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        probs_np = probabilities.cpu().numpy()
        
        # Get predicted class
        predicted_idx = torch.argmax(probabilities).item()
        predicted_label = self.label_encoder.classes_[predicted_idx]
        confidence = probs_np[predicted_idx]
        
        # Get all class probabilities
        all_probs = {
            label: float(probs_np[idx])
            for idx, label in enumerate(self.label_encoder.classes_)
        }
        
        # Extract attention weights (average across all layers and heads)
        attention_weights = None
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Average attention across all layers and heads
            # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            attentions = torch.stack(outputs.attentions)  # Stack all layers
            # Average across layers and heads: (batch_size, seq_len, seq_len)
            avg_attention = attentions.mean(dim=(0, 2))
            # Get attention from [CLS] token to all other tokens
            cls_attention = avg_attention[0, 0, :].cpu().numpy()
            attention_weights = cls_attention.tolist()
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'predicted_emotion': predicted_label,
            'confidence': float(confidence),
            'probabilities': all_probs,
            'tokens': tokens,
            'attention_weights': attention_weights
        }
    
    def predict_batch(self, texts):
        """
        Predict emotions for multiple texts
        
        Args:
            texts (list): List of input texts
            
        Returns:
            list: List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def get_top_k_predictions(self, text, k=3):
        """
        Get top K predictions
        
        Args:
            text (str): Input text
            k (int): Number of top predictions
            
        Returns:
            list: Top K predictions with probabilities
        """
        result = self.predict(text)
        
        if 'error' in result:
            return result
        
        probs = result['probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'text': text,
            'top_predictions': sorted_probs[:k]
        }


def load_model(model_path=None):
    """
    Load trained model for inference
    
    Args:
        model_path (str): Path to model directory
        
    Returns:
        EmotionPredictor: Loaded predictor instance
    """
    return EmotionPredictor(model_path)


if __name__ == "__main__":
    # Test the model
    print("Testing Emotion Predictor...")
    
    try:
        predictor = load_model()
        
        # Test samples
        test_texts = [
            "காதல் என்பது உயிரினும் இனிது",
            "மகிழ்ச்சி பெருகுது நெஞ்சில்",
            "துயரம் நிறைந்த என் உள்ளம்",
        ]
        
        print("\nPredictions:")
        print("=" * 80)
        
        for text in test_texts:
            result = predictor.predict(text)
            print(f"\nText: {text}")
            print(f"Predicted: {result['predicted_emotion']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Top 3 probabilities:")
            sorted_probs = sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for emotion, prob in sorted_probs:
                print(f"  {emotion:15} : {prob:.4f}")
    
    except FileNotFoundError:
        print("\n✗ Model not found!")
        print("Please train the model first using: python backend/train.py")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
