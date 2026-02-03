"""
Tamil Text Preprocessing Module
Cleans and normalizes Tamil poetry text for emotion classification
"""

import re
import unicodedata


def clean_text(text):
    """
    Clean and normalize Tamil text
    
    Args:
        text (str): Raw Tamil text input
        
    Returns:
        str: Cleaned and normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Unicode normalization (NFC form for Tamil)
    text = unicodedata.normalize('NFC', text)
    
    # Remove English characters but keep Tamil
    # Tamil Unicode range: \u0B80-\u0BFF
    text = re.sub(r'[^\u0B80-\u0BFF\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Lowercase is not applicable to Tamil script, but we normalize
    
    return text


def remove_punctuation(text):
    """
    Remove punctuation marks from text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without punctuation
    """
    # Remove common punctuation
    punctuation = r'[।॥,.\-!?;:()\'\"।،؛]'
    text = re.sub(punctuation, ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_text(text):
    """
    Complete text normalization pipeline
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Fully preprocessed text
    """
    text = clean_text(text)
    text = remove_punctuation(text)
    return text


def preprocess_dataset(texts):
    """
    Preprocess a list of texts
    
    Args:
        texts (list): List of text strings
        
    Returns:
        list: List of preprocessed texts
    """
    return [normalize_text(text) for text in texts]


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "காதல் என்பது உயிரினும் இனிது, உன் சிரிப்பில் உலகமே மறக்கிறேன்"
    print("Original:", sample_text)
    print("Cleaned:", clean_text(sample_text))
    print("Normalized:", normalize_text(sample_text))
