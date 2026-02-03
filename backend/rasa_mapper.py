"""
Emotion to Navarasa (+ Bhakti) Mapping Module
Maps primary emotions to Indian aesthetic emotion categories
"""

# Emotion to Rasa mapping dictionary
EMOTION_TO_RASA = {
    'Love': 'Shringara',
    'Joy': 'Hasya',
    'Sorrow': 'Karuna',
    'Anger': 'Raudra',
    'Courage': 'Veera',
    'Pride': 'Veera',
    'Confidence': 'Veera',
    'Fear': 'Bhayanaka',
    'Caution': 'Bhayanaka',
    'Disgust': 'Bibhatsa',
    'Betrayal': 'Bibhatsa',
    'Wonder': 'Adbhuta',
    'Clarity': 'Adbhuta',
    'Wisdom': 'Adbhuta',
    'Calmness': 'Shanta',
    'Contentment': 'Shanta',
    'Reverence': 'Bhakti',
    'Devotion': 'Bhakti',
    'Gratitude': 'Bhakti',
}

# Rasa descriptions for UI
RASA_DESCRIPTIONS = {
    'Shringara': 'Love, Beauty, Attraction (शृङ्गार)',
    'Hasya': 'Joy, Laughter, Humor (हास्य)',
    'Karuna': 'Sorrow, Compassion, Pathos (करुण)',
    'Raudra': 'Anger, Fury, Rage (रौद्र)',
    'Veera': 'Courage, Heroism, Pride (वीर)',
    'Bhayanaka': 'Fear, Terror, Anxiety (भयानक)',
    'Bibhatsa': 'Disgust, Aversion, Repulsion (बीभत्स)',
    'Adbhuta': 'Wonder, Amazement, Curiosity (अद्भुत)',
    'Shanta': 'Peace, Calmness, Serenity (शान्त)',
    'Bhakti': 'Devotion, Reverence, Faith (भक्ति)',
}

# Color scheme for visualization
RASA_COLORS = {
    'Shringara': '#FF69B4',  # Hot Pink
    'Hasya': '#FFD700',      # Gold
    'Karuna': '#4169E1',     # Royal Blue
    'Raudra': '#DC143C',     # Crimson
    'Veera': '#FF8C00',      # Dark Orange
    'Bhayanaka': '#8B008B',  # Dark Magenta
    'Bibhatsa': '#556B2F',   # Dark Olive Green
    'Adbhuta': '#00CED1',    # Dark Turquoise
    'Shanta': '#98FB98',     # Pale Green
    'Bhakti': '#DDA0DD',     # Plum
}


def map_to_rasa(emotion_label):
    """
    Map primary emotion to Navarasa category
    
    Args:
        emotion_label (str): Primary emotion label
        
    Returns:
        str: Corresponding Rasa category
    """
    return EMOTION_TO_RASA.get(emotion_label, 'Shanta')


def get_rasa_description(rasa):
    """
    Get description for a Rasa
    
    Args:
        rasa (str): Rasa name
        
    Returns:
        str: Description of the Rasa
    """
    return RASA_DESCRIPTIONS.get(rasa, 'Unknown Rasa')


def get_rasa_color(rasa):
    """
    Get color code for a Rasa
    
    Args:
        rasa (str): Rasa name
        
    Returns:
        str: Hex color code
    """
    return RASA_COLORS.get(rasa, '#808080')


def get_all_rasas():
    """
    Get list of all Rasa categories
    
    Returns:
        list: List of Rasa names
    """
    return list(RASA_DESCRIPTIONS.keys())


def get_emotion_to_rasa_pairs():
    """
    Get all emotion-rasa mapping pairs
    
    Returns:
        dict: Emotion to Rasa mapping
    """
    return EMOTION_TO_RASA.copy()


if __name__ == "__main__":
    # Test mappings
    test_emotions = ['Love', 'Joy', 'Sorrow', 'Anger', 'Devotion']
    print("Emotion → Rasa Mapping:")
    for emotion in test_emotions:
        rasa = map_to_rasa(emotion)
        description = get_rasa_description(rasa)
        color = get_rasa_color(rasa)
        print(f"{emotion:15} → {rasa:12} | {description:40} | {color}")
