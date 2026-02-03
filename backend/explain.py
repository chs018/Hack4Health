"""
Explainability Module
Creates attention heatmaps and token importance visualizations
"""

import numpy as np
from typing import List, Tuple


def normalize_attention_weights(attention_weights):
    """
    Normalize attention weights to [0, 1] range
    
    Args:
        attention_weights (list): Raw attention weights
        
    Returns:
        list: Normalized attention weights
    """
    if not attention_weights:
        return []
    
    weights = np.array(attention_weights)
    
    # Avoid division by zero
    min_val = weights.min()
    max_val = weights.max()
    
    if max_val - min_val < 1e-10:
        return [0.5] * len(weights)
    
    normalized = (weights - min_val) / (max_val - min_val)
    return normalized.tolist()


def get_color_for_weight(weight, colormap='red'):
    """
    Get RGB color based on attention weight
    
    Args:
        weight (float): Normalized weight [0, 1]
        colormap (str): Color scheme ('red', 'blue', 'green')
        
    Returns:
        str: RGB color string
    """
    # Clamp weight to [0, 1]
    weight = max(0.0, min(1.0, weight))
    
    if colormap == 'red':
        # White (low) to Red (high)
        r = int(255)
        g = int(255 * (1 - weight * 0.8))
        b = int(255 * (1 - weight * 0.8))
    elif colormap == 'blue':
        # White (low) to Blue (high)
        r = int(255 * (1 - weight * 0.8))
        g = int(255 * (1 - weight * 0.8))
        b = int(255)
    elif colormap == 'green':
        # White (low) to Green (high)
        r = int(255 * (1 - weight * 0.8))
        g = int(255)
        b = int(255 * (1 - weight * 0.8))
    else:
        # Default gradient
        r = int(255 * weight)
        g = int(255 * (1 - weight))
        b = 100
    
    return f"rgb({r}, {g}, {b})"


def create_highlighted_tokens(tokens, attention_weights, colormap='red'):
    """
    Create list of tokens with highlight colors based on attention
    
    Args:
        tokens (list): List of token strings
        attention_weights (list): Attention weights for each token
        colormap (str): Color scheme to use
        
    Returns:
        list: List of dicts with token and color information
    """
    if not tokens or not attention_weights:
        return []
    
    # Normalize weights
    normalized_weights = normalize_attention_weights(attention_weights)
    
    # Create highlighted tokens
    highlighted = []
    for token, weight in zip(tokens, normalized_weights):
        # Skip special tokens for better visualization
        if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            continue
        
        # Clean token (remove ## prefix from WordPiece)
        clean_token = token.replace('##', '')
        
        if not clean_token.strip():
            continue
        
        color = get_color_for_weight(weight, colormap)
        
        highlighted.append({
            'token': clean_token,
            'weight': float(weight),
            'color': color
        })
    
    return highlighted


def generate_html_heatmap(tokens, attention_weights):
    """
    Generate HTML string with colored tokens
    
    Args:
        tokens (list): List of tokens
        attention_weights (list): Attention weights
        
    Returns:
        str: HTML string with styled tokens
    """
    highlighted = create_highlighted_tokens(tokens, attention_weights)
    
    html_parts = ['<div style="line-height: 2.5; font-size: 18px;">']
    
    for item in highlighted:
        token = item['token']
        color = item['color']
        weight = item['weight']
        
        html_parts.append(
            f'<span style="background-color: {color}; '
            f'padding: 4px 6px; margin: 2px; border-radius: 4px; '
            f'display: inline-block;" '
            f'title="Attention: {weight:.3f}">{token}</span>'
        )
    
    html_parts.append('</div>')
    
    return ''.join(html_parts)


def get_top_k_important_tokens(tokens, attention_weights, k=5):
    """
    Get top K most important tokens based on attention
    
    Args:
        tokens (list): List of tokens
        attention_weights (list): Attention weights
        k (int): Number of top tokens to return
        
    Returns:
        list: Top K tokens with weights
    """
    if not tokens or not attention_weights:
        return []
    
    # Filter special tokens
    filtered = []
    for token, weight in zip(tokens, attention_weights):
        if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
            clean_token = token.replace('##', '')
            if clean_token.strip():
                filtered.append((clean_token, weight))
    
    # Sort by weight and get top K
    sorted_tokens = sorted(filtered, key=lambda x: x[1], reverse=True)
    return sorted_tokens[:k]


def create_attention_heatmap_data(tokens, attention_weights):
    """
    Create data structure for plotly heatmap
    
    Args:
        tokens (list): List of tokens
        attention_weights (list): Attention weights
        
    Returns:
        dict: Data for heatmap visualization
    """
    highlighted = create_highlighted_tokens(tokens, attention_weights)
    
    if not highlighted:
        return {
            'tokens': [],
            'weights': [],
            'colors': []
        }
    
    return {
        'tokens': [item['token'] for item in highlighted],
        'weights': [item['weight'] for item in highlighted],
        'colors': [item['color'] for item in highlighted]
    }


def explain_prediction(tokens, attention_weights, predicted_emotion, confidence):
    """
    Generate human-readable explanation of prediction
    
    Args:
        tokens (list): Tokens from input
        attention_weights (list): Attention weights
        predicted_emotion (str): Predicted emotion label
        confidence (float): Prediction confidence
        
    Returns:
        dict: Explanation data
    """
    top_tokens = get_top_k_important_tokens(tokens, attention_weights, k=5)
    highlighted = create_highlighted_tokens(tokens, attention_weights)
    
    # Create explanation text
    if top_tokens:
        token_list = ', '.join([f'"{token}"' for token, _ in top_tokens[:3]])
        explanation = (
            f"The model predicted **{predicted_emotion}** with "
            f"{confidence*100:.1f}% confidence. "
            f"Key tokens that influenced this prediction: {token_list}."
        )
    else:
        explanation = (
            f"The model predicted **{predicted_emotion}** with "
            f"{confidence*100:.1f}% confidence."
        )
    
    return {
        'explanation': explanation,
        'top_tokens': top_tokens,
        'highlighted_tokens': highlighted,
        'num_tokens_analyzed': len(highlighted)
    }


if __name__ == "__main__":
    # Test explainability functions
    print("Testing Explainability Module...")
    
    # Sample tokens and attention weights
    test_tokens = ['[CLS]', 'காதல்', '##்', 'என்', '##பது', '[SEP]']
    test_weights = [0.1, 0.8, 0.3, 0.6, 0.4, 0.1]
    
    print("\nOriginal tokens:", test_tokens)
    print("Attention weights:", test_weights)
    
    # Test normalization
    normalized = normalize_attention_weights(test_weights)
    print("\nNormalized weights:", [f"{w:.3f}" for w in normalized])
    
    # Test highlighted tokens
    highlighted = create_highlighted_tokens(test_tokens, test_weights)
    print("\nHighlighted tokens:")
    for item in highlighted:
        print(f"  {item['token']:15} | Weight: {item['weight']:.3f} | Color: {item['color']}")
    
    # Test top K tokens
    top_tokens = get_top_k_important_tokens(test_tokens, test_weights, k=3)
    print("\nTop 3 important tokens:")
    for token, weight in top_tokens:
        print(f"  {token:15} : {weight:.3f}")
    
    # Test explanation generation
    explanation = explain_prediction(
        test_tokens, test_weights,
        "Love", 0.92
    )
    print("\nExplanation:")
    print(explanation['explanation'])
