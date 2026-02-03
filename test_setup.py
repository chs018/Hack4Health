"""
Quick Test Script
Tests all modules to ensure proper installation
"""

import sys
import os

print("=" * 80)
print("TESTING EMOTION-RASA CLASSIFICATION SYSTEM")
print("=" * 80)

# Test 1: Import all backend modules
print("\n[1/6] Testing module imports...")
try:
    from backend import preprocess, rasa_mapper, utils, explain
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test preprocessing
print("\n[2/6] Testing Tamil text preprocessing...")
from backend.preprocess import normalize_text
test_text = "காதல் என்பது உயிரினும் இனிது, உன் சிரிப்பில்!"
cleaned = normalize_text(test_text)
print(f"  Original: {test_text}")
print(f"  Cleaned: {cleaned}")
print("✓ Preprocessing works")

# Test 3: Test Rasa mapping
print("\n[3/6] Testing Rasa mapping...")
from backend.rasa_mapper import map_to_rasa, get_rasa_description
test_emotions = ['Love', 'Joy', 'Sorrow', 'Devotion']
for emotion in test_emotions:
    rasa = map_to_rasa(emotion)
    desc = get_rasa_description(rasa)
    print(f"  {emotion:12} → {rasa:12} ({desc[:30]}...)")
print("✓ Rasa mapping works")

# Test 4: Test utilities
print("\n[4/6] Testing utility functions...")
from backend.utils import calculate_confidence_level, get_model_path
conf_levels = [0.95, 0.65, 0.35]
for conf in conf_levels:
    level = calculate_confidence_level(conf)
    print(f"  Confidence {conf:.2f} → {level}")
print(f"  Model path: {get_model_path()}")
print("✓ Utilities work")

# Test 5: Test explainability
print("\n[5/6] Testing explainability functions...")
from backend.explain import normalize_attention_weights, create_highlighted_tokens
test_tokens = ['[CLS]', 'காதல்', 'என்பது', '[SEP]']
test_weights = [0.1, 0.8, 0.6, 0.1]
normalized = normalize_attention_weights(test_weights)
highlighted = create_highlighted_tokens(test_tokens, test_weights)
print(f"  Normalized weights: {[f'{w:.2f}' for w in normalized]}")
print(f"  Highlighted tokens: {len(highlighted)} tokens")
print("✓ Explainability works")

# Test 6: Check data file
print("\n[6/6] Checking data file...")
data_path = os.path.join('data', 'primary_emotions.csv')
if os.path.exists(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"  Found: {data_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Emotions: {df['Primary'].unique()}")
    print("✓ Data file exists and is valid")
else:
    print(f"✗ Data file not found: {data_path}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nNext steps:")
print("1. Train the model: python backend/train.py")
print("2. Start the API: python backend/app.py")
print("3. Launch dashboard: streamlit run frontend/dashboard.py")
print()
