"""
Model Training Pipeline for Tamil Poetry Emotion Classification
Fine-tunes IndicBERT on Tamil poetry with emotion labels
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.preprocess import normalize_text
from backend.utils import save_pickle, ensure_dir, get_model_path, get_data_path


class EmotionDataset(Dataset):
    """Custom Dataset for emotion classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.title('Emotion Classification Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_class_distribution(labels, save_path):
    """Plot class distribution"""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='navy')
    plt.xlabel('Emotion Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution in Dataset', fontsize=14)
    plt.xticks(range(len(unique)), unique, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to: {save_path}")
    plt.close()


def train_model(data_path=None, model_name='ai4bharat/indic-bert', 
                output_dir=None, epochs=10, batch_size=8):
    """
    Train emotion classification model
    
    Args:
        data_path (str): Path to CSV data file
        model_name (str): HuggingFace model name
        output_dir (str): Directory to save model
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
    """
    # Set default paths
    if data_path is None:
        data_path = get_data_path('primary_emotions.csv')
    if output_dir is None:
        output_dir = get_model_path('emotion_model')
    
    ensure_dir(output_dir)
    
    print("=" * 80)
    print("TAMIL POETRY EMOTION CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 80)
    
    # Load data
    print(f"\n[1/8] Loading data from: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Preprocess texts
    print("\n[2/8] Preprocessing Tamil texts...")
    df['cleaned_text'] = df['Poem'].apply(normalize_text)
    df = df[df['cleaned_text'].str.len() > 0]  # Remove empty texts
    print(f"Samples after cleaning: {len(df)}")
    
    # Encode labels
    print("\n[3/8] Encoding emotion labels...")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Primary'])
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Save label encoder
    label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    save_pickle(label_encoder, label_encoder_path)
    
    # Plot class distribution
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plot_class_distribution(df['Primary'].values, class_dist_path)
    
    # Split data
    print("\n[4/8] Splitting data (80% train, 20% test)...")
    
    # Check if we have enough samples for stratified split
    from collections import Counter
    class_counts = Counter(df['label'].values)
    min_samples = min(class_counts.values())
    
    if min_samples < 2:
        print(f"⚠️  Warning: Dataset too small for stratified split (min samples: {min_samples})")
        print("Using simple random split instead...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'].values,
            df['label'].values,
            test_size=0.2,
            random_state=42,
            stratify=None  # Don't stratify for small datasets
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'].values,
            df['label'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label'].values
        )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Load tokenizer and model
    print(f"\n[5/8] Loading IndicBERT tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    print("\n[6/8] Creating PyTorch datasets...")
    train_dataset = EmotionDataset(X_train, y_train, tokenizer)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    print("\n[7/8] Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none'
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n[8/8] Training model...")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    trainer.train()
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    eval_results = trainer.evaluate()
    
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    # Predictions for confusion matrix
    print("\nGenerating confusion matrix...")
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        y_test, y_pred,
        label_encoder.classes_,
        cm_path
    )
    
    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {output_dir}")
    print(f"Label encoder saved to: {label_encoder_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Class distribution saved to: {class_dist_path}")
    
    return trainer, model, tokenizer, label_encoder


if __name__ == "__main__":
    # Train the model
    print("Starting training pipeline...")
    print("Note: This requires GPU for reasonable training time.")
    print("If you don't have GPU, consider using Google Colab.\n")
    
    try:
        trainer, model, tokenizer, label_encoder = train_model(
            epochs=10,
            batch_size=8
        )
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure data/primary_emotions.csv exists")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Check if you have enough memory")
        import traceback
        traceback.print_exc()
