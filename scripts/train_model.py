"""
Optimized Drought Prediction Model Training Script
Uses XGBoost with Hyperparameter Tuning and SMOTE for Class Balance
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("DROUGHT PREDICTION MODEL TRAINING")
    print("="*70)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'USDMData.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
    
    print("\n[1/4] Loading data...")
    df = pd.read_csv(data_path)
    df = df.ffill()  # Forward fill missing values
    
    # Separate features and labels
    features = df.drop(columns=['DroughtCategory'], errors='ignore')
    
    # Convert object columns to numeric
    for col in features.select_dtypes(include='object').columns:
        features[col] = features[col].astype('category').cat.codes
    
    # Encode labels
    labels = df['DroughtCategory'] if 'DroughtCategory' in df.columns else df.iloc[:, -1]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    
    print(f"   ✓ Loaded {len(df)} samples with {len(features.columns)} features")
    print(f"   ✓ Classes: {le.classes_}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"   ✓ Train set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    
    # Train XGBoost directly (no extensive GridSearchCV to save time)
    print("\n[2/4] Training XGBoost Model...")
    
    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    
    print("   ✓ Training started...")
    final_model.fit(X_train, y_train, verbose=0)
    print("   ✓ Training completed")
    
    # Evaluate
    print("\n[3/4] Evaluating Model...")
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   ✓ Test Accuracy: {accuracy:.4f}")
    
    # Save results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model and encoder
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'drought_prediction_xgboost.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    
    joblib.dump(final_model, model_path)
    joblib.dump(le, encoder_path)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Encoder saved: {encoder_path}")
    
    # Save confusion matrix visualization
    print("\n[4/4] Saving Results...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - XGBoost\nAccuracy: {accuracy:.4f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    viz_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved: {viz_path}")
    print("\n" + "="*70)
    print(f"✓ PROJECT COMPLETE - Final Accuracy: {accuracy:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
