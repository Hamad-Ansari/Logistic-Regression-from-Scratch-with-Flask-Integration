import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json

# FIXED: Remove the dot for direct import
from model import LogisticRegressionFromScratch

def load_and_preprocess_data():
    """Load and preprocess the Breast Cancer Wisconsin dataset"""
    print("Loading Breast Cancer Wisconsin dataset...")
    
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Create DataFrame for better understanding
    feature_names = data.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target distribution:\n{df['diagnosis'].value_counts()}")
    print(f"Malignant (1): {sum(y)}, Benign (0): {len(y)-sum(y)}")
    
    return X, y, feature_names

def main():
    """Main training function"""
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegressionFromScratch(learning_rate=0.01, epochs=2000, lambda_param=0.01)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model Evaluation Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    model.save_model('trained_model.json')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved successfully!")
    print(f"Final weights shape: {model.weights.shape}")
    print(f"Final bias: {model.bias:.4f}")

if __name__ == "__main__":
    main()