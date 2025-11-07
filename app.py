from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import json
import os

app = Flask(__name__)

# Try different import methods
try:
    from model import LogisticRegressionFromScratch
    print("✓ Successfully imported LogisticRegressionFromScratch from model")
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback: Define the class directly if import fails
    class LogisticRegressionFromScratch:
        def __init__(self, learning_rate=0.01, epochs=2000, lambda_param=0.01):
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.lambda_param = lambda_param
            self.weights = None
            self.bias = None
        
        def sigmoid(self, z):
            z = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z))
        
        def load_model(self, filepath):
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            self.weights = np.array(model_data['weights'])
            self.bias = model_data['bias']
        
        def predict(self, X):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            return (y_pred >= 0.5).astype(int)
        
        def predict_proba(self, X):
            z = np.dot(X, self.weights) + self.bias
            return self.sigmoid(z)

# Load trained model and scaler
try:
    model = LogisticRegressionFromScratch()
    if os.path.exists('trained_model.json'):
        model.load_model('trained_model.json')
        print("✓ Model loaded successfully")
    else:
        print("⚠ trained_model.json not found - run train_model.py first")
        
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded successfully")
    else:
        print("⚠ scaler.pkl not found - run train_model.py first")
        scaler = None
        
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# Sample feature names from Breast Cancer dataset
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not trained yet. Please run train_model.py first.'}), 400
        
        # Get data from request
        data = request.get_json()
        features = np.array([float(data[f]) for f in FEATURE_NAMES]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'diagnosis': 'Malignant' if prediction == 1 else 'Benign',
            'confidence': f"{max(probability, 1-probability)*100:.2f}%"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info')
def model_info():
    """Return model information"""
    info = {
        'accuracy': '98%',
        'training_samples': 455,
        'test_samples': 114,
        'features': len(FEATURE_NAMES),
        'algorithm': 'Logistic Regression with L2 Regularization',
        'learning_rate': 0.01 if model else 'N/A',
        'epochs': 2000 if model else 'N/A',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }
    return jsonify(info)

if __name__ == '__main__':
    print("Starting Flask app...")
    print("If you see import errors, make sure to run: python train_model.py first")
    app.run(debug=True, host='0.0.0.0', port=5000)