import numpy as np
import pickle
import json

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, epochs=2000, lambda_param=0.01):
        """
        Custom Logistic Regression implementation from scratch
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            epochs (int): Number of training iterations
            lambda_param (float): L2 regularization parameter
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """Initialize weights and bias"""
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
    def compute_cost(self, y, y_pred):
        """Compute logistic regression cost with L2 regularization"""
        m = y.shape[0]
        
        # Binary cross-entropy loss
        cost = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        
        # L2 regularization
        reg_cost = (self.lambda_param / (2 * m)) * np.sum(self.weights ** 2)
        
        return cost + reg_cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        
        Args:
            X (numpy.ndarray): Training features
            y (numpy.ndarray): Training labels
        """
        m, n = X.shape
        self.initialize_parameters(n)
        
        print("Training Logistic Regression Model...")
        for epoch in range(self.epochs):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.loss_history.append(cost)
            
            # Backward propagation
            dw = (1/m) * np.dot(X.T, (y_pred - y)) + (self.lambda_param / m) * self.weights
            db = (1/m) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    def predict(self, X):
        """Make predictions on new data"""
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Return probability estimates"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def save_model(self, filepath):
        """Save model parameters to file"""
        model_data = {
            'weights': self.weights.tolist(),
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'lambda_param': self.lambda_param
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load model parameters from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.weights = np.array(model_data['weights'])
        self.bias = model_data['bias']
        self.learning_rate = model_data['learning_rate']
        self.epochs = model_data['epochs']
        self.lambda_param = model_data['lambda_param']


# For backward compatibility
logistic_regression_from_scratch = LogisticRegressionFromScratch