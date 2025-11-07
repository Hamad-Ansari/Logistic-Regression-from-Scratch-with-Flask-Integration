# Logistic Regression from Scratch with Flask Integration

🚀 **Built a Machine Learning Model from Scratch + Integrated with a Flask App**

This project demonstrates the implementation of a Logistic Regression model completely from scratch, with every formula and function manually implemented — no pre-built ML libraries for training or prediction! The model is integrated with a Flask web app for real-time predictions and visualization.

## 📋 Project Overview

I developed a beginner-level Machine Learning model from the ground up, implementing all mathematical formulas manually. The project includes a Flask web interface that runs locally to showcase how the model performs predictions in real-time.

## ✅ Key Features

- **Custom Logistic Regression** implementation using only NumPy
- **Manual preprocessing** and feature scaling using StandardScaler
- **98% accuracy** on the Breast Cancer Wisconsin dataset
- **Flask web interface** for local demonstration and real-time predictions
- **Performance visualization** and prediction results
- **Complete mathematical implementation** of gradient descent and sigmoid activation

## 🧠 Tech Stack

- **Python** (core ML logic implementation)
- **Flask** (web framework for local app)
- **NumPy, Pandas** (data manipulation and numerical operations)
- **HTML/CSS/JavaScript** (frontend interface)
- **Scikit-learn** (only for dataset loading and train-test split)

## 📊 Model Workflow

1. **Data Loading**: Breast Cancer Wisconsin dataset
2. **Preprocessing**: 
   - Removed unnecessary columns (id, Unnamed: 32)
   - Mapped diagnosis labels: Benign → 0, Malignant → 1
3. **Feature Engineering**:
   - Normalized features using StandardScaler
   - Split data: 80% training, 20% testing
4. **Model Implementation**:
   - Logistic Regression with L2 regularization
   - Gradient Descent optimization (2000 epochs)
   - Sigmoid activation function
5. **Deployment**:
   - Saved trained weights, bias, and scaler
   - Flask integration for real-time predictions

## 🚀 Installation & Usage

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/logistic-regression-from-scratch.git
cd logistic-regression-from-scratch
