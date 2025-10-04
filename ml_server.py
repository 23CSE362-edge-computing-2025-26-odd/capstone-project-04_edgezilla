"""
Simple ML Server for Sepsis Detection
Run this server separately to provide sepsis prediction API
Usage: python ml_server.py
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

class SepsisMLModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load existing model or train a new one."""
        model_path = 'sepsis_model.joblib'
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
                print("Loaded existing sepsis model")
            except:
                print("Failed to load existing model, training new one...")
                self.train_model()
        else:
            self.train_model()
    
    def train_model(self):
        """Train a simple RandomForest model on the dataset."""
        try:
            # Load training data
            data_path = '../dataset/patient_data.csv'
            if not os.path.exists(data_path):
                print(f"Warning: Training data not found at {data_path}")
                print("Using minimal synthetic data for training...")
                self.create_minimal_model()
                return
            
            df = pd.read_csv(data_path)
            
            # Features and target
            features = ['heart_rate', 'blood_oxygen', 'temperature', 'movement']
            X = df[features]
            y = df['sepsis_label']
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X, y)
            
            # Save model
            joblib.dump(self.model, 'sepsis_model.joblib')
            self.is_trained = True
            
            print(f"Trained sepsis model on {len(df)} samples")
            
        except Exception as e:
            print(f"Error training model: {e}")
            self.create_minimal_model()
    
    def create_minimal_model(self):
        """Create a minimal model when no data is available."""
        # Create synthetic training data
        np.random.seed(42)
        X = np.random.rand(100, 4)  # 4 features
        X[:, 0] = X[:, 0] * 60 + 70  # heart_rate: 70-130
        X[:, 1] = X[:, 1] * 10 + 90  # blood_oxygen: 90-100
        X[:, 2] = X[:, 2] * 3 + 36   # temperature: 36-39
        X[:, 3] = np.random.randint(0, 2, 100)  # movement: 0 or 1
        
        # Simple rule-based labels
        y = ((X[:, 0] > 90) & (X[:, 1] < 95) & (X[:, 2] > 38)).astype(int)
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        
        print("Created minimal synthetic model")
    
    def predict(self, heart_rate, blood_oxygen, temperature, movement):
        """Make sepsis prediction."""
        if not self.is_trained:
            return 0
        
        try:
            features = np.array([[heart_rate, blood_oxygen, temperature, movement]])
            prediction = self.model.predict(features)[0]
            return int(prediction)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0

# Initialize the ML model
ml_model = SepsisMLModel()

@app.route('/predict_sepsis', methods=['POST'])
def predict_sepsis():
    """API endpoint for sepsis prediction."""
    try:
        data = request.get_json()
        
        # Extract features
        heart_rate = data.get('heart_rate', 80)
        blood_oxygen = data.get('blood_oxygen', 98.0)
        temperature = data.get('temperature', 37.0)
        movement = data.get('movement', 0)
        
        # Make prediction
        prediction = ml_model.predict(heart_rate, blood_oxygen, temperature, movement)
        
        return jsonify({
            'prediction': prediction,
            'status': 'success',
            'input': {
                'heart_rate': heart_rate,
                'blood_oxygen': blood_oxygen,
                'temperature': temperature,
                'movement': movement
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'prediction': 0,
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_trained': ml_model.is_trained
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'is_trained': ml_model.is_trained,
        'features': ['heart_rate', 'blood_oxygen', 'temperature', 'movement']
    })

if __name__ == '__main__':
    print("Starting Sepsis ML Server...")
    print("API Endpoints:")
    print("  POST /predict_sepsis - Make sepsis predictions")
    print("  GET /health - Health check")
    print("  GET /model_info - Model information")
    
    app.run(host='localhost', port=5002, debug=True)