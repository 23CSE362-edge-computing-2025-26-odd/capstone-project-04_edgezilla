import pandas as pd
import numpy as np
import random
import requests
import config
import os

class HealthDataGenerator:
    """Reads patient health parameters from a CSV dataset."""
    _instance = None
    _initialized = False
    
    def __new__(cls, csv_file_path=None):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(HealthDataGenerator, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, csv_file_path=None):
        """Initialize only once."""
        if self._initialized:
            return
        
        """
        Initialize the data generator with a CSV file.
        
        Args:
            csv_file_path: Path to the CSV file containing patient data.
                          If None, uses default path 'dataset/patient_data.csv'
        """
        if csv_file_path is None:
            # Default path relative to the project root
            csv_file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'patient_data.csv')
        
        self.csv_file_path = csv_file_path
        self.data = None
        self.current_index = 0
        self.load_data()
        self._initialized = True
    
    def load_data(self):
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.data)} records from {self.csv_file_path}")
            
            # Validate required columns
            required_columns = ['heart_rate', 'blood_oxygen', 'temperature', 'movement']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
                
        except FileNotFoundError:
            print(f"Warning: CSV file not found at {self.csv_file_path}")
            print("Falling back to synthetic data generation...")
            self.data = None
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            print("Falling back to synthetic data generation...")
            self.data = None
    
    def generate_heart_rate(self):
        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[self.current_index % len(self.data)]
            return int(row['heart_rate'])
        return int(np.random.normal(80, 15))  # Fallback

    def generate_blood_oxygen(self):
        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[self.current_index % len(self.data)]
            return float(row['blood_oxygen'])
        return round(np.random.uniform(94.0, 99.5), 1)  # Fallback

    def generate_temperature(self):
        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[self.current_index % len(self.data)]
            return float(row['temperature'])
        return round(np.random.normal(37.0, 0.5), 1)  # Fallback

    def generate_movement_data(self):
        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[self.current_index % len(self.data)]
            return int(row['movement'])
        return random.choice([0, 1])  # Fallback

    def generate_all(self):
        """Generate a complete set of health parameters and advance the index."""
        result = {
            "heart_rate": self.generate_heart_rate(),
            "blood_oxygen": self.generate_blood_oxygen(),
            "temperature": self.generate_temperature(),
            "movement": self.generate_movement_data()
        }
        
        # Advance to next row for subsequent calls
        self.current_index += 1
        
        return result
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._initialized = False

class PatientStateModel:
    """Makes API calls to ML server for sepsis detection."""
    def __init__(self, ml_server_url="http://localhost:5002"):
        """
        Initialize the patient state model.
        
        Args:
            ml_server_url: URL of the ML server for sepsis detection
        """
        self.ml_server_url = ml_server_url
        self.states = ['normal', 'suspicious', 'critical_sepsis']
        self.current_state = 'normal'
        self.last_prediction = 0
        
    def update_patient_state(self, health_data=None):
        """
        Updates patient state based on ML prediction.
        
        Args:
            health_data: Dictionary containing patient health parameters
            
        Returns:
            Current patient state string
        """
        if health_data is not None:
            sepsis_prediction = self.calculate_sepsis_risk(health_data)
            
            # Update state based on ML prediction
            if sepsis_prediction == 1:
                if self.current_state == 'normal':
                    self.current_state = 'suspicious'
                elif self.current_state == 'suspicious':
                    self.current_state = 'critical_sepsis'
            else:
                # Gradual recovery if no sepsis detected
                if self.current_state == 'critical_sepsis':
                    self.current_state = 'suspicious'
                elif self.current_state == 'suspicious' and random.random() > 0.7:
                    self.current_state = 'normal'
                    
        return self.current_state

    def calculate_sepsis_risk(self, data):
        """
        Make API call to ML server for sepsis detection.
        
        Args:
            data: Dictionary containing health parameters
            
        Returns:
            0 or 1 (no sepsis / sepsis detected)
        """
        try:
            # Prepare the payload for the ML API
            payload = {
                "heart_rate": data.get("heart_rate", 80),
                "blood_oxygen": data.get("blood_oxygen", 98.0),
                "temperature": data.get("temperature", 37.0),
                "movement": data.get("movement", 0)
            }
            
            # Make API call to ML server (fast timeout for speed)
            response = requests.post(
                f"{self.ml_server_url}/predict_sepsis",
                json=payload,
                timeout=0.5  # 0.5 second timeout (fast mode)
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = int(result.get("prediction", 0))
                self.last_prediction = prediction
                return prediction
            else:
                print(f"ML API error: {response.status_code} - {response.text}")
                return self.last_prediction  # Return last known prediction
                
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to ML server: {e}")
            # Fallback to simple heuristic if ML server is unavailable
            return self._fallback_sepsis_detection(data)
        except Exception as e:
            print(f"Error in sepsis prediction: {e}")
            return self.last_prediction
    
    def _fallback_sepsis_detection(self, data):
        """
        Fallback sepsis detection using simple heuristics when ML server is unavailable.
        
        Args:
            data: Dictionary containing health parameters
            
        Returns:
            0 or 1 (no sepsis / sepsis detected)
        """
        # Simple heuristic: high heart rate, low oxygen, high temperature
        heart_rate = data.get("heart_rate", 80)
        blood_oxygen = data.get("blood_oxygen", 98.0)
        temperature = data.get("temperature", 37.0)
        
        # Basic sepsis indicators
        sepsis_score = 0
        if heart_rate > 90:  # Tachycardia
            sepsis_score += 1
        if blood_oxygen < 95:  # Low oxygen saturation
            sepsis_score += 1
        if temperature > 38.0 or temperature < 36.0:  # Fever or hypothermia
            sepsis_score += 1
            
        # Return 1 if 2 or more indicators present
        return 1 if sepsis_score >= 2 else 0