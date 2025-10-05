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
            
            # Validate required columns for ML model
            required_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
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
    
    def generate_vitals_data(self):
        """Generate all vital signs data from CSV or synthetic fallback."""
        if self.data is not None and len(self.data) > 0:
            row = self.data.iloc[self.current_index % len(self.data)]
            return {
                'HR': int(row['HR']),
                'O2Sat': float(row['O2Sat']),
                'Temp': float(row['Temp']),
                'SBP': int(row['SBP']),
                'MAP': int(row['MAP']),
                'DBP': int(row['DBP']),
                'Resp': int(row['Resp']),
                'EtCO2': int(row['EtCO2'])
            }
        else:
            # Fallback synthetic generation
            return {
                'HR': int(np.random.normal(80, 15)),
                'O2Sat': round(np.random.uniform(94.0, 99.5), 1),
                'Temp': round(np.random.normal(37.0, 0.5), 1),
                'SBP': int(np.random.normal(120, 20)),
                'MAP': int(np.random.normal(85, 15)),
                'DBP': int(np.random.normal(75, 10)),
                'Resp': int(np.random.normal(16, 4)),
                'EtCO2': int(np.random.normal(35, 5))
            }

    def generate_all(self):
        """Generate a complete set of health parameters and advance the index."""
        vitals = self.generate_vitals_data()
        
        # Add legacy field mapping for backward compatibility
        result = {
            **vitals,
            "heart_rate": vitals['HR'],
            "blood_oxygen": vitals['O2Sat'],
            "temperature": vitals['Temp'],
            "movement": random.choice([0, 1])  # Keep for compatibility
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
    """Uses local ML model for sepsis detection instead of HTTP requests."""
    def __init__(self, device_type='edge'):
        """
        Initialize the patient state model with local ML inference.
        
        Args:
            device_type: 'edge' or 'cloud' - affects inference speed
        """
        from .local_ml_inference import LocalMLInference
        
        self.device_type = device_type
        self.ml_engine = LocalMLInference(device_type=device_type)
        self.states = ['normal', 'suspicious', 'critical_sepsis']
        self.current_state = 'normal'
        self.last_prediction = 0
        self.last_inference_time_ms = 0.0
        
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
        Use local ML model for sepsis detection with realistic inference timing.
        
        Args:
            data: Dictionary containing health parameters
            
        Returns:
            0 or 1 (no sepsis / sepsis detected)
        """
        try:
            # Prepare patient data for ML model (8 parameters)
            patient_vitals = {
                'HR': data.get('HR', data.get('heart_rate', 80)),
                'O2Sat': data.get('O2Sat', data.get('blood_oxygen', 98.0)),
                'Temp': data.get('Temp', data.get('temperature', 37.0)),
                'SBP': data.get('SBP', 120),  # Default values if not available
                'MAP': data.get('MAP', 85),
                'DBP': data.get('DBP', 75),
                'Resp': data.get('Resp', 16),
                'EtCO2': data.get('EtCO2', 35)
            }
            
            # Perform local ML inference with timing
            prediction, inference_time = self.ml_engine.predict_sepsis(patient_vitals)
            
            # Store inference time for metrics
            self.last_inference_time_ms = inference_time * 1000
            self.last_prediction = prediction
            
            return prediction
                
        except Exception as e:
            print(f"Local ML inference error: {e}")
            # Fallback to simple heuristic
            return self._fallback_sepsis_detection(data)
    
    def _fallback_sepsis_detection(self, data):
        """
        Fallback sepsis detection using simple heuristics when ML model is unavailable.
        
        Args:
            data: Dictionary containing health parameters
            
        Returns:
            0 or 1 (no sepsis / sepsis detected)
        """
        # Extract parameters with backward compatibility
        hr = data.get('HR', data.get('heart_rate', 80))
        o2_sat = data.get('O2Sat', data.get('blood_oxygen', 98.0))
        temp = data.get('Temp', data.get('temperature', 37.0))
        resp = data.get('Resp', 16)
        sbp = data.get('SBP', 120)
        
        # Enhanced sepsis indicators using SIRS criteria
        sepsis_score = 0
        
        # Temperature: fever or hypothermia
        if temp > 38.0 or temp < 36.0:
            sepsis_score += 1
            
        # Heart rate: tachycardia
        if hr > 90:
            sepsis_score += 1
            
        # Respiratory rate: tachypnea
        if resp > 20:
            sepsis_score += 1
            
        # Oxygen saturation: hypoxemia
        if o2_sat < 95:
            sepsis_score += 1
            
        # Blood pressure: hypotension
        if sbp < 90:
            sepsis_score += 1
            
        # Return 1 if 2 or more SIRS criteria present
        return 1 if sepsis_score >= 2 else 0