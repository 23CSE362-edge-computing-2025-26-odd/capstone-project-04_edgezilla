import numpy as np
import random
import config

class HealthDataGenerator:
    """Generates synthetic patient health parameters."""
    def generate_heart_rate(self):
        return int(np.random.normal(80, 15)) # Mean 80, std 15

    def generate_blood_oxygen(self):
        return round(np.random.uniform(94.0, 99.5), 1)

    def generate_temperature(self):
        return round(np.random.normal(37.0, 0.5), 1)

    def generate_movement_data(self):
        return random.choice([0, 1])

    def generate_all(self):
        return {
            "heart_rate": self.generate_heart_rate(),
            "blood_oxygen": self.generate_blood_oxygen(),
            "temperature": self.generate_temperature(),
            "movement": self.generate_movement_data()
        }

class PatientStateModel:
    """Simulates changes in a patient's condition."""
    def __init__(self):
        self.states = ['normal', 'suspicious', 'critical_sepsis']
        self.current_state = 'normal'

    def update_patient_state(self):
        """Transitions the patient to a new state based on simple probabilities."""
        # This is a simplistic model for now.
        transition_prob = random.random()
        if self.current_state == 'normal' and transition_prob < 0.05:
            self.current_state = 'suspicious'
        elif self.current_state == 'suspicious' and transition_prob < 0.02:
            self.current_state = 'critical_sepsis'
        elif self.current_state != 'normal' and transition_prob > 0.8:
            self.current_state = 'normal' # Patient recovers
        return self.current_state

    def calculate_sepsis_risk(self, data):
        # Placeholder for risk calculation logic
        return np.random.uniform(0.0, 0.3)