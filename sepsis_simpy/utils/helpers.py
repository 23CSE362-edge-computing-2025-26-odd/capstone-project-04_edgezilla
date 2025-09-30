import numpy as np
from datetime import datetime

class DataHelpers:
    @staticmethod
    def calculate_moving_average(data, window_size):
        """Calculates the moving average of a list or array."""
        if len(data) < window_size:
            return None
        return np.convolve(data, np.ones(window_size), 'valid') / window_size
    
    @staticmethod
    def format_timestamp(sim_time):
        """Formats simulation time for display."""
        return f"{int(sim_time // 3600):02d}:{int((sim_time % 3600) // 60):02d}:{int(sim_time % 60):02d}"

class SimulationHelpers:
    @staticmethod
    def validate_configuration(config):
        """Performs basic checks on the config file."""
        if config.SIMULATION_DURATION <= 0:
            raise ValueError("SIMULATION_DURATION must be positive.")
        if config.NUM_WARDS < 1 or config.WEARABLES_PER_WARD < 1:
            raise ValueError("Must have at least one ward and one wearable.")
        print("Configuration validated successfully.")