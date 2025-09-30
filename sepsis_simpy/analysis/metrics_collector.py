import pandas as pd

class SimulationMetrics:
    """Tracks and stores all key performance indicators during the simulation."""
    def __init__(self):
        self.decision_data = []
        self.performance_data = []
        self.learning_data = []

    def record_decision(self, timestamp, ward_id, action, state):
        """Records an offloading decision."""
        # Action: 0=edge, 1=cloud
        self.decision_data.append({
            'timestamp': timestamp,
            'ward_id': ward_id,
            'action': 'edge' if action == 0 else 'cloud',
            'cpu_util_state': state[0],
            'queue_len_state': state[2]
        })

    def record_latency(self, timestamp, ward_id, execution_time, action):
        """Records the end-to-end latency for a completed task."""
        self.performance_data.append({
            'timestamp': timestamp,
            'ward_id': ward_id,
            'latency': execution_time,
            'processing_location': 'edge' if action == 0 else 'cloud'
        })

    def update_learning_metrics(self, timestamp, agent_id, reward, loss, epsilon):
        """Records metrics related to the DQN agent's learning progress."""
        self.learning_data.append({
            'timestamp': timestamp,
            'agent_id': agent_id,
            'reward': reward,
            'loss': loss,
            'epsilon': epsilon
        })
        
    def get_dataframes(self):
        """Converts collected data into pandas DataFrames for analysis."""
        return {
            'decisions': pd.DataFrame(self.decision_data),
            'performance': pd.DataFrame(self.performance_data),
            'learning': pd.DataFrame(self.learning_data)
        }