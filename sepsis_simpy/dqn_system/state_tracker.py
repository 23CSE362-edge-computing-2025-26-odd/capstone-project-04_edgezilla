import numpy as np
import config

class StateNormalizer:
    """Normalizes state values to be within a consistent range (e.g., 0 to 1)."""
    def normalize(self, state_vector):
        # [cpu_util, mem_usage, queue_len, net_lat, throughput]
        # These max values should be tuned based on observation
        max_values = np.array([
            1.0, # CPU utilization is already 0-1
            config.EDGE_SERVER_RAM,
            50.0, # Max expected queue length
            100.0, # Max expected latency in ms
            1000.0 # Max expected throughput (tasks/sec)
        ])
        # Add a small epsilon to avoid division by zero
        normalized = np.array(state_vector) / (max_values + 1e-6)
        return np.clip(normalized, 0, 1)

class SystemStateTracker:
    """Monitors and collects the real-time system state for DQN decisions."""
    def __init__(self, infrastructure):
        self.infrastructure = infrastructure # Dict of {'edge_servers': [], 'cloud': obj, ...}
        self.normalizer = StateNormalizer()

    def get_edge_state(self, edge_server_id):
        """Collects the current state of a specific edge server."""
        edge = self.infrastructure['edge_servers'][edge_server_id]
        
        # Get raw metrics
        cpu_utilization = edge.get_utilization()
        memory_usage = np.random.uniform(0.2, 0.8) * config.EDGE_SERVER_RAM # Placeholder
        queue_length = len(edge.cpu.queue)
        
        # For simulation, we'll use configured latency. In a real system, this would be measured.
        network_latency_to_cloud = config.LATENCY_EDGE_TO_CLOUD
        
        # Throughput can be measured over a time window (placeholder for now)
        throughput = np.random.uniform(10, 50) # tasks processed per unit time

        # Assemble the raw state vector
        # Note: A timestamp is often useful, but for a stationary policy, we'll omit it for simplicity.
        raw_state = [cpu_utilization, memory_usage, queue_length, network_latency_to_cloud, throughput]
        
        # Normalize the state for the DQN
        normalized_state = self.normalizer.normalize(raw_state)
        
        return normalized_state

    def get_cloud_state(self):
        # Placeholder for future use
        pass

    def get_network_conditions(self):
        # Placeholder for future use
        pass