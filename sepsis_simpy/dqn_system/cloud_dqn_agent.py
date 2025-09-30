import numpy as np

class CloudDQNAgent:
    """
    Advanced cloud agent for global optimization.
    For simulation, this uses a rule-based approach to mimic a DQN's output.
    """
    def __init__(self):
        # Actions: 0=balanced, 1=cpu_intensive, 2=memory_intensive, 3=high_performance
        self.action_space = ['balanced', 'cpu_intensive', 'memory_intensive', 'high_performance']
        print("Initialized Cloud Coordination Agent.")

    def make_allocation_decision(self, aggregated_system_state):
        """
        Makes a high-level resource allocation decision based on system-wide metrics.
        
        Args:
            aggregated_system_state (dict): Contains metrics like 'avg_cpu_util', 'avg_queue_len'.
        """
        avg_cpu = aggregated_system_state.get('avg_cpu_util', 0)
        avg_queue = aggregated_system_state.get('avg_queue_len', 0)
        
        # Rule-based logic to simulate an intelligent decision
        if avg_cpu > 0.8 or avg_queue > 20:
            # If the system is heavily congested, switch to high performance mode
            strategy = 'high_performance'
        elif avg_cpu > 0.6:
            # If CPU is the main bottleneck
            strategy = 'cpu_intensive'
        else:
            # Default balanced strategy
            strategy = 'balanced'
            
        print(f"Cloud Agent Decision: System state (CPU: {avg_cpu:.2f}, Queue: {avg_queue:.2f}) -> Strategy: {strategy}")
        return strategy