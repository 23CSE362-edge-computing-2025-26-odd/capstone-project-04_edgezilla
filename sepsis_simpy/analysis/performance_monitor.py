# analysis/performance_monitor.py

import time
import numpy as np
from collections import defaultdict
import config

class LatencyTracker:
    """Tracks start and end times of REAL-WORLD code execution using a high-precision timer."""
    def __init__(self):
        self.start_times = {}
        # This will now ONLY store real-world latencies like DQN inference time
        self.latencies = defaultdict(list)

    def start(self, metric_name, item_id):
        """Starts a timer for a specific item."""
        # Use the high-precision performance counter
        self.start_times[(metric_name, item_id)] = time.perf_counter()

    def end(self, metric_name, item_id):
        """Stops a timer and records the duration."""
        key = (metric_name, item_id)
        if key in self.start_times:
            # Duration will be in seconds, but with much higher precision
            duration = time.perf_counter() - self.start_times.pop(key)
            self.latencies[metric_name].append(duration)
            return duration
        return None

class EnergyCalculator:
    """Models and calculates energy consumption."""
    def __init__(self):
        self.total_energy = defaultdict(float) # Joules

    def record_device_energy(self, device_type, state, duration_s):
        """Calculates and adds energy for device activity."""
        power_watts = 0
        if device_type == 'edge':
            power_watts = config.POWER_EDGE_SERVER_BUSY if state == 'busy' else config.POWER_EDGE_SERVER_IDLE
        elif device_type == 'cloud':
            power_watts = config.POWER_CLOUD_INSTANCE_BUSY if state == 'busy' else config.POWER_CLOUD_INSTANCE_IDLE
        elif device_type == 'wearable_tx':
            power_watts = config.POWER_WEARABLE_TRANSMIT
        
        energy_joules = power_watts * duration_s
        self.total_energy[f"{device_type}_{state}"] += energy_joules

    def record_network_energy(self, network_type, data_size_kb):
        """Calculates and adds energy for data transmission."""
        data_size_mb = data_size_kb / 1024.0
        energy_joules = 0
        if network_type == 'wireless':
            energy_joules = config.ENERGY_PER_MB_WIRELESS * data_size_mb
        elif network_type == 'wired':
            energy_joules = config.ENERGY_PER_MB_WIRED * data_size_mb
        self.total_energy[f"network_{network_type}"] += energy_joules

class ThroughputAnalyzer:
    """Measures processing rates and utilization."""
    def __init__(self):
        self.completed_tasks = defaultdict(int)
    
    def record_completion(self, task_type):
        """Increments the count for a completed task."""
        self.completed_tasks[task_type] += 1

class AccuracyEvaluator:
    """Compares DQN decisions against a baseline optimal policy."""
    def __init__(self):
        self.decision_log = [] # Entries: {'correct': bool, 'reason': str}

    def evaluate_decision(self, state, action):
        """Evaluates a decision and logs if it was 'correct'."""
        cpu_util, _, queue_len, _, _ = state
        
        # Define the simple "optimal" policy
        optimal_action = 0 # Default to Edge
        reason = "System load is low."
        
        if cpu_util > config.CPU_UTIL_THRESHOLD_FOR_OFFLOAD:
            optimal_action = 1 # Offload to Cloud
            reason = f"CPU util ({cpu_util:.2f}) exceeded threshold."
        elif queue_len > config.QUEUE_LENGTH_THRESHOLD_FOR_OFFLOAD:
            optimal_action = 1 # Offload to Cloud
            reason = f"Queue length ({queue_len}) exceeded threshold."
            
        is_correct = (action == optimal_action)
        self.decision_log.append({'correct': is_correct, 'dqn_action': action, 'optimal_action': optimal_action, 'reason': reason})

# The Central Hub for All Monitoring
class PerformanceMonitor:
    def __init__(self):
        self.realtime_latency = LatencyTracker() # For real-world code like DQN
        self.energy = EnergyCalculator()
        self.throughput = ThroughputAnalyzer()
        self.accuracy = AccuracyEvaluator()

        # NEW: Direct storage for SIMULATION-TIME latencies
        self.sim_latencies = defaultdict(list)

    def get_aggregated_results(self, sim_duration):
        results = {"system": {"total_simulation_time_s": sim_duration}}
        
        # Combine both real-time and simulation-time latencies into one report
        all_latencies = {**self.realtime_latency.latencies, **self.sim_latencies}

        results['latency'] = {
            name: {
                "count": len(values),
                "mean_s": np.mean(values) if values else 0,
                "median_s": np.median(values) if values else 0,
                "p95_s": np.percentile(values, 95) if values else 0,
            } for name, values in all_latencies.items()
        }
        
        # Energy, Throughput, and Accuracy aggregation remains the same
        total_system_energy = sum(self.energy.total_energy.values())
        results['energy'] = { "components": self.energy.total_energy, "total_system_energy_joules": total_system_energy }
        total_tasks = self.throughput.completed_tasks.get('task_processed', 0)
        results['throughput'] = { "completed_tasks": self.throughput.completed_tasks, "overall_tasks_per_second": total_tasks / sim_duration if sim_duration > 0 else 0 }
        total_decisions = len(self.accuracy.decision_log)
        correct_decisions = sum(1 for d in self.accuracy.decision_log if d['correct'])
        results['accuracy'] = { "total_decisions": total_decisions, "correct_decisions": correct_decisions, "accuracy_rate": (correct_decisions / total_decisions) if total_decisions > 0 else "N/A" }
        
        return results
