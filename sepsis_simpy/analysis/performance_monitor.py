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
    """Evaluates DQN decisions based on sepsis detection medical necessity."""
    def __init__(self):
        self.decision_log = [] # Entries: {'correct': bool, 'reason': str}
        self.sepsis_cases_detected = 0
        self.total_evaluations = 0

    def evaluate_decision(self, state, action, patient_data=None):
        """Evaluates a decision based on medical appropriateness for sepsis detection."""
        self.total_evaluations += 1
        
        # Extract sepsis indicators from patient data or state
        sepsis_risk = self._calculate_sepsis_risk(patient_data, state)
        
        # Define medically-appropriate optimal policy
        optimal_action = self._get_medical_optimal_action(sepsis_risk)
        
        is_correct = (action == optimal_action)
        reason = self._get_medical_reasoning(sepsis_risk, optimal_action, action)
        
        if sepsis_risk > 0.3:
            self.sepsis_cases_detected += 1
            
        self.decision_log.append({
            'correct': is_correct, 
            'dqn_action': action, 
            'optimal_action': optimal_action, 
            'sepsis_risk': sepsis_risk,
            'reason': reason
        })
        
    def _calculate_sepsis_risk(self, patient_data, state):
        """Calculate sepsis risk from available patient/state data."""
        if patient_data:
            # Use actual patient vitals if available
            hr = patient_data.get('HR', 75)
            spo2 = patient_data.get('SpO2', 98)
            
            risk_score = 0.0
            if hr > 100: risk_score += min((hr - 100) / 50, 0.4)
            if spo2 < 95: risk_score += min((95 - spo2) / 10, 0.4)
            
            return min(risk_score, 1.0)
        else:
            # Fallback: assume moderate risk for safety
            return 0.4
    
    def _get_medical_optimal_action(self, sepsis_risk):
        """Determine medically optimal action based on sepsis risk."""
        if sepsis_risk >= 0.5:
            return 1  # Cloud - high risk needs high-accuracy ML
        elif sepsis_risk >= 0.3:
            return 1  # Cloud - moderate risk, better safe than sorry
        else:
            return 0  # Edge - low risk can use faster processing
    
    def _get_medical_reasoning(self, sepsis_risk, optimal_action, actual_action):
        """Provide medical reasoning for evaluation."""
        if sepsis_risk >= 0.5:
            base_reason = f"High sepsis risk ({sepsis_risk:.2f}) requires cloud processing"
        elif sepsis_risk >= 0.3:
            base_reason = f"Moderate sepsis risk ({sepsis_risk:.2f}) benefits from cloud accuracy"
        else:
            base_reason = f"Low sepsis risk ({sepsis_risk:.2f}) can use edge processing"
            
        if actual_action == optimal_action:
            return f"✅ {base_reason} - Decision appropriate"
        else:
            return f"❌ {base_reason} - Decision suboptimal"

# DRLAccuracyEvaluator removed - using only SingleEdgeDRLAccuracyEvaluator for edge-only DRL

# The Central Hub for All Monitoring
class PerformanceMonitor:
    def __init__(self):
        self.realtime_latency = LatencyTracker() # For real-world code like DQN
        self.energy = EnergyCalculator()
        self.throughput = ThroughputAnalyzer()
        self.accuracy = AccuracyEvaluator()
        # DRL accuracy handling moved to SingleEdgeDRLAccuracyEvaluator

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
        
        # Add DQN accuracy metrics if available
        if hasattr(self, 'accuracy') and self.accuracy:
            total_decisions = len(self.accuracy.decision_log)
            correct_decisions = sum(1 for d in self.accuracy.decision_log if d['correct'])
            results['accuracy'] = { 
                "decision_log": self.accuracy.decision_log,
                "total_decisions": total_decisions, 
                "correct_decisions": correct_decisions, 
                "accuracy_rate": (correct_decisions / total_decisions) if total_decisions > 0 else "N/A" 
            }
        
        # Add DRL accuracy metrics if available
        if hasattr(self, 'drl_accuracy') and self.drl_accuracy:
            drl_total_decisions = len(self.drl_accuracy.decision_log)
            drl_correct_decisions = sum(1 for d in self.drl_accuracy.decision_log if d['correct'])
            drl_coord_stats = self.drl_accuracy.get_coordination_accuracy()
            
            results['drl_accuracy'] = {
                "decision_log": self.drl_accuracy.decision_log,
                "coordination_log": self.drl_accuracy.coordination_log,
                "total_decisions": drl_total_decisions,
                "correct_decisions": drl_correct_decisions, 
                "accuracy_rate": (drl_correct_decisions / drl_total_decisions) if drl_total_decisions > 0 else "N/A",
                "coordination_quality": drl_coord_stats['avg_quality'],
                "coordination_samples": drl_coord_stats['count']
            }
            
        # Add Sepsis-Aware DRL accuracy metrics if available (backward compatibility)
        if hasattr(self, 'single_edge_accuracy') and self.single_edge_accuracy:
            # Redirect to new evaluator for compatibility
            self.sepsis_drl_accuracy = self.single_edge_accuracy
        
        # Add Sepsis-Aware DRL accuracy metrics if available
        if hasattr(self, 'sepsis_drl_accuracy') and self.sepsis_drl_accuracy:
            sepsis_drl_stats = self.sepsis_drl_accuracy.get_accuracy_stats()
            
            results['sepsis_aware_drl_accuracy'] = {
                "decision_log": self.sepsis_drl_accuracy.decision_log,
                "performance_metrics": self.sepsis_drl_accuracy.performance_metrics,
                "total_decisions": sepsis_drl_stats['total_decisions'],
                "correct_decisions": sepsis_drl_stats['correct_decisions'],
                "accuracy_rate": sepsis_drl_stats['medical_accuracy'],
                "medical_appropriateness_score": sepsis_drl_stats['medical_appropriateness_score'],
                "avg_sepsis_risk": sepsis_drl_stats['avg_sepsis_risk'],
                "sepsis_cases_processed": sepsis_drl_stats['sepsis_cases_processed'],
                "cloud_routing_decisions": sepsis_drl_stats['cloud_routing_decisions'],
                "cloud_routing_rate": sepsis_drl_stats['cloud_routing_rate'],
                "avg_cpu_util": sepsis_drl_stats['avg_cpu_util'],
                "avg_queue_length": sepsis_drl_stats['avg_queue_length']
            }
        
        return results


class SepsisAwareDRLAccuracyEvaluator:
    """Evaluates sepsis-aware DRL decisions based on medical appropriateness for sepsis detection."""
    
    def __init__(self):
        self.decision_log = []  # Track all decisions made
        self.performance_metrics = []  # Track performance indicators
        self.sepsis_cases_processed = 0
        self.cloud_routing_decisions = 0  # Track cloud routing for sepsis cases
        
    def evaluate_decision(self, state, action, performance_data, patient_data=None):
        """
        Evaluate a single-edge DRL decision for medical appropriateness.
        
        Args:
            state: System state [cpu_util, memory_util, queue_len, network_latency, task_priority]
            action: Chosen action (0: process locally, 1: defer/queue)
            performance_data: Dict with latency, cpu_util, queue_length
            patient_data: Optional patient vital signs for sepsis risk assessment
        """
        cpu_util, memory_util, queue_len, network_latency, task_priority = state
        
        # Calculate sepsis risk for medical evaluation
        sepsis_risk = self._calculate_sepsis_risk(patient_data, task_priority)
        
        # Define medically-appropriate optimal action for sepsis-aware system
        optimal_action, reason = self._get_sepsis_medical_optimal(
            sepsis_risk, cpu_util, memory_util, queue_len
        )
        
        is_correct = (action == optimal_action)
        
        # Track sepsis-specific metrics
        if sepsis_risk > 0.3:
            self.sepsis_cases_processed += 1
            if action == 1:  # Cases routed to cloud
                self.cloud_routing_decisions += 1
        
        # Calculate medical appropriateness score
        medical_score = self._calculate_medical_appropriateness(
            sepsis_risk, action, performance_data
        )
        
        decision_entry = {
            'correct': is_correct,
            'single_edge_action': action,
            'optimal_action': optimal_action,
            'sepsis_risk': sepsis_risk,
            'medical_appropriateness': medical_score,
            'reason': reason,
            'cpu_utilization': cpu_util,
            'queue_length': queue_len,
            'task_priority': task_priority
        }
        
        self.decision_log.append(decision_entry)
        
        # Store performance metrics
        self.performance_metrics.append({
            'latency': performance_data.get('latency', 0),
            'cpu_util': cpu_util,
            'queue_length': queue_len,
            'medical_score': medical_score
        })
    
    def _calculate_sepsis_risk(self, patient_data, task_priority):
        """Calculate sepsis risk from available data."""
        if patient_data:
            # Use actual patient vitals if available
            hr = patient_data.get('HR', 75)
            spo2 = patient_data.get('SpO2', 98)
            
            risk_score = 0.0
            if hr > 100: risk_score += min((hr - 100) / 50, 0.4)
            if spo2 < 95: risk_score += min((95 - spo2) / 10, 0.4)
            
            return min(risk_score, 1.0)
        else:
            # Use task priority as proxy for urgency
            return min(task_priority * 0.6, 0.8) if task_priority else 0.2
    
    def _get_sepsis_medical_optimal(self, sepsis_risk, cpu_util, memory_util, queue_len):
        """Determine optimal action for sepsis-aware system with cloud routing capability."""
        
        # For sepsis-aware systems, we can route to edge (0) or cloud (1)
        # High sepsis risk should be routed to cloud for better care
        if sepsis_risk >= 0.5:
            # High risk - route to cloud for expert analysis
            return 1, f"High sepsis risk ({sepsis_risk:.2f}) - route to cloud for expert care"
        
        elif sepsis_risk >= 0.3:
            # Moderate risk - route to cloud if edge is busy, otherwise edge is ok
            if cpu_util > 0.8 or queue_len > 10:
                return 1, f"Moderate sepsis risk ({sepsis_risk:.2f}) - route to cloud due to edge load"
            else:
                return 0, f"Moderate sepsis risk ({sepsis_risk:.2f}) - process on edge"
        
        else:
            # Low risk - prefer edge processing for efficiency
            if cpu_util > 0.9 or queue_len > 15:
                return 1, f"Low sepsis risk ({sepsis_risk:.2f}) - route to cloud due to overload"
            else:
                return 0, f"Low sepsis risk ({sepsis_risk:.2f}) - efficient edge processing"
    
    def _calculate_medical_appropriateness(self, sepsis_risk, action, performance_data):
        """Calculate how medically appropriate the decision was."""
        
        # High sepsis risk cases should be routed to cloud for expert care
        if sepsis_risk >= 0.5:
            if action == 1:  # Routed to cloud
                return 0.9  # Excellent medical decision
            else:  # Processed on edge
                return 0.4  # Suboptimal (should have used cloud expertise)
        
        elif sepsis_risk >= 0.3:
            if action == 1:  # Routed to cloud
                return 0.8  # Good decision for moderate risk
            else:  # Processed on edge
                latency = performance_data.get('latency', 0.1)
                return max(0.6 - latency * 0.3, 0.3)  # Acceptable if fast
        
        else:
            # Low risk - edge processing is preferred for efficiency
            if action == 0:  # Edge processing
                return 0.8  # Good efficient decision
            else:  # Cloud processing
                return 0.6  # Acceptable but unnecessary cloud usage
    
    def get_accuracy_stats(self):
        """Get accuracy statistics for single-edge DRL with medical metrics."""
        if not self.decision_log:
            return {'accuracy': 0.0, 'count': 0, 'medical_score': 0.0}
        
        correct_decisions = sum(1 for d in self.decision_log if d['correct'])
        total_decisions = len(self.decision_log)
        avg_medical_score = sum(d['medical_appropriateness'] for d in self.decision_log) / total_decisions
        avg_sepsis_risk = sum(d['sepsis_risk'] for d in self.decision_log) / total_decisions
        
        return {
            'medical_accuracy': correct_decisions / total_decisions,
            'correct_decisions': correct_decisions,
            'total_decisions': total_decisions,
            'medical_appropriateness_score': avg_medical_score,
            'avg_sepsis_risk': avg_sepsis_risk,
            'sepsis_cases_processed': self.sepsis_cases_processed,
            'cloud_routing_decisions': self.cloud_routing_decisions,
            'cloud_routing_rate': self.cloud_routing_decisions / max(1, self.sepsis_cases_processed),
            'avg_cpu_util': sum(d['cpu_utilization'] for d in self.decision_log) / total_decisions,
            'avg_queue_length': sum(d['queue_length'] for d in self.decision_log) / total_decisions
        }
