import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import json
import time
from typing import Dict, List, Tuple, Any

class IPMetrics:
    """Utility class for tracking IP algorithm performance metrics."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.actions = deque(maxlen=window_size)
        self.state_values = deque(maxlen=window_size)
        self.planning_times = deque(maxlen=window_size)
        
    def update(self, reward, latency, action, state_value, planning_time):
        """Update metrics with new data."""
        self.rewards.append(reward)
        self.latencies.append(latency)
        self.actions.append(action)
        self.state_values.append(state_value)
        self.planning_times.append(planning_time)
    
    def get_statistics(self):
        """Get current performance statistics."""
        return {
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'avg_latency': np.mean(self.latencies) if self.latencies else 0.0,
            'action_distribution': self._get_action_distribution(),
            'avg_state_value': np.mean(self.state_values) if self.state_values else 0.0,
            'avg_planning_time': np.mean(self.planning_times) if self.planning_times else 0.0,
            'total_samples': len(self.rewards)
        }
    
    def _get_action_distribution(self):
        """Get action distribution statistics."""
        if not self.actions:
            return {}
        
        action_counts = {}
        for action in self.actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(self.actions)
        return {str(k): v/total for k, v in action_counts.items()}

class IPStateEncoder:
    """Utility class for encoding and decoding states for IP algorithms."""
    
    @staticmethod
    def encode_patient_state(hr, temp, bp, resp, o2sat, glucose):
        """Encode patient vital signs into normalized state vector."""
        # Normalize vital signs to [0, 1] range
        normalized_hr = (hr - 50) / 100  # 50-150 bpm
        normalized_temp = (temp - 36.0) / 4.0  # 36-40°C
        normalized_bp = (bp - 80) / 100  # 80-180 mmHg
        normalized_resp = (resp - 10) / 20  # 10-30 breaths/min
        normalized_o2sat = (o2sat - 90) / 10  # 90-100%
        normalized_glucose = (glucose - 70) / 130  # 70-200 mg/dL
        
        return np.array([
            np.clip(normalized_hr, 0, 1),
            np.clip(normalized_temp, 0, 1),
            np.clip(normalized_bp, 0, 1),
            np.clip(normalized_resp, 0, 1),
            np.clip(normalized_o2sat, 0, 1),
            np.clip(normalized_glucose, 0, 1)
        ])
    
    @staticmethod
    def decode_patient_state(state_vector):
        """Decode normalized state vector back to patient vital signs."""
        hr = state_vector[0] * 100 + 50
        temp = state_vector[1] * 4.0 + 36.0
        bp = state_vector[2] * 100 + 80
        resp = state_vector[3] * 20 + 10
        o2sat = state_vector[4] * 10 + 90
        glucose = state_vector[5] * 130 + 70
        
        return {
            'hr': hr, 'temp': temp, 'bp': bp, 'resp': resp, 'o2sat': o2sat, 'glucose': glucose
        }
    
    @staticmethod
    def encode_system_state(cpu_util, memory_util, queue_length, network_latency, throughput, fairness):
        """Encode system metrics into normalized state vector."""
        return np.array([
            np.clip(cpu_util, 0, 1),
            np.clip(memory_util, 0, 1),
            np.clip(queue_length / 50, 0, 1),  # Assume max queue of 50
            np.clip(network_latency / 1.0, 0, 1),  # Assume max latency of 1s
            np.clip(throughput / 100, 0, 1),  # Assume max throughput of 100
            np.clip(fairness, 0, 1)
        ])

class IPRewardCalculator:
    """Utility class for calculating rewards in IP algorithms."""
    
    @staticmethod
    def calculate_sepsis_reward(patient_state, is_sepsis_detected, action, latency, cpu_util):
        """Calculate reward for sepsis detection task."""
        base_reward = 0.0
        
        # Sepsis detection reward
        if is_sepsis_detected:
            base_reward += 10.0
        else:
            base_reward -= 1.0
        
        # Latency penalty
        latency_penalty = -latency * 2.0
        
        # CPU utilization penalty
        cpu_penalty = -abs(cpu_util - 0.7) * 5.0
        
        # Action-specific rewards
        if action == 0:  # Edge processing
            if cpu_util < 0.8:
                base_reward += 5.0
            else:
                base_reward -= 2.0
        else:  # Cloud offloading
            if cpu_util > 0.8:
                base_reward += 3.0
            else:
                base_reward -= 1.0
        
        return base_reward + latency_penalty + cpu_penalty
    
    @staticmethod
    def calculate_system_reward(cpu_util, memory_util, queue_length, throughput, fairness):
        """Calculate reward for system-level optimization."""
        efficiency_reward = throughput * 2.0
        fairness_reward = fairness * 20.0
        stability_penalty = abs(cpu_util - 0.75) * 15 + abs(memory_util - 0.7) * 10 + queue_length * 1.0
        
        if 0.5 < cpu_util < 0.9 and 0.4 < memory_util < 0.8:
            stability_bonus = 5.0
        else:
            stability_bonus = 0.0
        
        return efficiency_reward + fairness_reward - stability_penalty + stability_bonus

class IPPlanningUtils:
    """Utility functions for iterative planning algorithms."""
    
    @staticmethod
    def monte_carlo_rollout(state, action, transition_model, reward_model, horizon=5, num_rollouts=10):
        """Perform Monte Carlo rollout for action evaluation."""
        total_reward = 0.0
        current_state = state.copy()
        
        for _ in range(num_rollouts):
            rollout_reward = 0.0
            temp_state = current_state.copy()
            
            for step in range(horizon):
                next_state = transition_model(temp_state, action)
                reward = reward_model(temp_state, action, next_state)
                rollout_reward += reward
                temp_state = next_state
            
            total_reward += rollout_reward
        
        return total_reward / num_rollouts
    
    @staticmethod
    def value_iteration(state, value_function, transition_model, reward_model, horizon=5):
        """Perform value iteration for action selection."""
        best_action = 0
        best_value = float('-inf')
        
        for action in range(2):  # Assuming binary action space
            total_value = 0.0
            current_state = state.copy()
            
            for step in range(horizon):
                next_state = transition_model(current_state, action)
                reward = reward_model(current_state, action, next_state)
                value = value_function(next_state)
                total_value += reward + 0.99 * value
                current_state = next_state
            
            if total_value > best_value:
                best_value = total_value
                best_action = action
        
        return best_action, best_value
    
    @staticmethod
    def policy_iteration(state, policy_function, value_function, transition_model, reward_model, horizon=5):
        """Perform policy iteration for action selection."""
        # Get current policy
        action_probs = policy_function(state)
        
        # Evaluate each action under current policy
        best_action = 0
        best_value = float('-inf')
        
        for action in range(len(action_probs)):
            expected_value = 0.0
            current_state = state.copy()
            
            for step in range(horizon):
                next_state = transition_model(current_state, action)
                reward = reward_model(current_state, action, next_state)
                value = value_function(next_state)
                expected_value += action_probs[action] * (reward + 0.99 * value)
                current_state = next_state
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        return best_action, best_value

class IPLogger:
    """Utility class for logging IP algorithm performance."""
    
    def __init__(self, log_file="ip_performance.log"):
        self.log_file = log_file
        self.start_time = time.time()
    
    def log_step(self, step, edge_id, action, reward, latency, state_value, planning_time):
        """Log a single simulation step."""
        timestamp = time.time() - self.start_time
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'edge_id': edge_id,
            'action': action,
            'reward': reward,
            'latency': latency,
            'state_value': state_value,
            'planning_time': planning_time
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_cloud_update(self, step, strategy, cloud_reward, distributed_rewards):
        """Log cloud coordination update."""
        timestamp = time.time() - self.start_time
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'type': 'cloud_update',
            'strategy': strategy,
            'cloud_reward': cloud_reward,
            'distributed_rewards': distributed_rewards
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_performance_summary(self):
        """Get performance summary from log file."""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            rewards = []
            latencies = []
            planning_times = []
            
            for line in lines:
                try:
                    entry = json.loads(line.strip())
                    if 'reward' in entry:
                        rewards.append(entry['reward'])
                        latencies.append(entry['latency'])
                        planning_times.append(entry['planning_time'])
                except:
                    continue
            
            return {
                'total_steps': len(rewards),
                'avg_reward': np.mean(rewards) if rewards else 0.0,
                'avg_latency': np.mean(latencies) if latencies else 0.0,
                'avg_planning_time': np.mean(planning_times) if planning_times else 0.0,
                'simulation_time': time.time() - self.start_time
            }
        except:
            return {'error': 'Could not read log file'}

def create_ip_network(input_dim, hidden_dims, output_dim, dropout_rate=0.2):
    """Create a neural network for IP algorithms."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)

def normalize_state(state, state_bounds):
    """Normalize state vector using provided bounds."""
    normalized = np.zeros_like(state)
    for i, (min_val, max_val) in enumerate(state_bounds):
        normalized[i] = (state[i] - min_val) / (max_val - min_val)
        normalized[i] = np.clip(normalized[i], 0, 1)
    return normalized

def denormalize_state(normalized_state, state_bounds):
    """Denormalize state vector using provided bounds."""
    state = np.zeros_like(normalized_state)
    for i, (min_val, max_val) in enumerate(state_bounds):
        state[i] = normalized_state[i] * (max_val - min_val) + min_val
    return state
