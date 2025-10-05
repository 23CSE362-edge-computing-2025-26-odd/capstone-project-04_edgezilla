import flask
from flask import request, jsonify
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import copy
import threading

class CloudIPAgent:
    """
    Cloud-based Intelligent Planning Agent for resource allocation decisions.
    Uses hierarchical planning with multi-objective optimization.
    """
    def __init__(self, state_dim, action_dim, planning_horizon=8, replay_size=20000, batch_size=128, gamma=0.99, lr=0.0005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        
        # Multi-objective value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3)  # 3 objectives: efficiency, fairness, stability
        )
        
        # Hierarchical policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Target networks
        self.target_value_net = copy.deepcopy(self.value_net)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.update_counter = 0

    def aggregate_state_values(self, edge_state_values):
        """Aggregates state values from edge agents."""
        if not edge_state_values:
            return np.zeros(6)
        
        all_values = []
        for edge_id in sorted(edge_state_values.keys()):
            all_values.append(edge_state_values[edge_id])
        
        aggregated = all_values[:6]
        while len(aggregated) < 6:
            aggregated.append(0.0)
            
        return np.array(aggregated)

    def hierarchical_planning(self, state, num_levels=3):
        """Performs hierarchical planning for resource allocation."""
        current_state = state.copy()
        best_action = 0
        best_multi_objective_value = float('-inf')
        
        for level in range(num_levels):
            # Get multi-objective values
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                objectives = self.value_net(state_tensor).numpy().flatten()
                efficiency, fairness, stability = objectives
            
            # Get action probabilities
            action_probs = torch.softmax(self.policy_net(state_tensor), dim=1).numpy().flatten()
            
            # Evaluate each action with multi-objective criteria
            for action in range(self.action_dim):
                # Simulate resource allocation outcome
                next_state = self.simulate_resource_allocation(current_state, action)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    next_objectives = self.value_net(next_state_tensor).numpy().flatten()
                    next_efficiency, next_fairness, next_stability = next_objectives
                
                # Calculate weighted multi-objective value
                expected_efficiency = efficiency + self.gamma * next_efficiency * action_probs[action]
                expected_fairness = fairness + self.gamma * next_fairness * action_probs[action]
                expected_stability = stability + self.gamma * next_stability * action_probs[action]
                
                # Weighted combination of objectives
                multi_objective_value = (0.4 * expected_efficiency + 
                                       0.4 * expected_fairness + 
                                       0.2 * expected_stability)
                
                if multi_objective_value > best_multi_objective_value:
                    best_multi_objective_value = multi_objective_value
                    best_action = action
        
        return best_action, best_multi_objective_value

    def simulate_resource_allocation(self, state, action):
        """Simulates resource allocation outcome."""
        next_state = state.copy()
        if action == 0:  # Balanced allocation
            next_state[6] = min(1.0, state[6] + 0.05)  # CPU utilization
            next_state[7] = min(1.0, state[7] + 0.05)  # Memory utilization
        elif action == 1:  # CPU intensive
            next_state[6] = min(1.0, state[6] + 0.1)
            next_state[7] = max(0.0, state[7] - 0.05)
        elif action == 2:  # Memory intensive
            next_state[6] = max(0.0, state[6] - 0.05)
            next_state[7] = min(1.0, state[7] + 0.1)
        else:  # High performance
            next_state[6] = min(1.0, state[6] + 0.08)
            next_state[7] = min(1.0, state[7] + 0.08)
        return next_state

    def make_resource_allocation_decision(self, state):
        """Makes resource allocation decision using hierarchical planning."""
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action, _ = self.hierarchical_planning(state)
        return action

    def map_action_to_allocation(self, action):
        """Maps action to resource allocation strategy."""
        strategies = {
            0: {"priority": "balanced", "cpu_cores": 4, "memory_gb": 8},
            1: {"priority": "cpu_intensive", "cpu_cores": 8, "memory_gb": 4},
            2: {"priority": "memory_intensive", "cpu_cores": 2, "memory_gb": 16},
            3: {"priority": "high_performance", "cpu_cores": 8, "memory_gb": 16}
        }
        return strategies.get(action, strategies[0])

    def calculate_cloud_reward(self, cpu, mem, queue, throughput, fairness):
        """Calculates multi-objective cloud reward."""
        efficiency_reward = throughput * 2.0
        fairness_reward = fairness * 20.0
        stability_penalty = abs(cpu - 0.75) * 15 + abs(mem - 0.7) * 10 + queue * 1.0
        
        if 0.5 < cpu < 0.9 and 0.4 < mem < 0.8:
            stability_bonus = 5.0
        else:
            stability_bonus = 0.0
            
        return efficiency_reward + fairness_reward - stability_penalty + stability_bonus

    def distribute_reward_to_edges(self, total_reward, contributions):
        """Distributes reward to edge agents based on contributions."""
        if not contributions: return {}
        total_contrib = sum(contributions.values())
        if total_contrib == 0:
            share = total_reward / len(contributions)
            return {edge_id: share for edge_id in contributions}
        
        return {
            edge_id: total_reward * (contrib / total_contrib)
            for edge_id, contrib in contributions.items()
        }
        
    def store_and_update(self, state, action, reward, next_state, done):
        """Stores experience and updates networks."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Update value network
        current_values = self.value_net(states)
        next_values = self.target_value_net(next_states)
        expected_values = rewards.unsqueeze(1) + self.gamma * next_values * (1 - dones.unsqueeze(1))

        value_loss = nn.MSELoss()(current_values, expected_values.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network
        action_probs = torch.softmax(self.policy_net(states), dim=1)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(torch.log(selected_probs + 1e-8) * rewards).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
            
        return value_loss.item() + policy_loss.item()

app = flask.Flask(__name__)
agent = CloudIPAgent(state_dim=9, action_dim=4)
cloud_state_history = deque(maxlen=2)

@app.route('/aggregate_and_allocate', methods=['POST'])
def aggregate_and_allocate():
    data = request.json
    edge_state_values = data.get('edge_state_values', {})
    system_metrics = data.get('system_metrics', {})
    
    aggregated_values = agent.aggregate_state_values(edge_state_values)
    cloud_metrics = np.array([
        system_metrics.get('cpu_util', 0.0),
        system_metrics.get('memory_util', 0.0),
        system_metrics.get('queue_length', 0.0)
    ])
    current_cloud_state = np.concatenate([aggregated_values, cloud_metrics])
    
    action = agent.make_resource_allocation_decision(current_cloud_state)
    allocation_strategy = agent.map_action_to_allocation(action)
    cloud_reward = agent.calculate_cloud_reward(
        system_metrics.get('cpu_util', 0.0), system_metrics.get('memory_util', 0.0),
        system_metrics.get('queue_length', 0.0), system_metrics.get('throughput', 0.0),
        system_metrics.get('fairness_index', 0.0)
    )
    
    if len(cloud_state_history) > 0:
        prev_cloud_state = cloud_state_history[0]
        agent.store_and_update(prev_cloud_state, action, cloud_reward, current_cloud_state, False)
    
    cloud_state_history.append(current_cloud_state)
    
    distributed_rewards = agent.distribute_reward_to_edges(
        cloud_reward, system_metrics.get('edge_contributions', {})
    )
    
    return jsonify({
        'allocation_strategy': allocation_strategy,
        'cloud_reward': cloud_reward,
        'distributed_rewards': distributed_rewards,
        'cloud_epsilon': agent.epsilon
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'server': 'cloud_ip'})

if __name__ == '__main__':
    print("Starting Cloud IP Server on port 5003...")
    app.run(host='0.0.0.0', port=5003, debug=False)
