import flask
from flask import request, jsonify
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import copy
import json
import threading
import time

class IPAgent:
    """
    An Intelligent Planning Agent for making offloading decisions at the edge.
    Uses iterative planning with value function approximation.
    """
    def __init__(self, state_dim, action_dim, planning_horizon=5, batch_size=64, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        
        # Value function network for state evaluation
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Policy network for action selection
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Target networks for stable learning
        self.target_value_net = copy.deepcopy(self.value_net)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.memory = deque(maxlen=10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.global_reward = 0.0
        self.local_reward_history = deque(maxlen=100)
        self.value_history = deque(maxlen=50)
        self.action_history = deque(maxlen=100)
        self.update_counter = 0

    def get_state_value(self, state):
        """Predicts state value for planning."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.value_net(state).item()
            self.value_history.append(value)
            return value

    def get_action_probabilities(self, state):
        """Gets action probabilities from policy network."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state)
            probabilities = torch.softmax(logits, dim=1).numpy().flatten()
            return probabilities

    def iterative_planning(self, state, num_iterations=5):
        """Performs iterative planning to find optimal action."""
        current_state = state.copy()
        best_action = 0
        best_value = float('-inf')
        
        for iteration in range(num_iterations):
            # Get current state value
            state_value = self.get_state_value(current_state)
            
            # Get action probabilities
            action_probs = self.get_action_probabilities(current_state)
            
            # Evaluate each action
            for action in range(self.action_dim):
                # Simulate action outcome (simplified model)
                next_state = self.simulate_transition(current_state, action)
                next_value = self.get_state_value(next_state)
                
                # Calculate expected value
                expected_value = state_value + self.gamma * next_value * action_probs[action]
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
        
        return best_action, best_value

    def simulate_transition(self, state, action):
        """Simulates state transition for planning."""
        # Simplified transition model
        next_state = state.copy()
        if action == 0:  # Edge processing
            next_state[0] = min(1.0, state[0] + 0.1)  # Increase load
        else:  # Cloud offloading
            next_state[0] = max(0.0, state[0] - 0.1)  # Decrease load
        return next_state

    def calculate_local_reward(self, latency, cpu_util, queue_length, action):
        """Calculates a reward based on local performance metrics."""
        latency_reward = -latency / 10.0
        cpu_penalty = -abs(cpu_util - 0.7) * 10
        queue_penalty = -queue_length * 0.5
        action_reward = 0.0
        if action == 0:  # Edge processing
            action_reward = 5.0 if cpu_util < 0.8 else -2.0
        else:  # Cloud offloading
            action_reward = 3.0 if cpu_util > 0.8 else -1.0
        
        return latency_reward + cpu_penalty + queue_penalty + action_reward

    def store_transition(self, state, action, reward, next_state, done, latency=None, cpu_util=None, queue_length=None):
        """Stores an experience tuple in the replay memory."""
        # Enhance the base reward with local metrics and global feedback
        local_reward = self.calculate_local_reward(latency, cpu_util, queue_length, action)
        enhanced_reward = reward + local_reward + self.global_reward
        
        self.local_reward_history.append(enhanced_reward)
        self.action_history.append(action)
        self.memory.append((state, action, enhanced_reward, next_state, done))

    def update(self):
        """Trains the networks using a batch from the replay memory."""
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
        current_values = self.value_net(states).squeeze()
        next_values = self.target_value_net(next_states).squeeze()
        expected_values = rewards + self.gamma * next_values * (1 - dones)
        
        value_loss = nn.MSELoss()(current_values, expected_values.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        action_probs = torch.softmax(self.policy_net(states), dim=1)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(torch.log(selected_probs + 1e-8) * expected_values.detach()).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.update_counter += 1
        if self.update_counter % 50 == 0:
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        
        return value_loss.item() + policy_loss.item()

    def set_global_reward(self, gr):
        """Sets the global reward received from the cloud."""
        self.global_reward = gr

    def get_statistics(self):
        """Returns key statistics about the agent's performance."""
        return {
            'memory_size': len(self.memory), 'epsilon': self.epsilon,
            'avg_local_reward': np.mean(self.local_reward_history) if self.local_reward_history else 0.0,
            'action_distribution': {
                'edge': list(self.action_history).count(0) / len(self.action_history) if self.action_history else 0.0,
                'cloud': list(self.action_history).count(1) / len(self.action_history) if self.action_history else 0.0
            },
            'recent_values': list(self.value_history)[-5:] if self.value_history else []
        }

app = flask.Flask(__name__)
agents = {}
agent_lock = threading.Lock()

def get_agent(edge_id):
    """Safely retrieves or creates an agent instance."""
    with agent_lock:
        if edge_id not in agents:
            print(f"Creating new IP agent for edge_id: {edge_id}")
            agents[edge_id] = IPAgent(state_dim=6, action_dim=2)
        return agents[edge_id]

@app.route('/get_action', methods=['POST'])
def get_action_route():
    data = request.json
    agent = get_agent(data['edge_id'])
    
    if random.random() < agent.epsilon:
        action = random.randint(0, agent.action_dim - 1)
        is_explore = True
    else:
        action, _ = agent.iterative_planning(np.array(data['state']))
        is_explore = False
    
    state_value = agent.get_state_value(np.array(data['state']))
    return jsonify({'action': action, 'epsilon': agent.epsilon, 'state_value': state_value, 'is_explore': is_explore})

@app.route('/store_transition', methods=['POST'])
def store_transition_route():
    data = request.json
    agent = get_agent(data['edge_id'])
    agent.store_transition(
        np.array(data['state']), data['action'], data['reward'], np.array(data['next_state']), data['done'],
        data.get('latency'), data.get('cpu_util'), data.get('queue_length')
    )
    loss = agent.update()
    return jsonify({'status': 'OK', 'loss': loss})

@app.route('/set_global_reward', methods=['POST'])
def set_global_reward_route():
    data = request.json
    agent = get_agent(data['edge_id'])
    agent.set_global_reward(data['global_reward'])
    return jsonify({'status': 'OK'})

@app.route('/get_state_values', methods=['POST'])
def get_state_values_route():
    edge_ids = request.json.get('edge_ids', [])
    state_values = {}
    with agent_lock:
        for edge_id in edge_ids:
            if edge_id in agents and agents[edge_id].value_history:
                state_values[edge_id] = agents[edge_id].value_history[-1]
            else:
                state_values[edge_id] = 0.0
    return jsonify({'state_values': state_values})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'active_agents': len(agents), 'server': 'edge_ip'})

if __name__ == '__main__':
    print("Starting Edge IP Server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=False)
