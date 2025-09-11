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

class DQNAgent:
    """
    A Deep Q-Network Agent for making offloading decisions at the edge.
    """
    def _init_(self, state_dim, action_dim, replay_size=10000, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Q-Network for action-value prediction
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # Target network for stable Q-learning targets
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.global_reward = 0.0
        self.local_reward_history = deque(maxlen=100)
        self.q_vector_history = deque(maxlen=50)
        self.action_history = deque(maxlen=100)
        self.update_counter = 0

    def get_q_values(self, state):
        """Predicts Q-values for a given state."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state).numpy().flatten()
            self.q_vector_history.append(q_values.copy())
            return q_values

    def calculate_local_reward(self, latency, cpu_util, queue_length, action):
        """Calculates a reward based on local performance metrics."""
        # Penalty for high latency
        latency_reward = -latency / 10.0
        # Penalty for deviating from ideal CPU utilization (e.g., 70%)
        cpu_penalty = -abs(cpu_util - 0.7) * 10
        # Penalty for long task queues
        queue_penalty = -queue_length * 0.5
        # Action-specific reward/penalty
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
        """Trains the Q-network using a batch from the replay memory."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon to reduce exploration over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Periodically update the target network
        self.update_counter += 1
        if self.update_counter % 50 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

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
            'recent_q_vectors': list(self.q_vector_history)[-5:] if self.q_vector_history else []
        }

app = flask.Flask(_name_)
agents = {}
agent_lock = threading.Lock()

def get_agent(edge_id):
    """Safely retrieves or creates an agent instance."""
    with agent_lock:
        if edge_id not in agents:
            print(f"Creating new agent for edge_id: {edge_id}")
            # State: [HeartRate, Temp, BP, RespRate, O2Sat, Glucose]
            agents[edge_id] = DQNAgent(state_dim=6, action_dim=2)
        return agents[edge_id]

@app.route('/get_q_values', methods=['POST'])
def get_q_values_route():
    data = request.json
    agent = get_agent(data['edge_id'])
    q = agent.get_q_values(np.array(data['state']))
    return jsonify({'q_values': q.tolist(), 'epsilon': agent.epsilon})

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

@app.route('/get_q_vectors', methods=['POST'])
def get_q_vectors_route():
    edge_ids = request.json.get('edge_ids', [])
    q_vectors = {}
    with agent_lock:
        for edge_id in edge_ids:
            if edge_id in agents and agents[edge_id].q_vector_history:
                q_vectors[edge_id] = agents[edge_id].q_vector_history[-1].tolist()
            else:
                q_vectors[edge_id] = [0.0, 0.0]
    return jsonify({'q_vectors': q_vectors})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'active_agents': len(agents), 'server': 'edge_dqn'})

if _name_ == '_main_':
    print("Starting Edge DQN Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
