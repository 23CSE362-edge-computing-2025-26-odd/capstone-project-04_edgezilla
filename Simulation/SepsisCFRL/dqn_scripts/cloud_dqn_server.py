import flask
from flask import request, jsonify
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import copy
import threading

class CloudDQNAgent:
    def __init__(self, state_dim, action_dim, replay_size=20000, batch_size=128, gamma=0.99, lr=0.0005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.update_counter = 0

    def aggregate_q_vectors(self, edge_q_vectors):
        if not edge_q_vectors:
            return np.zeros(6)
        
        all_q_values = []
        for edge_id in sorted(edge_q_vectors.keys()):
            all_q_values.extend(edge_q_vectors[edge_id])
        
        aggregated = all_q_values[:6]
        while len(aggregated) < 6:
            aggregated.append(0.0)
            
        return np.array(aggregated)

    def make_resource_allocation_decision(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action = q_values.argmax().item()
        return action

    def map_action_to_allocation(self, action):
        strategies = {
            0: {"priority": "balanced", "cpu_cores": 4, "memory_gb": 8},
            1: {"priority": "cpu_intensive", "cpu_cores": 8, "memory_gb": 4},
            2: {"priority": "memory_intensive", "cpu_cores": 2, "memory_gb": 16},
            3: {"priority": "high_performance", "cpu_cores": 8, "memory_gb": 16}
        }
        return strategies.get(action, strategies[0])

    def calculate_cloud_reward(self, cpu, mem, queue, throughput, fairness):
        reward = throughput * 2.0 + fairness * 20.0
        reward -= abs(cpu - 0.75) * 15
        reward -= abs(mem - 0.7) * 10
        reward -= queue * 1.0
        if 0.5 < cpu < 0.9 and 0.4 < mem < 0.8:
            reward += 5.0
        return reward

    def distribute_reward_to_edges(self, total_reward, contributions):
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

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_counter += 1
        if self.update_counter % 100 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()

app = flask.Flask(__name__)
agent = CloudDQNAgent(state_dim=9, action_dim=4)
cloud_state_history = deque(maxlen=2)

@app.route('/aggregate_and_allocate', methods=['POST'])
def aggregate_and_allocate():
    data = request.json
    edge_q_vectors = data.get('edge_q_vectors', {})
    system_metrics = data.get('system_metrics', {})
    
    aggregated_q = agent.aggregate_q_vectors(edge_q_vectors)
    cloud_metrics = np.array([
        system_metrics.get('cpu_util', 0.0),
        system_metrics.get('memory_util', 0.0),
        system_metrics.get('queue_length', 0.0)
    ])
    current_cloud_state = np.concatenate([aggregated_q, cloud_metrics])
    
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
    return jsonify({'status': 'healthy', 'server': 'cloud_dqn'})

if __name__ == '__main__':
    print("Starting Cloud DQN Server on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False)