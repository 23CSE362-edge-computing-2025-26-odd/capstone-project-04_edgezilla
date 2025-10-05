import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
import copy
from abc import ABC, abstractmethod

class BaseIPAgent(ABC):
    """
    Base class for Intelligent Planning agents.
    Provides common functionality for iterative planning algorithms.
    """
    
    def __init__(self, state_dim, action_dim, planning_horizon=5, gamma=0.99, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.gamma = gamma
        self.lr = lr
        
        # Common attributes
        self.memory = deque(maxlen=10000)
        self.update_counter = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # History tracking
        self.reward_history = deque(maxlen=100)
        self.value_history = deque(maxlen=50)
        self.action_history = deque(maxlen=100)
        
    @abstractmethod
    def build_networks(self):
        """Build the neural networks for the specific agent."""
        pass
    
    @abstractmethod
    def plan_action(self, state, num_iterations=5):
        """Perform iterative planning to select an action."""
        pass
    
    @abstractmethod
    def update_networks(self, batch):
        """Update the neural networks with a batch of experiences."""
        pass
    
    def simulate_transition(self, state, action):
        """
        Simulate state transition for planning.
        Override in subclasses for domain-specific transitions.
        """
        # Default transition model
        next_state = state.copy()
        if action == 0:  # Default action 0
            next_state[0] = min(1.0, state[0] + 0.1)
        else:  # Default action 1
            next_state[0] = max(0.0, state[0] - 0.1)
        return next_state
    
    def calculate_reward(self, state, action, next_state, **kwargs):
        """
        Calculate reward for a transition.
        Override in subclasses for domain-specific rewards.
        """
        # Default reward function
        reward = 0.0
        if action == 0:
            reward = 1.0 if state[0] < 0.8 else -0.5
        else:
            reward = 0.5 if state[0] > 0.2 else -0.2
        return reward
    
    def store_experience(self, state, action, reward, next_state, done, **kwargs):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        self.reward_history.append(reward)
        self.action_history.append(action)
    
    def sample_batch(self, batch_size=64):
        """Sample a batch from replay memory."""
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)
    
    def update_epsilon(self):
        """Update exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_statistics(self):
        """Get agent performance statistics."""
        return {
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
            'action_distribution': {
                str(i): list(self.action_history).count(i) / len(self.action_history) 
                if self.action_history else 0.0
                for i in range(self.action_dim)
            },
            'update_count': self.update_counter
        }

class ValueBasedIPAgent(BaseIPAgent):
    """
    Value-based Intelligent Planning agent.
    Uses value function approximation for planning.
    """
    
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.build_networks()
    
    def build_networks(self):
        """Build value and policy networks."""
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        
        # Target networks
        self.target_value_net = copy.deepcopy(self.value_net)
        self.target_policy_net = copy.deepcopy(self.policy_net)
        
        # Optimizers
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def get_state_value(self, state):
        """Get state value estimate."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.value_net(state_tensor).item()
            self.value_history.append(value)
            return value
    
    def get_action_probabilities(self, state):
        """Get action probabilities from policy network."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy().flatten()
            return probabilities
    
    def plan_action(self, state, num_iterations=5):
        """Perform iterative planning using value function."""
        current_state = state.copy()
        best_action = 0
        best_value = float('-inf')
        
        for iteration in range(num_iterations):
            state_value = self.get_state_value(current_state)
            action_probs = self.get_action_probabilities(current_state)
            
            for action in range(self.action_dim):
                next_state = self.simulate_transition(current_state, action)
                next_value = self.get_state_value(next_state)
                
                expected_value = state_value + self.gamma * next_value * action_probs[action]
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
        
        return best_action, best_value
    
    def update_networks(self, batch):
        """Update value and policy networks."""
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
        
        self.update_counter += 1
        if self.update_counter % 50 == 0:
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        
        return value_loss.item() + policy_loss.item()

class PolicyBasedIPAgent(BaseIPAgent):
    """
    Policy-based Intelligent Planning agent.
    Uses direct policy optimization for planning.
    """
    
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.build_networks()
    
    def build_networks(self):
        """Build policy network."""
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        
        self.target_policy_net = copy.deepcopy(self.policy_net)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def get_action_probabilities(self, state):
        """Get action probabilities from policy network."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy().flatten()
            return probabilities
    
    def plan_action(self, state, num_iterations=5):
        """Perform iterative planning using policy gradient."""
        current_state = state.copy()
        best_action = 0
        best_expected_reward = float('-inf')
        
        for iteration in range(num_iterations):
            action_probs = self.get_action_probabilities(current_state)
            
            for action in range(self.action_dim):
                next_state = self.simulate_transition(current_state, action)
                reward = self.calculate_reward(current_state, action, next_state)
                
                expected_reward = reward + self.gamma * action_probs[action] * reward
                
                if expected_reward > best_expected_reward:
                    best_expected_reward = expected_reward
                    best_action = action
        
        return best_action, best_expected_reward
    
    def update_networks(self, batch):
        """Update policy network."""
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Policy gradient update
        action_probs = torch.softmax(self.policy_net(states), dim=1)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(torch.log(selected_probs + 1e-8) * rewards).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % 50 == 0:
            self.target_policy_net.load_state_dict(self.policy_net.state_dict())
        
        return policy_loss.item()
