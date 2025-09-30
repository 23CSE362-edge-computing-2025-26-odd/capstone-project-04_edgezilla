import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .experience_memory import ExperienceMemory, Transition

# DQN Network Architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class EdgeDQNAgent:
    """Deep Q-Network agent for offloading decisions on an edge server."""
    def __init__(self, state_size, action_size, agent_id, learning_rate=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network and Target Network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

        # Experience Replay Memory
        self.memory = ExperienceMemory(capacity=10000, batch_size=64)

        # Epsilon-Greedy Strategy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Hyperparameters
        self.gamma = gamma
        self.update_target_every = 10 # Update target network every 10 learning steps

    def get_q_values(self, state):
        """Gets Q-values for a given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.numpy()

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            q_values = self.get_q_values(state)
            return np.argmax(q_values)  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        """Stores a transition in the experience replay memory."""
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_model(self):
        """Updates the Q-network by learning from a batch of experiences."""
        if self.memory.get_size() < self.memory.batch_size:
            return  # Not enough experiences to learn

        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample_batch()

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get current Q-values from the main network
        current_q_values = self.q_network(states).gather(1, actions)

        # Get next Q-values from the target network
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        # Compute the target Q-value
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and perform backpropagation
        loss = self.loss_function(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        """Copies weights from the main Q-network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())