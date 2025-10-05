import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from .experience_memory import ExperienceMemory, Transition

# Enhanced DQN Network Architecture with Sepsis Awareness
class EnhancedQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[256, 256, 128, 64]):
        super(EnhancedQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Enhanced deeper network for better feature learning
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
            
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Separate value and advantage streams (Dueling DQN)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(), 
            nn.Linear(64, action_size)
        )
        
        # Feature extractor (all layers except last)
        self.feature_extractor = nn.Sequential(*layers[:-1])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # Extract features
        features = self.feature_extractor(state)
        
        # Dueling DQN: Combine value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# Backward compatibility
QNetwork = EnhancedQNetwork

class EdgeDQNAgent:
    """Enhanced Deep Q-Network agent with sepsis awareness for offloading decisions."""
    def __init__(self, state_size=7, action_size=2, agent_id=0, learning_rate=5e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size

        # Enhanced Q-Network and Target Network with sepsis awareness
        self.q_network = EnhancedQNetwork(state_size, action_size)
        self.target_network = EnhancedQNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Improved optimizer with learning rate scheduling
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        self.loss_function = nn.HuberLoss()  # More robust than MSE

        # Enhanced Experience Replay Memory
        self.memory = ExperienceMemory(capacity=50000, batch_size=128)

        # Improved Epsilon-Greedy Strategy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Enhanced Hyperparameters
        self.gamma = gamma
        self.update_target_every = 5  # More frequent target updates
        self.learning_step_counter = 0
        
        # Sepsis awareness tracking
        self.sepsis_decisions = 0
        self.correct_sepsis_decisions = 0
        self.medical_accuracy_history = []

    def get_q_values(self, state):
        """Gets Q-values for a given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.numpy()

    def choose_action(self, state, patient_data=None):
        """Enhanced action selection with sepsis awareness."""
        # Enhance state with patient vitals if available
        enhanced_state = self._enhance_state_with_vitals(state, patient_data)
        
        if random.random() < self.epsilon:
            # Sepsis-aware exploration
            if patient_data:
                sepsis_risk = self._calculate_sepsis_risk(patient_data)
                if sepsis_risk > 0.5:  # High risk - bias towards cloud
                    return 1 if random.random() < 0.8 else 0
            return random.randint(0, self.action_size - 1)  # Normal exploration
        else:
            q_values = self.get_q_values(enhanced_state)
            return np.argmax(q_values)  # Exploit
    
    def _enhance_state_with_vitals(self, state, patient_data):
        """Enhance system state with patient vital signs."""
        if len(state) >= 7:  # Already enhanced
            return state
            
        # Extract HR and SpO2 from patient data
        hr = patient_data.get('HR', 75) if patient_data else 75
        spo2 = patient_data.get('SpO2', 98) if patient_data else 98
        
        # Normalize vitals for neural network
        hr_norm = np.clip((hr - 60) / 100, -1, 1)  # Normal range ~60-160
        spo2_norm = np.clip((spo2 - 85) / 15, -1, 1)  # Normal range ~85-100
        
        enhanced_state = list(state) + [hr_norm, spo2_norm]
        return np.array(enhanced_state)
    
    def _calculate_sepsis_risk(self, patient_data):
        """Calculate sepsis risk from patient vital signs."""
        if not patient_data:
            return 0.2
            
        hr = patient_data.get('HR', 75)
        spo2 = patient_data.get('SpO2', 98)
        
        risk_score = 0.0
        if hr > 100:
            risk_score += min((hr - 100) / 50, 0.4)
        if spo2 < 95:
            risk_score += min((95 - spo2) / 10, 0.4)
            
        return min(risk_score, 1.0)

    def store_experience(self, state, action, reward, next_state, done):
        """Stores a transition in the experience replay memory."""
        self.memory.store_transition(state, action, reward, next_state, done)

    def update_model(self):
        """Enhanced Q-network update with improved learning mechanisms."""
        if self.memory.get_size() < self.memory.batch_size:
            return  # Not enough experiences to learn

        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample_batch()

        # Handle variable state sizes
        enhanced_states = []
        enhanced_next_states = []
        
        for state in states:
            if len(state) < self.state_size:
                padded = list(state) + [0.0] * (self.state_size - len(state))
                enhanced_states.append(padded)
            else:
                enhanced_states.append(state[:self.state_size])
                
        for state in next_states:
            if len(state) < self.state_size:
                padded = list(state) + [0.0] * (self.state_size - len(state))
                enhanced_next_states.append(padded)
            else:
                enhanced_next_states.append(state[:self.state_size])

        # Convert to PyTorch tensors
        states = torch.FloatTensor(enhanced_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(enhanced_next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Double DQN: Use main network to select action, target network to evaluate
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Double DQN improvement
        next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, next_actions)
        
        # Compute the target Q-value with enhanced reward shaping
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss and perform backpropagation
        loss = self.loss_function(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        self.scheduler.step()

        # Enhanced epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network more frequently
        self.learning_step_counter += 1
        if self.learning_step_counter % self.update_target_every == 0:
            self.update_target_network()
            
        return loss.item()
        
    def update_target_network(self):
        """Copies weights from the main Q-network to the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())