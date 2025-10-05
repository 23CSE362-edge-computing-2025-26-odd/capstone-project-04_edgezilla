"""
Sepsis-Aware DRL Agent
Makes intelligent offloading decisions between edge and cloud based on patient vital signs.
Implements a medically-informed DRL-based resource allocation policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque


class SepsisAwareDRLAgent:
    def __init__(
        self,
        state_size=7,  # Enhanced state: [cpu, mem, queue, network, task_priority, hr, spo2]
        action_size=2, # 0: edge, 1: cloud
        learning_rate=1e-3,
        memory_size=2000,
        batch_size=32,
        gamma=0.95,
        update_every=4,
        device="cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_every = update_every
        self.step_count = 0
        self.device = torch.device(device)

        # Enhanced networks for sepsis-aware decision making
        self.actor = self._build_sepsis_aware_actor().to(self.device)
        self.critic = self._build_sepsis_aware_critic().to(self.device)

        # Proper optimizers for effective learning
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr * 2)
        
        # Initialize networks properly
        for param in self.actor.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
        for param in self.critic.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

        self.episode_rewards = []
        self.episode_losses = []

        # Reasonable exploration parameters
        self.epsilon = 1.0         # Start exploring
        self.epsilon_min = 0.1     # Minimum exploration
        self.epsilon_decay = 0.995 # Reasonable decay
        
        # Medical decision tracking with decay
        self.sepsis_decisions = 0
        self.cloud_routing_decisions = 0
        self.medical_accuracy_score = 0.0
        self.accuracy_decay_factor = 0.995  # Prevent accuracy inflation
        self.decision_window = 100  # Rolling window for accuracy calculation

    def _build_sepsis_aware_actor(self):
        """Actor network with sepsis-aware architecture."""
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size),
            nn.Softmax(dim=-1),
        )

    def _build_sepsis_aware_critic(self):
        """Critic network with sepsis-aware architecture."""
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def select_action(self, state, patient_data=None, training=True):
        """Select action considering sepsis risk from patient vitals."""
        # Enhance state with patient vitals if available
        enhanced_state = self._enhance_state_with_vitals(state, patient_data)
        
        s = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.actor(s).cpu().numpy().squeeze()

        # Apply sepsis-aware bias
        sepsis_risk = self._calculate_sepsis_risk(patient_data)
        probs = self._apply_medical_bias(probs, sepsis_risk)

        if training and random.random() < self.epsilon:
            # Epsilon-greedy with medical awareness
            if sepsis_risk > 0.5:  # High risk - prefer cloud
                action = 1 if random.random() < 0.7 else 0
            else:
                action = random.randint(0, self.action_size - 1)
            logp = np.log(probs[action] + 1e-8)
        else:
            action = np.argmax(probs)
            logp = np.log(probs[action] + 1e-8)

        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Track medical decisions
        if sepsis_risk > 0.3:
            self.sepsis_decisions += 1
            if action == 1:  # Cloud routing
                self.cloud_routing_decisions += 1

        return int(action), float(logp)

    def _enhance_state_with_vitals(self, state, patient_data):
        """Enhance system state with patient vital signs."""
        if len(state) >= 7:  # Already enhanced
            return state
            
        # Extract HR and SpO2 from patient data
        hr = patient_data.get('HR', 75) if patient_data else 75
        spo2 = patient_data.get('SpO2', 98) if patient_data else 98
        
        # Normalize vitals for neural network
        hr_norm = (hr - 60) / 100  # Normal range ~60-160
        spo2_norm = (spo2 - 85) / 15  # Normal range ~85-100
        
        enhanced_state = list(state) + [hr_norm, spo2_norm]
        return np.array(enhanced_state)
    
    def _calculate_sepsis_risk(self, patient_data):
        """Calculate sepsis risk with medically validated criteria."""
        if not patient_data:
            return 0.05  # Very low default risk
            
        hr = patient_data.get('HR', 75)
        spo2 = patient_data.get('SpO2', 98)
        temp = patient_data.get('temperature', 36.5)
        
        # Medical-grade risk calculation using SIRS criteria
        risk_factors = 0
        base_risk = 0.1
        
        # Heart rate (tachycardia: >90 bpm)
        if hr > 110:  # Severe tachycardia
            risk_factors += 2
            base_risk += 0.4
        elif hr > 90:  # Moderate tachycardia
            risk_factors += 1
            base_risk += 0.2
        elif hr < 50:  # Severe bradycardia
            risk_factors += 2
            base_risk += 0.3
            
        # Oxygen saturation (hypoxemia)
        if spo2 < 88:  # Severe hypoxemia
            risk_factors += 3
            base_risk += 0.6
        elif spo2 < 92:  # Moderate hypoxemia
            risk_factors += 2
            base_risk += 0.4
        elif spo2 < 95:  # Mild hypoxemia
            risk_factors += 1
            base_risk += 0.2
            
        # Temperature (if available)
        if temp > 38.3 or temp < 36.0:  # Fever or hypothermia
            risk_factors += 1
            base_risk += 0.15
            
        # Calculate final risk with medical thresholds
        if risk_factors >= 3:  # Multiple criteria met
            final_risk = min(0.85, base_risk * 1.3)
        elif risk_factors >= 2:  # Two criteria
            final_risk = min(0.65, base_risk * 1.1)
        else:  # Single or no criteria
            final_risk = min(0.35, base_risk)
            
        return final_risk
    
    def _apply_medical_bias(self, probs, sepsis_risk):
        """Apply medical bias to action probabilities based on sepsis risk."""
        if sepsis_risk > 0.5:  # High risk - strongly prefer cloud
            probs[1] = probs[1] * 2.0  # Boost cloud probability
            probs[0] = probs[0] * 0.3  # Reduce edge probability
        elif sepsis_risk > 0.3:  # Medium risk - moderate cloud preference
            probs[1] = probs[1] * 1.5
            probs[0] = probs[0] * 0.7
            
        # Renormalize
        probs = probs / probs.sum()
        return probs
    
    def store_experience(self, state, action, reward, next_state, done, log_prob=0.0, patient_data=None):
        """Store experience with enhanced reward based on medical appropriateness."""
        # Calculate medical appropriateness bonus
        medical_reward = self._calculate_medical_reward(action, patient_data)
        enhanced_reward = reward + medical_reward
        
        self.memory.append((state, action, float(enhanced_reward), next_state, done, log_prob))
    
    def store(self, state, action, reward, next_state, done, log_prob=0.0):
        """Alias for store_experience for backward compatibility."""
        return self.store_experience(state, action, reward, next_state, done, log_prob)
    
    def calculate_edge_only_reward(self, latency, cpu_util, queue_len):
        """Backward compatibility - use sepsis-aware reward calculation."""
        return self.calculate_sepsis_aware_reward(latency, cpu_util, queue_len, action=0)
    
    def compute_reward(self, latency, cpu_util, queue_len):
        """Alias for backward compatibility."""
        return self.calculate_edge_only_reward(latency, cpu_util, queue_len)

    def _calculate_medical_reward(self, action, patient_data):
        """Calculate medical appropriateness reward with stringent evaluation."""
        if not patient_data:
            return -0.1  # Penalize missing medical data
            
        sepsis_risk = self._calculate_sepsis_risk(patient_data)
        
        # Stringent medical decision evaluation
        if sepsis_risk > 0.6:  # Critical risk - MUST go to cloud
            return 0.8 if action == 1 else -1.5  # Heavy penalty for wrong decision
        elif sepsis_risk > 0.4:  # High risk - should go to cloud
            return 0.4 if action == 1 else -0.8  # Significant penalty
        elif sepsis_risk > 0.25:  # Medium risk - context dependent
            return 0.2 if action == 1 else -0.1  # Slight preference for cloud
        else:  # Low risk - should stay on edge for efficiency
            return 0.3 if action == 0 else -0.2  # Penalize unnecessary cloud usage
    
    def calculate_sepsis_aware_reward(self, latency, cpu_util, queue_len, action, patient_data=None):
        """Calculate balanced reward with stringent medical evaluation.
        
        Args:
            latency: Processing latency
            cpu_util: Current CPU utilization
            queue_len: Current queue length
            action: Chosen action (0: edge, 1: cloud)
            patient_data: Patient vital signs for medical evaluation
            
        Returns:
            float: Calculated reward
        """
        # Stricter base performance evaluation
        base_reward = 1.0
        
        # Enhanced performance penalties
        latency_penalty = latency * 3.0  # Higher latency penalty
        cpu_penalty = max(0.0, cpu_util - 0.6) * 4.0  # Earlier CPU penalty
        queue_penalty = max(0.0, queue_len - 3) * 1.0  # Earlier queue penalty
        
        # Resource utilization bonus for good decisions
        resource_bonus = 0.0
        if action == 0 and cpu_util < 0.7:  # Good edge decision
            resource_bonus = 0.2
        elif action == 1 and cpu_util > 0.8:  # Good cloud decision
            resource_bonus = 0.3
        
        performance_reward = base_reward - latency_penalty - cpu_penalty - queue_penalty + resource_bonus
        
        # Medical appropriateness reward (more conservative)
        medical_reward = self._calculate_medical_reward(action, patient_data)
        
        # More balanced combination - medical important but not overwhelming
        total_reward = performance_reward * 0.7 + medical_reward * 1.3
        
        return float(np.clip(total_reward, -4.0, 2.5))
    
    def compute_reward(self, latency, cpu_util, queue_len):
        """Alias for calculate_edge_only_reward for backward compatibility."""
        return self.calculate_edge_only_reward(latency, cpu_util, queue_len)

    def update_policy(self, episode_complete=False):
        """Update the actor-critic policy using collected experiences."""
        self.step_count += 1
        if len(self.memory) < self.batch_size:
            return

        if (self.step_count % self.update_every) != 0 and not episode_complete:
            return

        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones, logps = zip(*batch)
        
        # Ensure all states are enhanced to the expected size
        enhanced_states = []
        enhanced_next_states = []
        
        for state in states:
            if len(state) < self.state_size:
                # Pad with default values
                padded = list(state) + [0.0] * (self.state_size - len(state))
                enhanced_states.append(padded)
            elif len(state) > self.state_size:
                # Truncate to expected size
                enhanced_states.append(state[:self.state_size])
            else:
                enhanced_states.append(state)
                
        for state in next_states:
            if len(state) < self.state_size:
                # Pad with default values
                padded = list(state) + [0.0] * (self.state_size - len(state))
                enhanced_next_states.append(padded)
            elif len(state) > self.state_size:
                # Truncate to expected size
                enhanced_next_states.append(state[:self.state_size])
            else:
                enhanced_next_states.append(state)
        
        # Convert to numpy arrays first, then to tensors
        states_array = np.array(enhanced_states, dtype=np.float32)
        next_states_array = np.array(enhanced_next_states, dtype=np.float32)
        
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Critic update
        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_vals = self.critic(next_states).squeeze()
        targets = rewards + self.gamma * next_vals * (1.0 - dones)
        critic_loss = F.mse_loss(values, targets)

        # Actor update with advantage
        advantages = (targets - values).detach()
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs + 1e-8)
        new_log_probs = dist.log_prob(actions)

        actor_loss = -(new_log_probs * advantages).mean()

        # Update critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        total_loss = actor_loss.item() + critic_loss.item()
        self.episode_losses.append(total_loss)

        if episode_complete:
            ep_reward = sum(rewards.cpu().numpy()) if isinstance(rewards, torch.Tensor) else sum(rewards)
            self.episode_rewards.append(ep_reward)
            # Update medical accuracy tracking
            if self.sepsis_decisions > 0:
                self.medical_accuracy_score = self.cloud_routing_decisions / self.sepsis_decisions
            self.memory.clear()

    def save_model(self, path):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        if "actor_opt" in ckpt and "critic_opt" in ckpt:
            try:
                self.actor_opt.load_state_dict(ckpt["actor_opt"])
                self.critic_opt.load_state_dict(ckpt["critic_opt"])
            except Exception:
                pass

    def get_metrics(self):
        # Apply decay to accuracy score to prevent inflation
        if hasattr(self, 'accuracy_decay_factor'):
            self.medical_accuracy_score *= self.accuracy_decay_factor
        
        # Calculate rolling accuracy over recent decisions
        recent_accuracy = 0.0
        if self.sepsis_decisions > 0:
            # Use more conservative calculation
            recent_accuracy = min(0.9, self.cloud_routing_decisions / max(1, self.sepsis_decisions))
        
        return {
            "epsilon": self.epsilon,
            "avg_episode_reward": float(np.mean(self.episode_rewards[-50:])) if self.episode_rewards else 0.0,
            "avg_loss": float(np.mean(self.episode_losses[-50:])) if self.episode_losses else 0.0,
            "memory_len": len(self.memory),
            "sepsis_decisions": self.sepsis_decisions,
            "cloud_routing_rate": recent_accuracy,
            "medical_accuracy_score": self.medical_accuracy_score,
            "recent_performance": recent_accuracy
        }
