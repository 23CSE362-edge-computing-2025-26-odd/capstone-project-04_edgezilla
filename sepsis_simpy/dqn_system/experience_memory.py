import random
from collections import namedtuple, deque
import numpy as np

# A transition is a single step of experience
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceMemory:
    """A circular buffer to store experiences for DQN training."""
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory_buffer = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        """Saves a transition to the memory buffer."""
        experience = Transition(state, action, reward, next_state, done)
        self.memory_buffer.append(experience)

    def sample_batch(self):
        """Samples a random batch of transitions from memory."""
        batch = random.sample(self.memory_buffer, self.batch_size)
        
        # Unzip the batch into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def get_size(self):
        """Returns the current number of experiences in memory."""
        return len(self.memory_buffer)