import numpy as np
import torch
import random
from collections import deque, namedtuple


class MultiAgentPrioritizedReplayBufferNStep:
    """
    Implementation of a Prioritized Replay Buffer with N-step returns for multi-agent reinforcement learning.

    Features:
    - **Prioritized Sampling**: Samples experiences with higher priority more frequently, based on TD errors.
    - **N-Step Returns**: Supports N-step returns to reduce variance in expected returns and stabilize training.
    - **Multi-Agent Support**: Handles experiences for multiple agents simultaneously.
    - **Beta Annealing**: Gradually adjusts the importance sampling weights during training.

    Internal Buffers:
    - `self.n_step_buffer_s_a_ns`: Stores tuples of (state, action, next_state) for N-step transitions.
    - `self.n_step_buffer_rewards`: Stores rewards for N-step transitions.
    - `self.n_step_buffer_dones`: Stores terminal flags for N-step transitions.
    """

    def __init__(
            self,
            buffer_size,
            batch_size,
            device_type,
            random_seed,
            n_steps,
            gamma=0.99,
            alpha=0.6,
            beta=0.4):
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences to store in the buffer.
            batch_size (int): Number of experiences to sample in each batch.
            device_type (str): Device to store tensors (e.g., 'cuda' or 'cpu').
            random_seed (int): Random seed for reproducibility.
            n_step (int): Number of steps to consider for N-step returns.
            gamma (float): Discount factor for future rewards.
            alpha (float): Priority exponent (controls how much prioritization is used).
            beta (float): Importance sampling exponent (controls bias correction).
        """
        self.agent_count = 2
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device_type = device_type
        self.random_seed = random.seed(random_seed)
        self.n_steps = n_steps
        self.gamma = gamma
        self.gamma_cumulative = np.array([self.gamma ** step_idx for step_idx in range(self.n_steps)])
        self.alpha = alpha
        self.position = 0
        self.size = 0  # Tracks the number of experiences in the buffer
        
        # Buffers for N-step transitions
        self.n_step_buffer_s_a_ns = deque(maxlen=n_steps)
        self.n_step_buffer_rewards = deque(maxlen=n_steps)
        self.n_step_buffer_dones = deque(maxlen=n_steps)
        
        # Experience tuple definition
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "next_discount_value"])

        # Beta annealing parameters
        self.beta_start = beta
        self.beta_end = 1.0
        self.beta_frames = 1000000.0  # Number of frames to anneal beta to self.beta_end
        self.frame_count: float = 0.0

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a new experience to the buffer.

        Args:
            states (array): Current states for all agents.
            actions (array): Actions taken by all agents.
            rewards (array): Rewards received by all agents.
            next_states (array): Next states for all agents.
            dones (array): Terminal flags for all agents.
        """

        # Add new transition (SARS-D values) to the rolling n-step buffer (deque)
        self.n_step_buffer_s_a_ns.append((states, actions, next_states))
        self.n_step_buffer_rewards.append(rewards)
        self.n_step_buffer_dones.append(dones)
        
        # Stop if there are not enough transitions for N-step return
        if len(self.n_step_buffer_rewards) < self.n_steps:
            return
                      
        # Get state and action from the first transition
        states  = self.n_step_buffer_s_a_ns[0][0]
        actions = self.n_step_buffer_s_a_ns[0][1]
        
        # Handle terminal states within the N-step window
        terminal_indices = np.where(np.array(self.n_step_buffer_dones))[0]  # Get indices of terminal states.
        if terminal_indices.size > 0:
            first_terminal_index = terminal_indices[0]  # Get the first terminal index
            n_step_rewards = np.dot(self.gamma_cumulative[:first_terminal_index + 1], np.array(self.n_step_buffer_rewards)[:first_terminal_index + 1])
            nth_states = self.n_step_buffer_s_a_ns[first_terminal_index][2]
            nth_dones  = self.n_step_buffer_dones[first_terminal_index]
            next_discount_value = self.gamma_cumulative[first_terminal_index]
        else:
            n_step_rewards = np.dot(self.gamma_cumulative, np.array(self.n_step_buffer_rewards))
            nth_states = self.n_step_buffer_s_a_ns[-1][2]
            nth_dones  = self.n_step_buffer_dones[-1]
            next_discount_value = self.gamma_cumulative[-1]

        next_discount_value = self.gamma * next_discount_value

        # Create N-step experience
        e = self.experience(states, actions, n_step_rewards, nth_states, nth_dones, next_discount_value)
        
        # Add experience to the replay buffer with maximum priority
        # (Add with maximum priority for new experiences (1.0 for the first one!)).
        max_priority = np.max(self.priorities) if self.size > 0 else 1.0
        if self.size < self.buffer_size:  # If buffer/memory has space available just add experience sample to memory.
            self.memory.append(e)
            self.size += 1
        else:                             # Otherwise replace another experience at self.position
            self.memory[self.position] = e
            
        self.priorities[self.position] = max_priority                # Update priority at position of replaced experience.
        self.position = (self.position + 1) % self.buffer_size  # Move self.positon further and rewind if needed!
        
    def sample(self):
        """
        Sample a batch of experiences from the buffer with prioritization.

        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones, indices, weights, next_discount_values).
        """

        if self.size < self.batch_size:
            return None

        # Update beta for importance sampling
        self.beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * (self.frame_count / self.beta_frames))
        self.frame_count += 1.0
            
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / np.sum(priorities)
        
        # Sample indices based on priorities
        indices_of_experiences_sampled = np.random.choice(self.size, self.batch_size, replace=False, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices_of_experiences_sampled]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Extract sampled experiences
        experiences_sampled = [self.memory[idx] for idx in indices_of_experiences_sampled]
        
        # Convert experiences to tensors by stacking the
        # experiences for all all agents
        states = torch.from_numpy(
            np.stack([e.state for e in experiences_sampled if e is not None], axis=0)
        ).float().to(self.device_type)  # Shape: (batch_size, agent_count, state_size)

        actions = torch.from_numpy(
            np.stack([e.action for e in experiences_sampled if e is not None], axis=0)
        ).float().to(self.device_type)  # Shape: (batch_size, agent_count, action_size)

        rewards = torch.from_numpy(
            np.stack([e.reward for e in experiences_sampled if e is not None], axis=0)
        ).float().to(self.device_type)  # Shape: (batch_size, agent_count, 1)

        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences_sampled if e is not None], axis=0)
        ).float().to(self.device_type)  # Shape: (batch_size, agent_count, state_size)

        dones = torch.from_numpy(
            np.stack([any(e.done) for e in experiences_sampled if e is not None], axis=0).astype(np.uint8)
        ).float().to(self.device_type).unsqueeze_(-1)  # Shape: (batch_size, 1)

        next_discount_values = torch.from_numpy(
            np.stack([e.next_discount_value for e in experiences_sampled if e is not None], axis=0)
        ).float().to(self.device_type)  # Shape: (batch_size, agent_count, 1)

        # Convert weights to tensor
        weights = torch.from_numpy(weights).float().to(self.device_type)
        
        return (states, actions, rewards, next_states, dones, weights, indices_of_experiences_sampled, next_discount_values)
    
    def update_priorities(self, indices_of_experiences_sampled, td_errors_abs):
        """
        Update priorities for sampled experiences.

        Args:
            indices (array): Indices of sampled experiences.
            td_errors_abs (array): New priorities based on TD errors.
        """
        
        self.priorities[indices_of_experiences_sampled] = td_errors_abs

    def __len__(self):
        """Return the current size of the memory."""
        return self.size
    