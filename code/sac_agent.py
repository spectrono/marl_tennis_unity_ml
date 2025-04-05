import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from base_agent import BaseAgent
from typing import List, Tuple
from replay_buffers import MultiAgentPrioritizedReplayBufferNStep


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) Agent

    The SAC agent is designed to maximize a trade-off between the reward and the entropy of the policy.
    This encourages exploration and helps in learning robust policies.
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            device_type='cpu',
            buffer_size=100000,
            batch_size=64,
            n_steps=1,
            random_seed: int = 0,
            actor_hidden_layers: List[int] = [256, 256],
            critic_hidden_layers: List[int] = [256, 256],
            lr_actor: float = 1e-4,
            lr_critic: float = 1e-4,
            lr_alpha: float = 1e-4,
            gamma: float = 0.99,
            tau: float = 1e-3,
            alpha: float = 0.2):
        """
        Initialize the SAC Agent

        Parameters:
        - state_size (int): Dimension of the state space.
        - action_size (int): Dimension of the action space.
        - device_type (str): Device to run the model on ('cpu' or 'cuda').
        - buffer_size (int): Size of the replay buffer.
        - batch_size (int): Batch size for training.
        - n_steps (int): Number of steps for n-step learning.
        - random_seed (int): Seed for random number generators.
        - actor_hidden_layers (List[int]): List of hidden layer sizes for the actor network.
        - critic_hidden_layers (List[int]): List of hidden layer sizes for the critic networks.
        - lr_actor (float): Learning rate for the actor network.
        - lr_critic (float): Learning rate for the critic networks.
        - lr_alpha (float): Learning rate for the entropy coefficient.
        - gamma (float): Discount factor for future rewards.
        - tau (float): Soft update coefficient for target networks.
        - alpha (float): Initial entropy coefficient.
        """
        super().__init__(device_type=device_type, gamma=gamma, n_steps=n_steps, buffer_size=buffer_size, batch_size=batch_size, random_seed=random_seed, tau=tau)

        self.state_size = state_size
        self.action_size = action_size
        self.lr_alpha = lr_alpha
        self.alpha = alpha

        # Actor Network (Policy)
        self.actor = self._build_actor_network(actor_hidden_layers).to(self.device_type)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Two critic networks for stability (double Q-learning)
        self.critic1 = self._build_critic_network(critic_hidden_layers).to(self.device_type)
        self.critic2 = self._build_critic_network(critic_hidden_layers).to(self.device_type)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # Target networks
        self.critic1_target = self._build_critic_network(critic_hidden_layers).to(self.device_type)
        self.critic2_target = self._build_critic_network(critic_hidden_layers).to(self.device_type)
        self.soft_update(self.critic1, self.critic1_target, 1.0)
        self.soft_update(self.critic2, self.critic2_target, 1.0)

        # Automatic entropy tuning
        self.target_entropy = -torch.prod(torch.Tensor([action_size])).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device_type)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay buffer
        self.replay_buffer = MultiAgentPrioritizedReplayBufferNStep(
            self.buffer_size, self.batch_size, self.device_type, random_seed, self.n_steps, gamma)

    def _build_actor_network(self, hidden_layers: List[int]) -> nn.Module:
        """
        Build the actor (policy) network

        Parameters:
        - hidden_layers (List[int]): List of hidden layer sizes.

        Returns:
        - nn.Module: Actor network.

        The actor network outputs the mean and log standard deviation of a Gaussian distribution,
        which is used to sample actions. The reparameterization trick is used to allow gradient-based
        optimization through the sampling process.        
        """
        layers = []
        in_features = self.state_size

        # Hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size

        # Mean and log std output layers for reparameterization trick
        mean_layer = nn.Linear(in_features, self.action_size)
        log_std_layer = nn.Linear(in_features, self.action_size)

        class ActorNetwork(nn.Module):
            def __init__(self, mean_layer, log_std_layer, base_layers, device_type):
                super().__init__()
                self.base_layers = nn.Sequential(*base_layers)
                self.mean_layer = mean_layer
                self.log_std_layer = log_std_layer

                # Clipping constants for log std
                self.LOG_STD_MIN = -20
                self.LOG_STD_MAX = 2

                self.device_type = device_type

            def forward(self, state, deterministic=False, with_log_prob=True):
                x = self.base_layers(state)
                mean = self.mean_layer(x)
                log_std = torch.clamp(
                    self.log_std_layer(x),
                    min=self.LOG_STD_MIN,
                    max=self.LOG_STD_MAX
                )
                std = torch.exp(log_std)

                # Reparameterization trick
                if deterministic:
                    action = mean
                else:
                    # Sample from the distribution
                    epsilon = torch.randn_like(mean).to(self.device_type)
                    action = mean + std * epsilon

                # Squash the action through tanh to bound in [-1, 1]
                squashed_action = torch.tanh(action)

                # Calculate log probability
                if with_log_prob:
                    # Change of variables for squashing
                    log_prob = torch.sum(
                        -log_std - 0.5 * np.log(2 * np.pi) -
                        0.5 * ((action - mean) / std) ** 2 -
                        torch.log(1 - squashed_action.pow(2) + 1e-6),
                        dim=-1,
                        keepdim=True
                    )
                else:
                    log_prob = None

                return squashed_action, log_prob, mean

        return ActorNetwork(mean_layer, log_std_layer, layers, self.device_type)

    def _build_critic_network(self, hidden_layers: List[int]) -> nn.Module:
        """
        Build the critic Q-network

        Parameters:
        - hidden_layers (List[int]): List of hidden layer sizes.

        Returns:
        - nn.Module: Critic network.

        The critic network estimates the Q-value of a state-action pair.
        """
        layers = []
        in_features = self.state_size + self.action_size

        # Hidden layers
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU()
            ])
            in_features = hidden_size

        # Output layer (Q-value)
        output_layer = nn.Linear(in_features, 1)

        class CriticNetwork(nn.Module):
            def __init__(self, output_layer, base_layers, device_type):
                super().__init__()
                self.base_layers = nn.Sequential(*base_layers)
                self.output_layer = output_layer

            def forward(self, state, action):
                x = torch.cat((state, action), dim=1)
                x = self.base_layers(x)
                return self.output_layer(x)

        return CriticNetwork(output_layer, layers, self.device_type)

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action from the policy

        Parameters:
        - state (np.ndarray): Current state.
        - deterministic (bool): Whether to sample from the policy or use the mean action.

        Returns:
        - np.ndarray: Selected action.

        This method uses the actor network to select an action given the current state.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device_type)

        with torch.no_grad():
            action, _, _ = self.actor(state, deterministic=deterministic)

        return action.cpu().numpy()[0]

    def learn(self, experiences: Tuple[torch.Tensor, ...], agent_idx):
        """
        Update policy and critics using the sampled experiences.

        Parameters:
        - experiences (Tuple[torch.Tensor, ...]): Tuple of tensors containing the experiences.
        - agent_idx (int): Index of the agent.

        This method updates the actor and critic networks using the sampled experiences from the replay buffer.
        It also performs automatic entropy tuning.
        """
        # Unpack experiences
        states, actions, n_step_rewards, next_states_nth, dones, weights, _, next_discount_values = experiences

        # Move tensors to the appropriate device
        states = states.to(self.device_type)
        actions = actions.to(self.device_type)
        n_step_rewards = n_step_rewards.to(self.device_type)
        next_states_nth = next_states_nth.to(self.device_type)
        dones = dones.to(self.device_type)
        weights = weights.to(self.device_type)
        next_discount_values = next_discount_values.to(self.device_type)

        # Compute targets for Q-networks
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs, _ = self.actor(next_states_nth[:, agent_idx, :])

            # Compute Q-targets using two critic networks
            q1_next = self.critic1_target(next_states_nth[:, agent_idx, :], next_actions)
            q2_next = self.critic2_target(next_states_nth[:, agent_idx, :], next_actions)
            min_q_next = torch.min(q1_next, q2_next)

            # Compute Q-targets with entropy bonus
            q_targets = n_step_rewards[:,agent_idx].unsqueeze(-1) + (1 - dones) * self.gamma * (min_q_next - self.alpha * next_log_probs)

        # Compute current Q-values
        q1_current = self.critic1(states[:, agent_idx, :], actions[:, agent_idx, :])
        q2_current = self.critic2(states[:, agent_idx, :], actions[:, agent_idx, :])

        # Critic loss (MSE)
        # with importance sampling
        td_errors_0 = q1_current - q_targets
        td_errors_1 = q2_current - q_targets
        critic1_loss = (weights * torch.square(td_errors_0)).mean()  # Use importance sampling weights from PER
        critic2_loss = (weights * torch.square(td_errors_1)).mean()  # Use importance sampling weights from PER

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        actions_pred, log_probs, _ = self.actor(states[:, agent_idx, :])
        q1_pred = self.critic1(states[:, agent_idx, :], actions_pred)
        q2_pred = self.critic2(states[:, agent_idx, :], actions_pred)
        min_q_pred = torch.min(q1_pred, q2_pred)

        # Policy loss with entropy regularization
        actor_loss = torch.mean(self.alpha * log_probs - min_q_pred)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Automatic entropy tuning
        alpha_loss = torch.mean(-self.log_alpha * (log_probs + self.target_entropy).detach())

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update alpha
        self.alpha = torch.exp(self.log_alpha).item()

        # Soft update of target networks
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)

        with torch.no_grad():
            td_min_errors_abs = torch.abs(torch.min(td_errors_0, td_errors_1)).detach().cpu().numpy().flatten()
            td_error_means = np.mean(td_min_errors_abs)


        # Fill the metrics dictionary
        metrics = {}  # Initialize the metrics dictionary
        metrics['actor_loss'] = actor_loss.item()
        metrics['critic1_loss'] = critic1_loss.item()
        metrics['critic2_loss'] = critic2_loss.item()
        metrics['q_value_critic1'] = q1_current.mean().item()
        metrics['q_value_critic2'] = q2_current.mean().item()
        metrics['td_error_means'] = td_error_means
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha'] = self.alpha

        return td_min_errors_abs, metrics
    
    def save(self, filepath: str):
        """
        Save model weights.

        Parameters:
        - filepath (str): Path to save the model weights.

        This method saves the weights of the actor and critic networks, as well as their optimizers.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """
        Load model weights.

        Parameters:
        - filepath (str): Path to load the model weights from.

        This method loads the weights of the actor and critic networks, as well as their optimizers.
        """
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])