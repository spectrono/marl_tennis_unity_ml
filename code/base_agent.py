import torch
import torch.nn as nn
import random
import numpy as np


class BaseAgent(nn.Module):
    def __init__(self, device_type='cpu', gamma=0.99, n_steps=1, buffer_size=100000, batch_size=64, lr_actor=1e-4, lr_critic=1e-4, tau=1e-3, random_seed=None):
        super(BaseAgent, self).__init__()
        self.device_type = torch.device(device_type)
        self.gamma = gamma
        self.n_steps = n_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_local = None
        self.actor_target = None
        self.critic_local = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.replay_buffer = None
        self.metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'mean_Q_value': [],
            'mean_td_error': []
        }
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau

        # Set random seed for reproducibility
        if random_seed is not None:
            self.set_seed(random_seed)

    def set_seed(self, random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self.device_type == 'cuda':
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def act(self, state):
        raise NotImplementedError

    def learn(self, experiences):
        raise NotImplementedError

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def get_metrics(self):
        return self.metrics

    def get_parameters(self):
        return {
            'device_type': self.device_type,
            'gamma': self.gamma,
            'n_steps': self.n_steps,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'tau': self.tau,
            'random_seed': self.random_seed if hasattr(self, 'random_seed') else None
        }

    def set_learning_rates(self, lr_actor, lr_critic):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr_actor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr_critic

    def set_tau(self, tau):
        self.tau = tau

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps