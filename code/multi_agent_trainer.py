import time
import numpy as np
import torch
import argparse 
import os
from unity_ml_tennis_environment_adapter import UnityMLTennisEnvironmentAdapter
from sac_agent import SACAgent
from replay_buffers import MultiAgentPrioritizedReplayBufferNStep
from training_monitor import TrainingMonitor
from training_monitor_plugins_meta_data import (
    EpisodeLengthPlugin,
    LearningProgressPlugin,
    AlphaLossPlugin,
    AlphaPlugin)
from training_monitor_plugins_reward_based import (
    RewardPlugin,
    CooperativeRewardPlugin,)
from training_monitor_plugins_loss_based import (
    ActorLossPlugin,
    Critic1LossPlugin,
    Critic2LossPlugin,
    Critic1QValuePlugin,
    Critic2QValuePlugin,
    TDErrorPlugin)


class MultiAgentTrainer:
    """
    Centralized training framework for multiple agents with a shared replay buffer and combined priorities.
    Supports Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self,
                 agent_class,
                 agents_count: int,
                 state_size: int,
                 action_size: int,
                 agent_params: dict = None):
        """
        Initialize multi-agent training setup with a shared replay buffer and combined priorities.
        """
        self.agents_count = agents_count
        self.agents = [
            agent_class(state_size, action_size, **(agent_params or {}))
            for _ in range(agents_count)]

        # Use a shared replay buffer with combined priorities
        self.shared_replay_buffer = MultiAgentPrioritizedReplayBufferNStep(
            buffer_size=agent_params.get('buffer_size', 100000),
            batch_size=agent_params.get('batch_size', 64),
            device_type=agent_params.get('device_type', 'cpu'),
            random_seed=agent_params.get('random_seed', 0),
            n_steps=agent_params.get('n_steps', 1),
            gamma=agent_params.get('gamma', 0.99))

    def train(self,
              log_folder,
              checkpoint_frequency,
              print_out_frequency,
              training_monitor,
              env,
              max_episodes_count: int = 30,
              max_time_steps_per_episode: int = 1000,
              target_score: float = 0.5):
        """
        Train multiple agents in the environment using a shared replay buffer with combined priorities.
        """

        # Create log directories
        log_dir = log_folder
        checkpoints_dir = os.path.join(log_dir, "checkpoints")
        best_dir = os.path.join(log_dir, "best")
        solved_dir = os.path.join(log_dir, "solved")
        final_dir = os.path.join(log_dir, "final")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)
        os.makedirs(solved_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        # Log agent parameters and network arcitectures at the start of training
        for agent_id, agent in enumerate(self.agents):
            training_monitor.log_agent_parameters(agent_id, agent.get_parameters())
            if agent.actor is not None:
                training_monitor.log_agent_actor_network_architecture(agent_id, agent.actor)
            if agent.critic1 is not None:
                training_monitor.log_agent_critic_network_architecture(agent_id, agent.critic1)
        training_monitor.save_agent_parameters_as_json(log_dir)
        training_monitor.save_agent_network_architectures_as_json(log_dir)

        # Remeber average cooperative reward plugin for controlling the training process
        cooperative_reward_plugin = None
        for plugin in training_monitor.plugins:
            if isinstance(plugin, CooperativeRewardPlugin):
                cooperative_reward_plugin: CooperativeRewardPlugin = plugin
        is_solved = False
        best_cooperative_reward = -np.inf

        # Training loop
        for episode_idx in range(1, max_episodes_count + 1):
            # Reset environment
            states = env.reset(use_train_mode=True)
            episode_lengths = [0] * self.agents_count
            cooperative_reward = 0

            # Episode tracking
            episode_rewards = np.zeros(self.agents_count)

            for time_step_idx in range(max_time_steps_per_episode):

                episode_lengths = [length + 1 for length in episode_lengths]

                actions = np.array([agent.act(state) for agent, state in zip(self.agents, states)])
                next_states, rewards, dones = env.step(actions)
                self.shared_replay_buffer.add(states, actions, rewards, next_states, dones)

                # Accumulate rewards
                episode_rewards += rewards
                cooperative_reward += max(rewards)
                states = next_states

                # Learn for each agent if enough samples in replay buffer
                if len(self.shared_replay_buffer) > self.shared_replay_buffer.batch_size:

                    combined_experiences = self.shared_replay_buffer.sample()

                    combined_td_errors = []
                    for agent_idx, agent in enumerate(self.agents):                        
                        td_errors, learning_metrics = agent.learn(combined_experiences, agent_idx)
                        combined_td_errors.append(td_errors)

                        # Log learning metrics
                        for metric_name, value in learning_metrics.items():
                            training_monitor.log_metric(metric_name, value, agent_idx)

                    # Update priorities in the replay buffer with combined TD errors
                    combined_priorities = np.max(combined_td_errors, axis=0)  # Use max TD error as priority OR MIN?!?!?
                    combined_priorities = combined_priorities + 1e-5  # Add small constant to avoid zero priority
                    indices = combined_experiences[6]
                    self.shared_replay_buffer.update_priorities(indices, combined_priorities)

                # Check for episode termination
                if np.any(dones):
                    break
            
            # Update training monitor with episode statistics
            training_monitor.log_metric("cooperative_reward", cooperative_reward)
            for agent_idx, reward in enumerate(episode_rewards):
                training_monitor.log_metric("reward", reward, agent_idx)
                training_monitor.log_metric("episode_length", episode_lengths[agent_idx], agent_idx)

            # Verbose output
            if (episode_idx) % print_out_frequency == 0:
                if len(cooperative_reward_plugin.cooperative_rewards) > 100:
                    latetst_cooperative_reward = np.mean(cooperative_reward_plugin.cooperative_rewards[-100:])
                else:
                    latetst_cooperative_reward = -np.inf
                print(f"Episode {episode_idx}/{max_episodes_count}\tCurrent Cooperative Reward: {cooperative_reward:.2f}\tAverage Cooperative Reward: {latetst_cooperative_reward:.2f}")

            # Save best model if cooperative reward exceeds previous best
            if cooperative_reward > best_cooperative_reward:
                best_cooperative_reward = cooperative_reward
                training_monitor.save_current_state_plots_to_disk(best_dir, episode_idx)
                for agent_idx, agent in enumerate(self.agents):
                    agent_save_path = os.path.join(best_dir, f"best_agent{agent_idx}_at_episode_{episode_idx}.pth")
                    self.agents[agent_idx].save(agent_save_path)
                print(f">>> Best model saved at episode {episode_idx} with cooperative reward: {best_cooperative_reward:.2f}")

            # Save checkpoints (training monitor and agent states)
            if (episode_idx) % checkpoint_frequency == 0:
                training_monitor.save_current_state_plots_to_disk(checkpoints_dir, episode_idx)
                for agent_idx, agent in enumerate(self.agents):
                    agent_save_path = os.path.join(checkpoints_dir, f"checkpoint_agent{agent_idx}_at_episode_{episode_idx}.pth")
                    self.agents[agent_idx].save(agent_save_path)

            # Check for environment solution
            if self._check_solve_condition(cooperative_reward_plugin.cooperative_rewards, target_score) and not is_solved:
                print(f">>> Environment solved in {episode_idx} episodes!")
                training_monitor.save_current_state_plots_to_disk(solved_dir, episode_idx)
                for agent_idx, agent in enumerate(self.agents):
                    agent_save_path = os.path.join(solved_dir, f"solved_agent{agent_idx}_at_episode_{episode_idx}.pth")
                    self.agents[agent_idx].save(agent_save_path)
                training_monitor.save(solved_dir)
                is_solved = True

        # Save complete training monitor state and agent parameters
        training_monitor.save(final_dir)
        training_monitor.save_current_state_plots_to_disk(final_dir, episode_idx)
        for agent_idx, agent in enumerate(self.agents):
            agent_save_path = os.path.join(final_dir, f"final_agent{agent_idx}_at_episode_{episode_idx}.pth")
            self.agents[agent_idx].save(agent_save_path)

    def _check_solve_condition(self, cooperative_rewards, target_score: float) -> bool:
        """
        Check if the environment is considered solved.
        """
        if ((len(cooperative_rewards) > 100) and
            (np.mean(cooperative_rewards[-100:]) > target_score)):
            return True
        return False

def train_agent(target_score, max_episodes_count, log_folder, checkpoint_frequency, print_out_frequency):

    # Initialize environment
    env_adapter = UnityMLTennisEnvironmentAdapter()
    agent_count = env_adapter.get_agent_count()

    # Initialize the trainer with SAC agents
    trainer = MultiAgentTrainer(
        agent_class=SACAgent,
        agents_count=env_adapter.get_agent_count(),
        state_size=env_adapter.get_observation_size(),
        action_size=env_adapter.get_action_size(),
        agent_params={
            'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
            'buffer_size': 100000,
            'batch_size': 64,
            'n_steps': 1,
            'random_seed': 42,  # Set random seed for reproducibility
            'actor_hidden_layers': [256, 256],
            'critic_hidden_layers': [256, 256],
            'lr_actor': 1e-4,
            'lr_critic': 1e-4,
            'lr_alpha': 1e-4,
            'gamma': 0.99,
            'tau': 1e-3,
            'alpha': 0.2
        }
    )

    # Initialize logger plugins
    actor_loss_plugin = ActorLossPlugin(agent_count)
    td_error_plugin = TDErrorPlugin(agent_count)
    critic1_loss_plugin = Critic1LossPlugin(agent_count)
    critic1_qvalue_plugin = Critic1QValuePlugin(agent_count)
    critic2_loss_plugin = Critic2LossPlugin(agent_count)
    critic2_qvalue_plugin = Critic2QValuePlugin(agent_count)

    reward_plugin = RewardPlugin(agent_count)
    cooperative_reward_plugin = CooperativeRewardPlugin(target_score=target_score)
    learning_progress_plugin = LearningProgressPlugin(agent_count)
    episode_length_plugin = EpisodeLengthPlugin(agent_count, True)
    alpha_loss_plugin = AlphaLossPlugin(agent_count)
    alpha_plugin = AlphaPlugin(agent_count)

    # Initialize training monitor with plugins
    training_monitor = TrainingMonitor(
        plugins=[actor_loss_plugin, td_error_plugin, critic1_loss_plugin, critic1_qvalue_plugin, critic2_loss_plugin, critic2_qvalue_plugin, reward_plugin, cooperative_reward_plugin, learning_progress_plugin, episode_length_plugin, alpha_loss_plugin, alpha_plugin])

    # Train the agents
    trainer.train(log_folder, checkpoint_frequency, print_out_frequency, training_monitor, env_adapter, max_episodes_count=max_episodes_count, max_time_steps_per_episode=1000, target_score=target_score)

    # Close the environment
    env_adapter.close_env()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load and visualize training data.")
    parser.add_argument("--log_folder_base", type=str, default="logs", help="Base path to the folder containing the training data. Wil be appended by current date and time.")
    parser.add_argument("--checkpoint_frequency", type=int, default=500, help="Frequency of logging intermediate results.")
    parser.add_argument("--printout_frequency", type=int, default=100, help="Frequency of printing intermediate results to the console.")
    parser.add_argument("--target_score", type=float, default=0.5, help="Cooperative target score.")
    parser.add_argument("--max_episodes_count", type=int, default=2000, help="Maximum nb of training episodes.")
    args = parser.parse_args()

    log_dir = args.log_folder_base + "_" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    train_agent(args.target_score, args.max_episodes_count, log_dir, args.checkpoint_frequency, args.printout_frequency)
