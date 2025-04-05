import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from training_monitor import TrainingMonitor


class EpisodeLengthPlugin(TrainingMonitor):
    def __init__(self, agent_count, are_lengths_for_all_agents_equal):
        self.episode_lengths = [[] for _ in range(agent_count)]
        self.are_lengths_for_all_agents_equal = are_lengths_for_all_agents_equal
        self.rolling_mean_window = 100

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "episode_length":
            self.episode_lengths[agent_idx].append(value)

    def visualize(self):

        if self.are_lengths_for_all_agents_equal:
            plt.plot(self.episode_lengths[0], label="Episode Length")
            rolling_mean_episode_lengths = pd.Series(self.episode_lengths[0]).rolling(window=self.rolling_mean_window).mean()
            plt.plot(rolling_mean_episode_lengths, color='orange', linestyle='--', label='Rolling Mean (100 Episods)')
        else:
            for agent_idx, lengths in enumerate(self.episode_lengths):
                plt.plot(lengths, label=f"Agent {agent_idx} Episode Length")
                rolling_mean_episode_lengths = pd.Series(lengths).rolling(window=self.rolling_mean_window).mean()
                plt.plot(rolling_mean_episode_lengths, color='orange', linestyle='--', label=f'Agent {agent_idx} Rolling Mean (100 Episods)')
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Length Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_meta_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        if self.are_lengths_for_all_agents_equal:
            plt.plot(self.episode_lengths[0], label="Episode Length")
            rolling_mean_episode_lengths = pd.Series(self.episode_lengths[0]).rolling(window=self.rolling_mean_window).mean()
            plt.plot(rolling_mean_episode_lengths, color='orange', linestyle='--', label='Rolling Mean (100 Episods)')
        else:
            for agent_idx, lengths in enumerate(self.episode_lengths):
                plt.plot(lengths, label=f"Agent {agent_idx} Episode Length")
                rolling_mean_episode_lengths = pd.Series(lengths).rolling(window=self.rolling_mean_window).mean()
                plt.plot(rolling_mean_episode_lengths, color='orange', linestyle='--', label=f'Agent {agent_idx} Rolling Mean (100 Episods)')
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Length Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, lengths in enumerate(self.episode_lengths):
            np.save(os.path.join(directory, f"episode_length_agent_{agent_idx}.npy"), lengths)

    def load(self, directory):
        for agent_idx in range(len(self.episode_lengths)):
            file_path = os.path.join(directory, f"episode_length_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.episode_lengths[agent_idx] = np.load(file_path).tolist()


class LearningProgressPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.rewards = [[] for _ in range(agent_count)]
        self.learning_progress = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "reward":
            self.rewards[agent_idx].append(value)
            if len(self.rewards[agent_idx]) > 200:
                # Compute recent and previous averages
                recent_avg = np.mean(self.rewards[agent_idx][-100:])
                previous_avg = np.mean(self.rewards[agent_idx][-200:-100])

                # Compute normalized progress
                epsilon = 1e-6  # Small constant to prevent division by zero
                progress = (recent_avg - previous_avg) / (abs(recent_avg) + abs(previous_avg) + epsilon)

                # Append progress to the learning progress list
                self.learning_progress[agent_idx].append(progress)

    def visualize(self):
        for agent_idx, progress in enumerate(self.learning_progress):
            # Pad the progress data with NaN values for the first 199 episodes
            padded_progress = [np.nan] * 199 + progress
            plt.plot(range(1, len(padded_progress) + 1), padded_progress, label=f"Agent {agent_idx} Learning Progress")
        plt.axhline(y=0, color='magenta', linestyle='--', label=f'ZERO progress border')
        plt.xlabel("Episode")
        plt.ylabel("Progress Indicator")
        plt.title("Learning Progress Over Time")
        plt.legend()
        plt.xlim(1, len(padded_progress))  # Ensure x-axis starts at episode 1
        plt.show()

    def visualize_intermediate_meta_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, progress in enumerate(self.learning_progress):
            # Pad the progress data with NaN values for the first 199 episodes
            padded_progress = [np.nan] * 199 + progress
            plt.plot(range(1, len(padded_progress) + 1), padded_progress, label=f"Agent {agent_idx} Learning Progress")
        plt.axhline(y=0, color='magenta', linestyle='--', label=f'ZERO progress border')
        plt.xlabel("Episode")
        plt.ylabel("Progress Indicator")
        plt.title("Learning Progress Over Time")
        plt.legend()
        plt.xlim(1, len(padded_progress))  # Ensure x-axis starts at episode 1

    def save(self, directory):
        for agent_idx, progress in enumerate(self.learning_progress):
            np.save(os.path.join(directory, f"learning_progress_agent_{agent_idx}.npy"), progress)

    def load(self, directory):
        for agent_idx in range(len(self.learning_progress)):
            file_path = os.path.join(directory, f"learning_progress_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.learning_progress[agent_idx] = np.load(file_path).tolist()


class AlphaLossPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.alpha_losses = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "alpha_loss":
            self.alpha_losses[agent_idx].append(value)

    def visualize(self):
        for agent_idx, losses in enumerate(self.alpha_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Alpha Loss")
        plt.yscale("symlog")
        plt.xlabel("Steps")
        plt.ylabel("Alpha Loss")
        plt.title("Individual Alpha Losses Over Time")
        plt.legend()
        plt.show()
    
    def visualize_intermediate_meta_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, losses in enumerate(self.alpha_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Alpha Loss")
        plt.yscale("symlog")
        plt.xlabel("Steps")
        plt.ylabel("Alpha Loss")
        plt.title("Individual Alpha Losses Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, losses in enumerate(self.alpha_losses):
            np.save(os.path.join(directory, f"alpha_loss_agent_{agent_idx}.npy"), losses)

    def load(self, directory):
        for agent_idx in range(len(self.alpha_losses)):
            file_path = os.path.join(directory, f"alpha_loss_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.alpha_losses[agent_idx] = np.load(file_path).tolist()


class AlphaPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.alphas = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "alpha":
            self.alphas[agent_idx].append(value)

    def visualize(self):
        for agent_idx, alphas in enumerate(self.alphas):
            plt.plot(alphas, label=f"Agent {agent_idx} Alpha")
        plt.yscale("symlog")
        plt.xlabel("Steps")
        plt.ylabel("Alpha")
        plt.title("Individual Alphas Over Time")
        plt.legend()
        plt.show()
    
    def visualize_intermediate_meta_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, alphas in enumerate(self.alphas):
            plt.plot(alphas, label=f"Agent {agent_idx} Alpha")
        plt.yscale("symlog")
        plt.xlabel("Steps")
        plt.ylabel("Alpha")
        plt.title("Individual Alphas Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, alphas in enumerate(self.alphas):
            np.save(os.path.join(directory, f"alpha_agent_{agent_idx}.npy"), alphas)

    def load(self, directory):
        for agent_idx in range(len(self.alphas)):
            file_path = os.path.join(directory, f"alpha_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.alphas[agent_idx] = np.load(file_path).tolist()

