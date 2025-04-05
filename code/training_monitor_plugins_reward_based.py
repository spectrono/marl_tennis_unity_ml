import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from training_monitor import TrainingMonitor

class RewardPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.rewards = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "reward":
            self.rewards[agent_idx].append(value)

    def visualize(self):
        for agent_idx, rewards in enumerate(self.rewards):
            plt.plot(rewards, label=f"Agent {agent_idx} Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Individual Rewards Over Time")
        plt.legend()
        plt.show()
    
    def visualize_intermediate_meta_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, rewards in enumerate(self.rewards):
            plt.plot(rewards, label=f"Agent {agent_idx} Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Individual Rewards Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, rewards in enumerate(self.rewards):
            np.save(os.path.join(directory, f"reward_agent_{agent_idx}.npy"), rewards)

    def load(self, directory):
        for agent_idx in range(len(self.rewards)):
            file_path = os.path.join(directory, f"reward_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.rewards[agent_idx] = np.load(file_path).tolist()


class CooperativeRewardPlugin(TrainingMonitor):
    def __init__(self, target_score):
        self.cooperative_rewards = []
        self.target_score: Any = target_score
        self.rolling_mean_window = 100

    def log_metric(self, metric_name, value, unused=None):
        if metric_name == "cooperative_reward":
            self.cooperative_rewards.append(value)

    def visualize_intermediate_meta_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        plt.plot(self.cooperative_rewards, label=f"Cooperative Reward")
        rolling_mean_rewards = pd.Series(self.cooperative_rewards).rolling(window=100).mean()
        plt.plot(rolling_mean_rewards, color='orange', linestyle='--', label='Rolling Mean (100 Episods)')
        plt.axhline(y=self.target_score, color='magenta', linestyle='--', label=f'Target Score {self.target_score:.2f}')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Cooperative Reward Over Time")
        plt.legend()

    def visualize(self):
        plt.plot(self.cooperative_rewards, label=f"Cooperative Reward")
        rolling_mean_rewards = pd.Series(self.cooperative_rewards).rolling(window=100).mean()
        plt.plot(rolling_mean_rewards, color='orange', linestyle='--', label='Rolling Mean (100 Episods)')
        plt.axhline(y=self.target_score, color='magenta', linestyle='--', label=f'Target Score {self.target_score:.2f}')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Cooperative Reward Over Time")
        plt.legend()
        plt.show()

    def save(self, directory):
        np.save(os.path.join(directory, "cooperative_reward.npy"), self.cooperative_rewards)

    def load(self, directory):
        file_path = os.path.join(directory, "cooperative_reward.npy")
        if os.path.exists(file_path):
            self.cooperative_rewards = np.load(file_path).tolist()
