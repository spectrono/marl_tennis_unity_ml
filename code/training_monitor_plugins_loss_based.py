import matplotlib.pyplot as plt
import numpy as np
from training_monitor import TrainingMonitor
import os


class ActorLossPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.actor_losses = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "actor_loss":
            self.actor_losses[agent_idx].append(value)

    def visualize(self):
        for agent_idx, losses in enumerate(self.actor_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Actor Loss")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Actor Losses Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_loss_based_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, losses in enumerate(self.actor_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Actor Loss")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Actor Losses Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, losses in enumerate(self.actor_losses):
            np.save(os.path.join(directory, f"actor_loss_agent_{agent_idx}.npy"), losses)

    def load(self, directory):
        for agent_idx in range(len(self.actor_losses)):
            file_path = os.path.join(directory, f"actor_loss_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.actor_losses[agent_idx] = np.load(file_path).tolist()

class Critic1LossPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.critic_losses = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "critic1_loss":
            self.critic_losses[agent_idx].append(value)

    def visualize(self):
        for agent_idx, losses in enumerate(self.critic_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Critic 1 Loss")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Critic 1 Losses Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_loss_based_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, losses in enumerate(self.critic_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Critic 1 Loss")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Critic 1 Losses Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, losses in enumerate(self.critic_losses):
            np.save(os.path.join(directory, f"critic1_loss_agent_{agent_idx}.npy"), losses)

    def load(self, directory):
        for agent_idx in range(len(self.critic_losses)):
            file_path = os.path.join(directory, f"critic1_loss_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.critic_losses[agent_idx] = np.load(file_path).tolist()

class Critic2LossPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.critic_losses = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "critic2_loss":
            self.critic_losses[agent_idx].append(value)

    def visualize(self):
        for agent_idx, losses in enumerate(self.critic_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Critic 2 Loss")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Critic 2 Losses Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_loss_based_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, losses in enumerate(self.critic_losses):
            plt.plot(losses, label=f"Agent {agent_idx} Critic 2 Loss")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Critic 2 Losses Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, losses in enumerate(self.critic_losses):
            np.save(os.path.join(directory, f"critic2_loss_agent_{agent_idx}.npy"), losses)

    def load(self, directory):
        for agent_idx in range(len(self.critic_losses)):
            file_path = os.path.join(directory, f"critic2_loss_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.critic_losses[agent_idx] = np.load(file_path).tolist()

class Critic1QValuePlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.q_values = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "q_value_critic1":
            self.q_values[agent_idx].append(value)

    def visualize(self):
        for agent_idx, qvals in enumerate(self.q_values):
            plt.plot(qvals, label=f"Agent {agent_idx} Critic 1 Q-Values")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Q-Value")
        plt.title("Critic 1 Q-Values Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_loss_based_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, qvals in enumerate(self.q_values):
            plt.plot(qvals, label=f"Agent {agent_idx} Critic 1 Q-Values")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Q-Value")
        plt.title("Critic 1 Q-Values Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, qvals in enumerate(self.q_values):
            np.save(os.path.join(directory, f"q_value_critic1_agent_{agent_idx}.npy"), qvals)

    def load(self, directory):
        for agent_idx in range(len(self.q_values)):
            file_path = os.path.join(directory, f"q_value_critic1_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.q_values[agent_idx] = np.load(file_path).tolist()


class Critic2QValuePlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.q_values = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "q_value_critic2":
            self.q_values[agent_idx].append(value)

    def visualize(self):
        for agent_idx, qvals in enumerate(self.q_values):
            plt.plot(qvals, label=f"Agent {agent_idx} Critic 2 Q-Values")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Q-Value")
        plt.title("Critic 2 Q-Values Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_loss_based_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, qvals in enumerate(self.q_values):
            plt.plot(qvals, label=f"Agent {agent_idx} Critic 2 Q-Values")
        plt.yscale("symlog")
        plt.xlabel("Step")
        plt.ylabel("Q-Value")
        plt.title("Critic 2 Q-Values Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, qvals in enumerate(self.q_values):
            np.save(os.path.join(directory, f"q_value_critic2_agent_{agent_idx}.npy"), qvals)

    def load(self, directory):
        for agent_idx in range(len(self.q_values)):
            file_path = os.path.join(directory, f"q_value_critic2_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.q_values[agent_idx] = np.load(file_path).tolist()


class TDErrorPlugin(TrainingMonitor):
    def __init__(self, agent_count):
        self.td_error_means = [[] for _ in range(agent_count)]

    def log_metric(self, metric_name, value, agent_idx):
        if metric_name == "td_error_means":
            self.td_error_means[agent_idx].append(value)

    def visualize(self):
        for agent_idx, td_error in enumerate(self.td_error_means):
            plt.plot(td_error, label=f"Agent {agent_idx} TD-Error Means of Min")
        plt.xlabel("Step")
        plt.ylabel("TD-Error Mean")
        plt.title("TD-Error Over Time")
        plt.legend()
        plt.show()

    def visualize_intermediate_loss_based_results(self, plt, plot_index):
        plt.subplot(3, 2, plot_index)
        for agent_idx, losses in enumerate(self.td_error_means):
            plt.plot(losses, label=f"Agent {agent_idx} TD-Error Means of Min")
        plt.xlabel("Step")
        plt.ylabel("TD-Error Mean")
        plt.title("TD-Error Over Time")
        plt.legend()

    def save(self, directory):
        for agent_idx, losses in enumerate(self.td_error_means):
            np.save(os.path.join(directory, f"td_error_agent_{agent_idx}.npy"), losses)

    def load(self, directory):
        for agent_idx in range(len(self.td_error_means)):
            file_path = os.path.join(directory, f"td_error_agent_{agent_idx}.npy")
            if os.path.exists(file_path):
                self.td_error_means[agent_idx] = np.load(file_path).tolist()
