import matplotlib.pyplot as plt
import os
import json  # Import JSON module for saving dictionaries


class TrainingMonitor:
    def __init__(self, plugins):
        self.plugins = plugins
        self.agent_parameters = {}
        self.agent_actor_networks = {}
        self.agent_critic_networks = {}

    def log_agent_parameters(self, agent_id, parameters):
        self.agent_parameters[agent_id] = parameters

    def log_agent_actor_network_architecture(self, agent_id, network):
            """
            Log the actor network architecture for a specific agent.

            Args:
                agent_id (int): The ID of the agent.
                network (torch.nn.Module): The PyTorch model representing the agent's network.
            """
            self.agent_actor_networks[agent_id] = str(network)  # Store the string representation of the network

    def log_agent_critic_network_architecture(self, agent_id, network):
            """
            Log the actor network architecture for a specific agent.

            Args:
                agent_id (int): The ID of the agent.
                network (torch.nn.Module): The PyTorch model representing the agent's network.
            """
            self.agent_critic_networks[agent_id] = str(network)  # Store the string representation of the network

    def log_metric(self, metric_name, value, agent_idx=None):
        for plugin in self.plugins:
            if agent_idx is not None:
                plugin.log_metric(metric_name, value, agent_idx)
            else:
                plugin.log_metric(metric_name, value, None)

    def visualize(self):
        for plugin in self.plugins:
            plugin.visualize()

    def save(self, directory):
        for plugin in self.plugins:
            plugin.save(directory)

    def load(self, directory):
        for plugin in self.plugins:
            plugin.load(directory)

    def save_current_state_plots_to_disk(self, directory, episode):
        plt.figure(figsize=(18, 12))
        plot_index = 1
        for plugin in self.plugins:
            if hasattr(plugin, 'visualize_intermediate_loss_based_results'):
                plugin.visualize_intermediate_loss_based_results(plt, plot_index)
                plot_index += 1
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"intermediate_loss_based_plots_episode_{episode}.png"))
        plt.close()

        plt.figure(figsize=(18, 12))
        plot_index = 1
        for plugin in self.plugins:
            if hasattr(plugin, 'visualize_intermediate_meta_results'):
                plugin.visualize_intermediate_meta_results(plt, plot_index)
                plot_index += 1

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"intermediate_meta_plots_episode_{episode}.png"))
        plt.close()

    def save_agent_parameters_as_json(self, directory):
            """
            Save the logged agent parameters to disk as JSON files.

            Args:
                directory (str): The directory where the JSON files will be saved.
            """
            if not os.path.exists(directory):
                os.makedirs(directory)  # Ensure the directory exists

            for agent_id, parameters in self.agent_parameters.items():
                file_path = os.path.join(directory, f"agent_parameters_{agent_id}.json")
                with open(file_path, 'w') as json_file:
                    json.dump(parameters, json_file, indent=4)  # Save parameters as a JSON file


    def save_agent_network_architectures_as_json(self, directory):
        """
        Save the logged agent network architectures to disk as JSON files.

        Args:
            directory (str): The directory where the JSON files will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)  # Ensure the directory exists

        # Save actor networks
        for agent_id, network in self.agent_actor_networks.items():
            file_path = os.path.join(directory, f"agent_actor_network_{agent_id}.json")
            with open(file_path, 'w') as json_file:
                json.dump({"actor_architecture": network}, json_file, indent=4)

        # Save critic networks
        for agent_id, network in self.agent_critic_networks.items():
            file_path = os.path.join(directory, f"agent_critic_network_{agent_id}.json")
            with open(file_path, 'w') as json_file:
                json.dump({"critic_architecture": network}, json_file, indent=4)  # Save network as a JSON file