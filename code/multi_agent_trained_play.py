import argparse
import json
from os import path
import numpy as np
from sac_agent import SACAgent
from unity_ml_tennis_environment_adapter import UnityMLTennisEnvironmentAdapter

def trained_play(
        env_adapter: UnityMLTennisEnvironmentAdapter,
        number_of_episodes_to_play: int,
        path_to_trained_model_agent_0=None,
        path_to_parameters_agent_0=None,
        path_to_trained_model_agent_1=None,
        path_to_parameters_agent_1=None):
    """
    Simulates plays in the Unity ML Tennis environment using trained agents.

    Args:
        env_adapter (UnityMLTennisEnvironmentAdapter): The environment adapter for the Unity ML Tennis environment.
        number_of_episodes_to_play (int): The number of episodes to play.
        path_to_trained_model_agent_0 (str): Path to the trained model for agent 0.
        path_to_parameters_agent_0 (str): Path to the parameters for agent 0.
        path_to_trained_model_agent_1 (str): Path to the trained model for agent 1.
        path_to_parameters_agent_1 (str): Path to the parameters for agent 1.

    This function will play a specified number of episodes in the Unity ML Tennis environment
    using two trained agents. It tracks and prints the best score achieved across
    all episodes.
    """

    # Load agent 0
    agent_0 = None
    if path_to_trained_model_agent_0 and path_to_parameters_agent_0:
        with open(path_to_parameters_agent_0, 'r') as f:
            agent_params = json.load(f)
            agent_0 = SACAgent(env_adapter.get_observation_size(), env_adapter.get_action_size(), **(agent_params or {}))
            agent_0.load(path_to_trained_model_agent_0)            
            print(f"Agent 0 loaded from {path_to_trained_model_agent_0} with parameters from {path_to_parameters_agent_0}")
    else:
        print("No trained model or parameters provided for agent 0.")
        return

    # Load agent 1
    agent_1 = None
    if path_to_trained_model_agent_1 and path_to_parameters_agent_1:
        with open(path_to_parameters_agent_1, 'r') as f:
            agent_params = json.load(f)
            agent_1 = SACAgent(env_adapter.get_observation_size(), env_adapter.get_action_size(), **(agent_params or {}))
            agent_1.load(path_to_trained_model_agent_1)            
            print(f"Agent 1 loaded from {path_to_trained_model_agent_1} with parameters from {path_to_parameters_agent_1}")
    else:
        print("No trained model or parameters provided for agent 1.")
        return

    agents = [agent_0, agent_1]  # List of agents to be used in the environment

    best_score = -np.inf  # Initialize the best score to negative infinity
    agent_count = env_adapter.get_agent_count()  # Get the number of agents in the environment
    action_size = env_adapter.get_action_size()  # Get the size of the action space

    for episode_idx in range(number_of_episodes_to_play):
        env_adapter.reset(use_train_mode=False)  # Reset the environment before each episode
        states = env_adapter.get_observations()  # Get the initial observations for each agent
        episode_rewards = np.zeros(agent_count)  # Initialize scores for each agent
        cooperative_reward = 0
        episode_length = 0
        while True:
            episode_length += 1
            actions = np.array([agent.act(state, deterministic=True) for agent, state in zip(agents, states)])
            next_states, rewards, dones = env_adapter.step(actions)

            # Accumulate rewards
            episode_rewards += rewards
            cooperative_reward += max(rewards)
            states = next_states

            if np.any(dones):  # If any agent is done, exit the loop
                break

        print(f'Episode {episode_idx} ---> Max total score (over both agents): {cooperative_reward:0.2f} after {episode_length} steps')

        if cooperative_reward > best_score:  # Update the best score if the current score is higher
            best_score = cooperative_reward
            print(f'Best score updated: {best_score:0.2f}')
    print(f'>>> Best score: {best_score:0.2f} <<< after {number_of_episodes_to_play} episodes')

if __name__ == "__main__":
    """
    Main function to create the Unity ML Tennis environment and play with trained agents.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Play episodes in the Unity ML Tennis environment using trained agents.")
    parser.add_argument("--number_of_episodes_to_play", type=int, default=200, help="Number of episodes to play.")
    parser.add_argument("--path_to_trained_model_agent_0", type=str, default="logs/solved_agent0_at_episode_1804.pth", help="Path to the trained model for agent 0.")
    parser.add_argument("--path_to_parameters_agent_0", type=str, default="logs/agent_parameters_0.json", help="Path to the parameters for agent 0.")
    parser.add_argument("--path_to_trained_model_agent_1", type=str, default="logs/solved_agent1_at_episode_1804.pth", help="Path to the trained model for agent 1.")
    parser.add_argument("--path_to_parameters_agent_1", type=str, default="logs/agent_parameters_1.json", help="Path to the parameters for agent 1.")
    args = parser.parse_args()

    # Create the tennis environment
    env_adapter = UnityMLTennisEnvironmentAdapter(use_headless=False)

    # Call the trained_play function with parsed arguments
    trained_play(
        env_adapter,
        number_of_episodes_to_play=args.number_of_episodes_to_play,
        path_to_trained_model_agent_0=args.path_to_trained_model_agent_0,
        path_to_parameters_agent_0=args.path_to_parameters_agent_0,
        path_to_trained_model_agent_1=args.path_to_trained_model_agent_1,
        path_to_parameters_agent_1=args.path_to_parameters_agent_1)

    # Close the environment
    env_adapter.close()
    print("Environment closed.")
