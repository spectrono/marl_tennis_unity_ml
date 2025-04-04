import numpy as np
from unity_ml_tennis_environment_adapter import UnityMLTennisEnvironmentAdapter

def random_play(env_adapter: UnityMLTennisEnvironmentAdapter, number_of_episodes_to_play: int):
    """
    Simulates random plays in the Unity ML Tennis environment.

    Args:
        env_adapter (UnityMLTennisEnvironmentAdapter): The environment adapter for the Unity ML Tennis environment.
        number_of_episodes_to_play (int): The number of episodes to play.

    This function will play a specified number of episodes in the Unity ML Tennis environment
    using random actions for each agent. It tracks and prints the best score achieved across
    all episodes.
    """

    best_score = -np.inf  # Initialize the best score to negative infinity
    agent_count = env_adapter.get_agent_count()  # Get the number of agents in the environment
    action_size = env_adapter.get_action_size()  # Get the size of the action space

    for episode_idx in range(number_of_episodes_to_play):
        env_adapter.reset(use_train_mode=False)  # Reset the environment before each episode
        observations = env_adapter.get_observations()  # Get the initial observations for each agent
        scores = np.zeros(agent_count)  # Initialize scores for each agent

        while True:
            actions = np.random.randn(agent_count, action_size)  # Select random actions for each agent
            actions = np.clip(actions, -1, 1)  # Clip actions to be within the range [-1, 1]
            env_response = env_adapter.step(actions)  # Send actions to the environment

            next_observations = env_response[0]  # Get the next state for each agent
            rewards = env_response[1]  # Get the rewards for each agent
            dones = env_response[2]  # Check if the episode is finished for any agent
            scores += rewards  # Update the scores with the received rewards
            observations = next_observations  # Update observations for the next step

            if np.any(dones):  # If any agent is done, exit the loop
                break

        max_of_scores = np.max(scores)  # Get the maximum score among all agents
        print(f'Episode {episode_idx} ---> Max total score (over both agents): {max_of_scores:0.2f}')

        if max_of_scores > best_score:  # Update the best score if the current score is higher
            best_score = max_of_scores
            print(f'Best score updated: {best_score:0.2f}')

    print(f'>>> Best score: {best_score:0.2f} <<<')


def main():
    """
    Main function to create the Unity ML Tennis environment and play random episodes.
    """
    # Create the tennis environment
    env_adapter = UnityMLTennisEnvironmentAdapter(use_headless=False)

    # Play 20 episodes with random actions
    random_play(env_adapter, number_of_episodes_to_play=20)

    # Close the environment
    env_adapter.close_env()

if __name__ == "__main__":
    main()
