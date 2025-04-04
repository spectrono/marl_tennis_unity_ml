from unityagents import UnityEnvironment
from unity_ml_tennis_environment import setup_tennis_environment

class UnityMLTennisEnvironmentAdapter:
    '''
    Adapter class to simplify interactions with the Unity ML Tennis environment.
    After setup, it reduces the amount of code needed to interact with the environment.
    '''

    def __init__(self):
        '''
        Initializes the environment adapter.

        Sets up the Unity ML Tennis environment and the brain.
        Configures the number of agents to 2, as required by the tennis environment.
        '''
        self.env, self.brain_name, self.agent_count, self.action_size, self.observation_size = setup_tennis_environment()
        self.brain = self.env.brains[self.brain_name]
        self.observations_current = None
        self.observations_next = None

        assert self.agent_count == 2, "This adapter is configured for 2 agents only."
        assert self.brain is not None, "Brain must be initialized."

    def reset(self, use_train_mode: bool):
        '''
        Resets the environment and sets the desired mode.

        Args:
            use_train_mode (bool): True for training mode, False for inference mode.

        Returns:
            Current state/observation of the environment.
        '''
        env_info = self.env.reset(train_mode=use_train_mode)[self.brain_name]
        self.observations_current = env_info.vector_observations
        return self.observations_current

    def is_training_mode_active(self) -> bool:
        '''
        Checks if the environment is in training mode.

        Returns:
            True if in training mode, False if in inference mode.
        '''
        return self.env.training_mode

    def get_agent_count(self) -> int:
        '''
        Returns the number of agents in the environment.

        Returns:
            Number of agents.
        '''
        return self.agent_count

    def get_action_size(self) -> int:
        '''
        Returns the size of the action space.

        Returns:
            Size of the action space.
        '''
        return self.action_size

    def get_observation_size(self) -> int:
        '''
        Returns the size of the observation space.

        Returns:
            Size of the observation space.
        '''
        return self.observation_size

    def get_observations(self):
        '''
        Returns the current state of the environment.

        Returns:
            Current states/observations for both agents.
        '''
        return self.observations_current

    def step(self, actions):
        '''
        Applies actions to the environment and returns the resulting observations, rewards, and done states.

        Args:
            actions (numpy.ndarray): Actions to be applied to the environment.
                                     Should be a 2D array of shape (2, action_size).

        Returns:
            tuple: Contains observations, rewards, and done states for each agent.
        '''
        env_info = self.env.step(actions)[self.brain_name]
        return (env_info.vector_observations, env_info.rewards, env_info.local_done)

    def close_env(self):
        '''
        Closes the environment and cleans up resources.
        '''
        self.env.close()
