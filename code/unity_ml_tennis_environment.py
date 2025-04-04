from unityagents import UnityEnvironment


def setup_tennis_environment():
    '''
    Simplifies the setup of the UnityEnvironment and return only what is really needed for training.

    Environment will NOT be in training mode after initialization!

    Return:
        env: The unity environment
        brain_name(stirng): Name of the brain
        nb_agents(int): How many agents are defined
        action_size(int): Dimension of action space
        state_size(int): Dimension of state space

    '''
    print('\n>>>>>>>>>>>>>>> Setting up environment <<<<<<<<<<<<<<<\n\n')
    env = UnityEnvironment(file_name='Tennis_Linux_NoVis/Tennis.x86')
    print(env.brain_names)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    agents_count = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space 
    states = env_info.vector_observations
    assert(len(states) == agents_count, "Mismatch between number of agents and states")
    state_size = states[0].shape[0]

    return env, brain_name, agents_count, action_size, state_size
