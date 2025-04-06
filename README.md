# Project Details

This project aims to solve the **Tennis environment** from Unity ML using **Deep Reinforcement Learning**. It is a multi-agent learning problem.

## Tennis Environment

The environment simulates a cooperative or competitive task where two agents control rackets to hit a ball over a net. In this project, the goal is to keep the ball in play for as long as possible, maximizing the cooperative reward.

### Simulation Details

- Agents: Two agents control rackets in a 3D environment.
- Ball: A ball is launched into play, and the agents must hit it back and forth over the net.
- Net: A net divides the environment into two halves, one for each agent.

#### Action Space

The action space is continuous and consists of two dimensions for each agent:

- Horizontal Movement: Controls the racket's movement along the horizontal axis.
- Vertical Movement: Controls the racket's movement along the vertical axis.

Each action is a floating-point value in the range [-1, 1], where:

- -1 represents the minimum movement in the respective direction.
- 1 represents the maximum movement in the respective direction.

#### Observation Space

The observation space for each agent is a 24-dimensional vector that includes:

- Ball Position
- Ball Velocity
- Racket Position
- Racket Velocity
- Opponent Racket Position
- Opponent Racket Velocity

Each agent observes the environment independently, but the observations include information about both agents and the ball.

#### Reward Structure

- +0.1 Reward: An agent receives a reward of +0.1 each time it successfully hits the ball over the net.
- -0.01 Penalty: An agent receives a penalty of -0.01 if it fails to return the ball and the ball hits the ground on its side of the net.

The cooperative reward for each episode is calculated as the maximum score achieved by either agent during that episode. This encourages both agents to work together to maximize the total reward.

## How to Solve the Environment

The agents initially achieve an average score of **+0.5** (over 100 consecutive episodes, taking the maximum score between both agents). The goal is to reach this performance using deep reinforcement learning techniques.

## Approach Taken

To solve the environment, a variant of **Soft Actor-Critic (SAC)** is implemented (see [REPORT.md](./REPORT.md) for details). SAC is a state-of-the-art algorithm for deep reinforcement learning, particularly effective in handling continuous action spaces.

## Getting Started

The code has been tested on **Linux (Pop!_OS 22.04 LTS)**. All code is developed and tested with:

- **Python 3.9.21**
- **PyTorch 2.6.0**
- Standard libraries such as `numpy` and `matplotlib`.

For further details on dependencies, refer to the provided `requirements_*.txt` files:
- [Packages installed via Conda](./requirements_conda.txt)
- [Packages installed via Pip](./requirements_pip.txt)

### Download the Tennis Environment

Download the Unity ML Tennis environment from the following links:
- [Headless Tennis Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) (recommended for training, as it speeds up the process).
- [Normal Tennis Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) (required for generating video recordings).

Unzip the downloaded environment into the root directory of the repository. Note that the training code uses the **headless version** of the environment, while the normal version is used for video generation.

## General Instructions

All code is located in the **`code/`** folder. The main training script is **`multi_agent_train.py`**, which sets up and trains the agents. It includes a set of hyperparameters that have been successfully used to solve the Tennis environment. Feel free to modify these hyperparameters to experiment with improved performance. For details on the training results and hyperparameters, refer to the [REPORT.md](./REPORT.md) file.

### Environment Setup

Follow the instructions in the `requirements_*.txt` files to set up the required dependencies. Ensure that the Unity ML Tennis environment is downloaded and placed in the correct location as described above.

### Multi-Agent Random Play

Before training, you can observe random agents interacting with the Tennis environment. Use the following command to watch the agents:

```bash
python multi_agent_random_play.py
```

### Multi-Agent Training

To train the agents, use the *multi_agent_trainer.py* script. Below are the usage instructions:

```bash
python multi_agent_trainer.py [OPTIONS]
```

*Options:*

- *--log_folder_base* (default: "logs"): Base path to the folder containing the training data. The current date and time will be appended to this path.
- *--checkpoint_frequency* (default: 500): Frequency (in episodes) of logging intermediate results and saving checkpoints.
- *--printout_frequency* (default: 100): Frequency (in episodes) of printing intermediate results to the console.
- *--target_score* (default: 0.5): Cooperative target score that the agents aim to achieve.
- *--max_episodes_count* (default: 2000): Maximum number of training episodes.

#### Example Command

```bash
python multi_agent_trainer.py --log_folder_base logs --checkpoint_frequency 500 --printout_frequency 100 --target_score 0.5 --max_episodes_count 2000
```

This command trains the agents for up to 2000 episodes, logs intermediate results every 500 episodes, prints progress to the console every 100 episodes, and saves training data to a folder named logs_<current_date_time>.

### Training Monitor

During training, intermediate results such as checkpoints of the agents' trained weights and informative plots of their progress will be saved in the sub-folder `logs_<current_date_time>/checkpoints/`.

Whenever an agent's performance exceeds all previous performances, its trained weights and corresponding learning plots will be saved in the `logs_<current_date_time>/best/` directory. When the environment is first solved, a checkpoint will be stored in `logs_<current_date_time>/solved/`. Finally, after completing the specified number of episodes, a final checkpoint will be saved in `logs_<current_date_time>/final/`.

### Trained Multi-Agent Tennis Play

After training, you can watch the trained agents play the Tennis environment using the script `multi_agent_trained_play.py`. It allows you to simulate episodes in the Unity ML Tennis environment using trained agents. You can specify the number of episodes to play and provide paths to the trained models and their corresponding parameters for both agents.

#### Usage

```bash
python multi_agent_trained_play.py [OPTIONS]
```

*Options:*

- `--number_of_episodes_to_play` (default: `200`): The number of episodes to simulate.
- `--path_to_trained_model_agent_0` (default: `"logs/solved/solved_agent0_at_episode_1804.pth"`): Path to the trained model for agent 0.
- `--path_to_parameters_agent_0` (default: `"logs/agent_parameters_0.json"`): Path to the parameters for agent 0.
- `--path_to_trained_model_agent_1` (default: `"logs/solved/solved_agent1_at_episode_1804.pth"`): Path to the trained model for agent 1.
- `--path_to_parameters_agent_1` (default: `"logs/agent_parameters_1.json"`): Path to the parameters for agent 1.

#### Example Command

To simulate 200 episodes using the default trained models and parameters, run:

```bash
python multi_agent_trained_play.py
```

This command will:

1. Load the trained model for agent 0 from `logs/solved_agent0_at_episode_1804.pth` and its parameters from agent_parameters_0.json.
2. Load the trained model for agent 1 from `logs/solved_agent1_at_episode_1804.pth` and its parameters from agent_parameters_1.json.
3. Simulate 200 episodes in the Unity ML Tennis environment.
4. Print the maximum cooperative reward achieved in each episode and track the best score across all episodes.

#### Example Output

```plaintext
Agent 0 loaded from logs/solved_agent0_at_episode_1804.pth with parameters from logs/agent_parameters_0.json
Agent 1 loaded from logs/solved_agent1_at_episode_1804.pth with parameters from logs/agent_parameters_1.json
Episode 0 ---> Max total score (over both agents): 1.50 after 240 steps
Best score updated: 1.50
Episode 1 ---> Max total score (over both agents): 2.00 after 313 steps
Best score updated: 2.00
...
>>> Best score: 30.50 <<< after 200 episodes
Environment closed.
```

#### Custom Use-Case

If you want to simulate 100 episodes using custom trained models and parameters, you can specify the paths as follows:

```bash
python multi_agent_trained_play.py \
    --number_of_episodes_to_play 100 \
    --path_to_trained_model_agent_0 "custom_path/agent0_model.pth" \
    --path_to_parameters_agent_0 "custom_path/agent0_params.json" \
    --path_to_trained_model_agent_1 "custom_path/agent1_model.pth" \
    --path_to_parameters_agent_1 "custom_path/agent1_params.json"
```

This command will load the specified models and parameters, simulate 100 episodes, and print the results.
