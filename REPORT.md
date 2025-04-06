# Report

To address the challenges of solving the Tennis environment, I initially implemented the **Deterministic Policy Gradient (DDPG)** algorithm. DDPG had demonstrated promising results in my previous projects, particularly in continuous control tasks. However, I observed significant limitations in the agents' ability to balance **exploration** and **exploitation** effectively. During training, the agents often exhibited periods of strong performance, only to lose their **previously learned behaviors** and fail to recover. This instability highlighted the sensitivity of DDPG to the noise process used for exploration.

To mitigate this issue, I experimented with manually adjusting the noise process during training, which occasionally led to improved performance. Additionally, I implemented an **adaptive learning controller** that dynamically adjusted the noise process based on various metrics. Despite these efforts, the results remained unsatisfactory, as the agents struggled to achieve consistent and robust learning.

Ultimately, I adopted the **Soft Actor-Critic (SAC)** algorithm, which is explicitly designed to address the balance between exploration and exploitation. According to the literature ([SAC](https://arxiv.org/abs/1801.01290v2)), SAC incorporates **entropy regularization** to encourage exploration while maintaining stability in learning. This approach proved to be more effective in handling the challenges of the Tennis environment, leading to improved performance and more reliable learning outcomes.

## Learning algorithm: Soft Actor-Critic (SAC)

The Soft Actor-Critic (SAC) is an off-policy reinforcement learning algorithm designed to achieve stable and efficient learning in complex environments by balancing exploration and exploitation. It aims to learn policies that are robust and can generalize well, even in high-dimensional action spaces. It optimizes a stochastic policy in an entropy-regularized framework. It aims to maximize a trade-off between expected return (reward) and entropy, which quantifies the randomness of the policy. SAC is particularly effective in environments with continuous action spaces and has demonstrated robust performance and sample efficiency.

### Discussion of SAC's features

Compared to other off-policy algorithms, SAC addresses several shortcomings:

- *Exploration*:
  Traditional off-policy algorithms like Deep Deterministic Policy Gradient (DDPG) can suffer from insufficient exploration, leading to suboptimal policies. SAC's entropy regularization encourages more exploration, mitigating this issue.
- *Stability*:
  Algorithms like Proximal Policy Optimization (PPO) can be sensitive to hyperparameters and may require careful tuning. SAC's automatic tuning of the temperature parameter alpha helps stabilize the learning process.
- *Sample Efficiency*:
  By using a replay buffer and off-policy learning, SAC can make better use of collected data, improving sample efficiency compared to on-policy methods.
- *Robustness*:
  The stochastic nature of SAC's policy makes it more robust to noise and uncertainties in the environment, which can be a challenge for deterministic policies.

Depending on the scenario, there may also be downsides to be considered.

- *Hyperparameter Sensitivity*:
  Although SAC can automatically tune the temperature parameter alpha, other hyperparameters, such as learning rates and network architectures, still require careful tuning. Poor choices can lead to suboptimal performance.
- *Sample Efficiency in Sparse Reward Environments*:
  In environments with sparse rewards, SAC's reliance on entropy regularization might not be sufficient to drive exploration effectively. Other methods specifically designed for sparse reward settings might perform better.
- *Convergence Issues*:
  The stochastic nature of SAC's policy can sometimes lead to slower convergence compared to deterministic policies, especially in environments where precise actions are crucial.
- *Over-Exploration*:
  In some cases, the emphasis on exploration can lead to over-exploration, where the agent spends too much time exploring suboptimal actions, especially if the environment has many irrelevant or distracting states.

In general SAC is a powerful and versatile algorithm that addresses several challenges in reinforcement learning, particularly in continuous action spaces. Its emphasis on entropy regularization and automatic tuning of the temperature parameter makes it a robust choice for many applications, offering a good balance between exploration and exploitation.

### Key technical details implemented in this project

- Multi-Agent approach:
  Two separate agents are trained at the same time. Each time step one independent learning step for each agent is performed. Experience sampled for that learning step is always from the same step in time for both agents. So, the agents always learn on the same step in time.
- Actor-Critic Architecture:
  The SACAgent class uses an actor-critic architecture, where the actor network generates actions and the critic networks evaluate the quality of those actions. The actor network outputs the mean and log standard deviation of a Gaussian distribution, allowing for exploration through sampling.
- Entropy Regularization:
  The agent maximizes a trade-off between the reward and the entropy of the policy, encouraging exploration. The entropy coefficient alpha is automatically tuned to maintain a desired level of entropy.
- Double Q-Learning:
  Two critic networks are used to mitigate overestimation bias in the Q-value estimates. The minimum Q-value from the two critics is used for updates.
- Off-Policy Learning: Prioritized Replay Buffer with N-step returns and Importance Sampling:
  The agent uses a prioritized replay buffer to store and sample experiences for training.
  N-step learning is employed to improve the stability and efficiency of learning. The critic loss is weighted using importance sampling weights from the prioritized replay buffer to correct for bias introduced by non-uniform sampling.
- Soft Updates:
  Target networks are updated using a soft update rule, which helps stabilize training by slowly incorporating changes from the main networks.

### Note on Hyperparameter Exploration

Hyperparameter tuning is a critical aspect of deep reinforcement learning, as the choice of hyperparameters can significantly influence the training outcomes. The large number of hyperparameters involved often leads to highly variable results. Understanding the detailed progression of training can greatly facilitate the identification of effective hyperparameter configurations. Moreover, it allows for the early termination of experiments with suboptimal or misleading hyperparameters, saving valuable computational resources.

To address this, I implemented a plugin-based training monitor. This modular design enables the seamless addition of new plugins for specific logging or visualization tasks, providing a flexible and extensible framework for monitoring training progress. By offering detailed insights into the training process, this approach supports a more systematic and scientific exploration of hyperparameter spaces. See e.g. the [plots of loss-related learning indicators](./logs/plots_loss_based_plots_episode_2000.png) and the [plots of training progress](./logs/plots_meta_plots_episode_2000.png).

## Architecture of used neural networks

The Soft Actor-Critic (SAC) algorithm employs an actor-critic architecture, which consists of two main components: the actor network and the critic networks

### Actor networks

Purpose: The actor network is responsible for generating actions based on the current state. It defines the policy, which maps states to actions. It's structure looks like the following:

Architecture:

- *Input Layer*:
  The input to the actor network is the state representation, which is a vector of size *24* for the tennis environment.
- *Hidden Layers*:
  The network consists of 2 layers, each fully connected (dense). The size of these layers is *256*. Each hidden layer is followed by a Rectified Linear Unit (ReLU) activation function, which introduces non-linearity to the model.
- *Output Layers*:
  The actor network has two output layers:
  - *Mean Layer*:
    Outputs the mean (μ) of the action distribution. It has action_size units, corresponding to the dimensions of the action space.
  - *Log Standard Deviation Layer*:
    Outputs the log standard deviation (log ⁡σ) of the action distribution. This also has *2* units. The log standard deviation is clipped between -20 and 2 to ensure numerical stability.
  - *Reparameterization Trick*:
    To allow gradient-based optimization through the sampling process, the reparameterization trick is used. Actions are sampled from a Gaussian distribution parameterized by the mean and standard deviation. The sampled action is then squashed using the tanh function to ensure it lies within the range [−1,1].
  - *Action Selection*:
    During training, actions are sampled from the Gaussian distribution. During evaluation, the mean action can be used for deterministic behavior.

### Critic Networks

Purpose: The critic networks estimate the Q-value, which represents the expected return (cumulative future reward) of taking a given action in a given state. SAC uses two critic networks to mitigate overestimation bias.

Architecture:

- Input Layer:
  The input to each critic network is the concatenation of the state and action vectors, resulting in a vector of size 2 * (24 + 2) = 52.
- Hidden Layers:
  Similar to the actor network, the critic networks consist of *2* hidden layers, each fully connected and followed by ReLU activations. The size of these layers is set to *256*.
- Output Layer:
  The output layer is a *single neuron* that produces the Q-value estimate for the given state-action pairs of both agents.

## Hyperparameter choosen

Below is a list of all hyperparameters used, along with their explanations and choosen values:

### Summary Table

| **Hyperparameter**     | **Explanation**                                            | **Choosen Value** |
| ---------------------- | ---------------------------------------------------------- | ----------------- |
| `target_score`         | Cooperative target score to solve the environment          | `0.5`             |
| `max_episodes_count`   | Maximum number of training episodes                        | `2000`            |
| `actor_hidden_layers`  | Hidden layer sizes for the actor network                   | `[256, 256]`      |
| `critic_hidden_layers` | Hidden layer sizes for the critic networks                 | `[256, 256]`      |
| `buffer_size`          | Maximum number of experiences in the replay buffer         | `100000`          |
| `batch_size`           | Number of experiences sampled per training step            | `64`              |
| `n_steps`              | Number of steps for N-step returns                         | `1`               |
| `lr_actor`             | Learning rate for the actor network                        | `1e-4`            |
| `lr_critic`            | Learning rate for the critic networks                      | `1e-4`            |
| `lr_alpha`             | Learning rate for the entropy coefficient (alpha)          | `1e-4`            |
| `gamma`                | Discount factor for future rewards                         | `0.99`            |
| `tau`                  | Soft update factor for target networks                     | `1e-3`            |
| `alpha`                | Entropy coefficient for exploration-exploitation trade-off | `0.2`             |
| `alpha*`               | Priority exponent parameter (Replay buffer)                | `0.6`             |
| `beta`                 | Importance sampling exponent parameter (Replay buffer)     | `0.4`             |
| `beta_end`             | Final value of beta (Replay buffer)                        | `1.0`             |
| `beta_frames`          | Number of steps to anneal beta (Replay buffer)             | `100000`          |
| `checkpoint_frequency` | Frequency (in episodes) for saving checkpoints             | `500`             |
| `printout_frequency`   | Frequency (in episodes) for printing training progress     | `100`             |
| `device_type`          | Device used for training (`'cuda'` or `'cpu'`)             | `'cuda'`          |
| `random_seed`          | Random seed for reproducibility                            | `42`              |

## Results and plots of learning progress

[Training progress with plot of rewards](./logs/plots_meta_plots_episode_2000.png)

Given these hyperparameter the implemented agents solved the environment after 1806 episodes. They reached a rolling mean over the previous 100 episodes over the maximum of both agents of above 0.5. See the upper right figure in the above mentioned plot of the training progress. It stabily improved performance even further. The training was done until episode 2000 at which it reached an even higher perfromance.

Here are different losses which have been measured during the training: [Losses during training](./logs/plots_loss_based_plots_episode_2000.png)

### Trained weights

- [Agent 0](./logs/solved_agent0_at_episode_1804.pth)
- [Agent 1](./logs/solved_agent1_at_episode_1804.pth)


### Video of trained agents

- [Video of trained agents](./recordings/trained_agents_play.mp4)

## Ideas for future work

- Regarding the current implementation of SAC itself, I would analyze the influence of different:
  - learning rates
  - n-step lengths
  - batch sizes
  - tau for soft updates
  
  on the stability of learning, especially in sparse reward structures.
- Implementing Distributed Distributional Deterministic Policy Gradients [D4PG](https://arxiv.org/abs/1804.08617) which also showed good results on continuous control tasks.
- Impleneting a Central Training and Decentral Execution architecture. Using only one critic for the two actors in the environment.
- Langevin Soft Actor-Critic ([LSAC](https://openreview.net/forum?id=FvQsk3la17)):
  This variant of SAC focuses on enhancing critic learning through uncertainty estimation, which improves exploration efficiency. LSAC uses techniques like distributional Langevin Monte Carlo for Q updates and parallel tempering to explore multiple modes of the Q function's posterior. This approach has demonstrated performance improvements in continuous control tasks by addressing the sample efficiency issues common in actor-critic algorithms.
