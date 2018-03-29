# Building Agents with Imagination

![](https://i.imgur.com/un9gSKe.gif)

Intelligent agents must have the capability to ‘imagine’ and reason about the future. Beyond that they must be able to construct a plan using this knowledge. [[1]](https://deepmind.com/blog/agents-imagine-and-plan/) This tutorial presents a new family of approaches for imagination-based planning:
-  Imagination-Augmented Agents for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1707.06203)
-  Learning and Querying Fast Generative Models for Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1802.03006)

## The tutorial consists of 4 parts:

#### 1. MiniPacman Environemnt
MiniPacman is played in a 15 × 19 grid-world. Characters, the ghosts and Pacman, move through a maze.<br>
[[minipacman.ipynb]](#)

#### 2. Actor Critic
Training standard model-free agent to play MiniPacman with advantage actor-critic (A2C)<br>
[[actor-critic.ipynb]](#)

#### 3. Environment Model
Environment model is a recurrent neural network which can be trained in an unsupervised
fashion from agent trajectories: given a past state and current action, the environment model predicts
the next state and reward.<br>
[[environment-model.ipynb]](#)

#### 4. Imagination Augmented Agent
The I2A learns to combine information from its model-free and imagination-augmented paths. The environment model is rolled out over multiple time steps into the future, by initializing the imagined trajectory with the present time real observation, and subsequently feeding simulated observations into the model. Then a rollout encoder processes the imagined trajectories as a whole and **learns to interpret it**, i.e. by extracting any information useful for the agent’s decision, or even ignoring it when necessary This allows the agent to benefit from model-based imagination without the pitfalls of conventional model-based planning.<br> 
[[imagination-augmented agent.ipynb]](#)

## More materials on model based + model free RL

  - The Predictron: End-To-End Learning and Planning [[arxiv]](https://arxiv.org/abs/1612.08810)
  - Model-Based Planning in Discrete Action Spaces [[arxiv]](https://arxiv.org/abs/1705.07177)
  - Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics [[arxiv]](https://arxiv.org/abs/1706.04317)
  -  World Models [[arxiv]](https://worldmodels.github.io/)
