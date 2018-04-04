# Building Agents with Imagination

<img width="160px" height="22px" href="https://github.com/pytorch/pytorch" src="https://pp.userapi.com/c847120/v847120960/82b4/xGBK9pXAkw8.jpg">

![](https://i.imgur.com/un9gSKe.gif)

Intelligent agents must have the capability to ‘imagine’ and reason about the future. Beyond that they must be able to construct a plan using this knowledge. [[1]](https://deepmind.com/blog/agents-imagine-and-plan/) This tutorial presents a new family of approaches for imagination-based planning:
-  Imagination-Augmented Agents for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1707.06203)
-  Learning and Querying Fast Generative Models for Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1802.03006)

## The tutorial consists of 4 parts:

#### 1. MiniPacman Environemnt
MiniPacman is played in a 15 × 19 grid-world. Characters, the ghosts and Pacman, move through a maze. The environment was written by [@sracaniere](https://github.com/sracaniere) from DeepMind.<br>
[[minipacman.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/1.minipacman.ipynb)

#### 2. Actor Critic
Training standard model-free agent to play MiniPacman with advantage actor-critic (A2C)<br>
[[actor-critic.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/2.actor-critic.ipynb)

#### 3. Environment Model
Environment model is a recurrent neural network which can be trained in an unsupervised
fashion from agent trajectories: given a past state and current action, the environment model predicts
the next state and reward.<br>
[[environment-model.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/3.environment-model.ipynb)

#### 4. Imagination Augmented Agent [in progress]
The I2A learns to combine information from its model-free and imagination-augmented paths. The environment model is rolled out over multiple time steps into the future, by initializing the imagined trajectory with the present time real observation, and subsequently feeding simulated observations into the model. Then a rollout encoder processes the imagined trajectories as a whole and **learns to interpret it**, i.e. by extracting any information useful for the agent’s decision, or even ignoring it when necessary This allows the agent to benefit from model-based imagination without the pitfalls of conventional model-based planning.<br> 
[[imagination-augmented agent.ipynb]](https://github.com/higgsfield/Building-Agents-with-Imagination/blob/master/4.imagination-augmented%20agent.ipynb)

## More materials on model based + model free RL

  - The Predictron: End-To-End Learning and Planning [[arxiv]](https://arxiv.org/abs/1612.08810)
  - Model-Based Planning in Discrete Action Spaces [[arxiv]](https://arxiv.org/abs/1705.07177)
  - Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics [[arxiv]](https://arxiv.org/abs/1706.04317)
  - Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1803.00101v1.pdf)
  - TEMPORAL DIFFERENCE MODELS: MODEL-FREE DEEP RL FOR MODEL-BASED CONTROL [[arxiv]](https://arxiv.org/pdf/1802.09081v1.pdf)
  - Universal Planning Networks [[arxiv]](https://arxiv.org/pdf/1804.00645.pdf)
  -  World Models [[arxiv]](https://worldmodels.github.io/)
  - Recall Traces: Backtracking Models for Efficient Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1804.00379.pdf)
