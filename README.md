# Sokoban RL Solver

This project applies Reinforcement Learning algorithms to solve Sokoban puzzles. The following algorithms have been implemented:

- Monte Carlo Policy Optimization (Every Visit)
- Temporal Difference Learning (Q-Learning)
- REINFORCE
- Actor-Critic
- Deep Q-Network

To solve a puzzle using one of our algorithms, run:

```bash
python .\solve_puzzle.py <puzzle_file> <algorithm>
```

Algorithms: monte_carlo, td_learning, reinforce, actor_critic, dqn

The following algorithms from stable_baselines3 are also available:

- Proximal Policy Optimization
- Asynchronous Advantage Actor Critic
- Deep Q-Network

To solve a puzzle using a stable_baselines3 algorithm, run:

```bash
python .\sb3_solve_puzzle.py <puzzle_file> <algorithm>
```

Algorithms: PPO, A2C, DQN

Examples:

```bash
python .\solve_puzzle.py .\data\puzzles\level_1.txt monte_carlo
```

```bash
python .\sb3_solve_puzzle.py .\data\puzzles\level_2.txt PPO
```
