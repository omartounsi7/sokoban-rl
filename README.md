# Sokoban RL Solver

This project applies Reinforcement Learning algorithms to solve Sokoban puzzles. So far, the following algorithms have been implemented:

- Monte Carlo Policy Optimization (Every Visit)
- Temporal Difference Learning (Q-Learning)
- REINFORCE
- Actor-Critic
- Deep Q-Network (Vanilla)

Usage:

```bash
python .\solve_puzzle.py <puzzle_file> <algorithm>
```

Algorithms: monte_carlo, td_learning, reinforce, actor_critic, dqn

Examples:

```bash
python .\solve_puzzle.py .\data\puzzles\level_1.txt monte_carlo
```

```bash
python .\solve_puzzle.py .\data\puzzles\level_2.txt td_learning
```

```bash
python .\solve_puzzle.py .\data\puzzles\level_3.txt reinforce
```

```bash
python .\solve_puzzle.py .\data\puzzles\level_4.txt actor_critic
```

```bash
python .\solve_puzzle.py .\data\puzzles\level_5.txt dqn
```
