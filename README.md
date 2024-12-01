# Sokoban RL Solver

This project applies Reinforcement Learning algorithms to solve Sokoban puzzles. So far the following algorithms have been implemented:

- Monte Carlo Policy Optimization (Every Visit)
- REINFORCE
- Actor-Critic

Usage:

```bash
python solve_puzzle.py <puzzle_file> <algorithm>
```

Algorithms: monte_carlo, reinforce, actor_critic

Examples:

```bash
python .\solve_puzzle.py .\data\puzzles\level_1.txt monte_carlo
```

```bash
python .\solve_puzzle.py .\data\puzzles\level_2.txt reinforce
```

```bash
python .\solve_puzzle.py .\data\puzzles\level_3.txt actor_critic
```
