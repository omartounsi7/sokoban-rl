# Sokoban RL Solver

This project applies Reinforcement Learning (RL) algorithms to solve Sokoban puzzles. So far the following algorithms have been implemented:

- Monte Carlo Policy Optimization (Every Visit). Usage:

```bash
python .\montecarlo.py <puzzle_file> <number_of_episodes> <discount_factor> <exploration_rate>
```

Example:

```bash
python .\montecarlo.py .\puzzles\easy.txt 2000 0.85 0.9
```
