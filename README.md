# Sokoban RL Solver

This project applies Reinforcement Learning algorithms to solve Sokoban puzzles. So far the following algorithms have been implemented:

- Monte Carlo Policy Optimization (Every Visit)

```bash
python .\montecarlo.py <puzzle_file> <number_of_episodes> <discount_factor> <exploration_rate>
```

- REINFORCE

```bash
python reinforce.py <puzzle_file> <number_of_episodes>
```

- Actor-Critic

```bash
python actorcritic.py <puzzle_file> <number_of_episodes>
```
