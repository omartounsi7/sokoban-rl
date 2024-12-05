import sys
import os
import time
import psutil
from src.SokobanEnv import SokobanEnv
from src.montecarlo import mc_policy_evaluation
from src.reinforce import reinforce_policy_gradient
from src.actorcritic import actor_critic_policy_gradient
from src.dqn import deep_q_learning
from src.td import td_learning

def main():
    if len(sys.argv) != 3:
        print("Usage: python solve_puzzle.py <puzzle_file> <algorithm>")
        print("Available algorithms: monte_carlo, reinforce, actor_critic, dqn")
        sys.exit(1)

    puzzle_path = sys.argv[1]
    algorithm = sys.argv[2]

    try:
        env = SokobanEnv(puzzle_path)
    except FileNotFoundError:
        print(f"Error: Puzzle file '{puzzle_path}' not found.")
        sys.exit(1)

    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    if algorithm == "monte_carlo":
        policy = mc_policy_evaluation(env)
    elif algorithm == "reinforce":
        policy, rewards = reinforce_policy_gradient(env)
    elif algorithm == "actor_critic":
        policy, rewards = actor_critic_policy_gradient(env)
    elif algorithm == "dqn":
        policy = deep_q_learning(env)
    elif algorithm == "td_learning":
        policy = td_learning(env)
    else:
        print(f"Error: Unknown algorithm '{algorithm}'.")
        print("Available algorithms: monte_carlo, reinforce, actor_critic, dqn, td_learning")
        sys.exit(1)

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    print(f"Execution time: {time_to_train:.2f}s")
    print(f"Memory use: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()

if __name__ == "__main__":
    main()
