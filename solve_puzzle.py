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
from src.util import generate_state_space

# from src.util import compute_v_and_q_from_policy
# from src.constants import GAMMA, NUMERIC_ACTION_SPACE

def main():
    if len(sys.argv) != 3:
        print("Usage: python solve_puzzle.py <puzzle_file> <algorithm>")
        print("Available algorithms: monte_carlo, td_learning, reinforce, actor_critic, dqn")
        sys.exit(1)

    puzzle_path = sys.argv[1]
    algorithm = sys.argv[2]

    try:
        env = SokobanEnv(puzzle_path)
    except FileNotFoundError:
        print(f"Error: Puzzle file '{puzzle_path}' not found.")
        sys.exit(1)

    # if puzzle_path.endswith(".txt"):
    #     policy_path = puzzle_path.replace(".txt", "_opt_policy.txt")
    # else:
    #     print(f"Error: Invalid puzzle file format '{puzzle_path}'. Expected a .txt file.")
    #     sys.exit(1)

    # try:
    #     with open(policy_path, "r") as policy_file:
    #         opt_policy = eval(policy_file.read().strip())
    # except (FileNotFoundError, Exception) as e:
    #     print(f"Error: Could not process the optimal policy file '{policy_path}': {e}")
    #     sys.exit(1)

    # numeric_opt_policy = [NUMERIC_ACTION_SPACE[action] for action in opt_policy]
    # V_opt, Q_opt = compute_v_and_q_from_policy(env, numeric_opt_policy, GAMMA)

    # with open(puzzle_path, "r") as file:
    #     initial_level = [list(line.rstrip("\n")) for line in file.readlines()]
    # state_space = generate_state_space(initial_level)
    # print(f"Estimated size of the state space: {len(state_space)}")

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
        print("Available algorithms: monte_carlo, td_learning, reinforce, actor_critic, dqn")
        sys.exit(1)

    after = process.memory_info().rss / 1024 / 1024
    time_to_train = time.time() - start_time
    print(f"Execution time: {time_to_train:.2f}s")
    print(f"Memory use: {after - before:.2f} MB")

    env.autoplay(policy)
    env.root.mainloop()

if __name__ == "__main__":
    main()
