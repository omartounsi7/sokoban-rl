import sys
from src.SokobanEnv import SokobanEnv
from src.algorithms import mc_policy_evaluation

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python montecarlo.py <puzzle_file> <number_of_episodes> <discount_factor> <exploration_rate>")
        sys.exit(1)
    
    level_file = sys.argv[1]
    env = SokobanEnv(level_file)
    policy = mc_policy_evaluation(env, num_episodes=int(sys.argv[2]), gamma=float(sys.argv[3]), epsilon=float(sys.argv[4]))
    env.autoplay(policy)
    env.root.mainloop()