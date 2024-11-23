import sys
import tkinter as tk
from SokobanGame import Sokoban
from algorithms import mc_policy_evaluation

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python montecarlo.py puzzle number_of_episodes discount_factor exploration_rate")
        sys.exit(1)

    root = tk.Tk()
    game = Sokoban(master=root, level_file=sys.argv[1])

    game.policy = mc_policy_evaluation(initial_state=game.level, num_episodes=int(sys.argv[2]), gamma=float(sys.argv[3]), epsilon=float(sys.argv[4]))
    game.auto_play()
    game.print_metrics()

    root.mainloop()