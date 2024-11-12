import sys
import tkinter as tk
from SokobanGame import Sokoban

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python playPolicy.py puzzles/testPolicyPuzzle.txt")
        sys.exit(1)

    root = tk.Tk()
    game = Sokoban(root, sys.argv[1])
    
    value_function = game.value_iteration()
    print("Optimal Value Function:")
    for state, value in value_function.items():
        print(f"State: {state}, Value: {value}")

    print("\nOptimal Policy:")
    for state, action in game.policy.items():
        print(f"State: {state}, Action: {action}")

    game.plot_value_function(value_function)
    root.mainloop()