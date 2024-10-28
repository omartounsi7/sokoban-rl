import sys
import tkinter as tk
from SokobanGame import Sokoban


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python playHuman.py puzzles/testHumanPuzzle.txt")
        sys.exit(1)

    root = tk.Tk()
    game = Sokoban(root, sys.argv[1])
    root.mainloop()
