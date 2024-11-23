import sys
import tkinter as tk
from SokobanGame import Sokoban

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python humanPlayer.py <puzzle_file>")
        sys.exit(1)

    root = tk.Tk()
    game = Sokoban(master=root, level_file=sys.argv[1])
    root.mainloop()