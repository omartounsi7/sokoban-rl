import sys
import tkinter as tk
from SokobanGame import Sokoban

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python playPolicy.py puzzles/testPolicyPuzzle.txt")
        sys.exit(1)

    testPolicy = {
        (
            ('#','#','#','#','#','#'),
            ('#','@',' ','$','.','#'),
            ('#','#','#','#','#','#')
        ): 'right',
        (
            ('#','#','#','#','#','#'),
            ('#',' ','@','$','.','#'),
            ('#','#','#','#','#','#')
        ): 'right'
    }

    root = tk.Tk()
    game = Sokoban(root, sys.argv[1], policy=testPolicy)
    root.mainloop()
