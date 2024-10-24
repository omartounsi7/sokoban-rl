from sokobanGame import Sokoban as game
import argparse

# from nnwrapper import NNetWrapper as nn
# from coach import Coach
import torch
import os
import sys


def display_board(board):
    """
    Displays the current state of the game board.
    """
    print(board)


def get_user_move():
    """
    Captures user input and maps it to a game move.
    w - up, s - down, a - left, d - right
    """
    move_mapping = {
        "w": (-1, 0),  # up
        "d": (0, 1),  # right
        "s": (1, 0),  # down
        "a": (0, -1),  # left
    }

    while True:
        move = input("Move (w/a/s/d): ").lower()
        if move in move_mapping:
            return move_mapping[move]
        else:
            print("Invalid input! Use 'w' (up), 'a' (left), 's' (down), 'd' (right).")


if __name__ == "__main__":
    fh = open(os.path.join("data", "puzzle1.txt"))
    fcontent = fh.read()
    fh.close()

    sys.setrecursionlimit(10000)
    g = game(fcontent)
    board = g.get_initial_board()

    while not g.has_puzzle_ended(board):
        display_board(board)
        move = get_user_move()
        board.execute_move(move)

    print("Congratulations! You've solved the puzzle!")
