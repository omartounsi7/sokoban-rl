from src.constants import *
import tkinter as tk
import time
import src.util

TILESIZE = 40
XOFFSET = 25
YOFFSET = 25

class Sokoban:
    def __init__(self, master, level_file):
        self.master = master
        self.master.title("Sokoban")
        self.level_file = level_file
        self.policy = None
        self.load_level()
        self.create_widgets()
        self.draw_game()
        self.bind_keys()
        self.game_over = False

    def load_level(self):
        with open(self.level_file, "r") as file:
            lines = file.readlines()
        self.level = [list(line.rstrip("\n")) for line in lines]
        self.initial_level = [row.copy() for row in self.level]
        self.height = len(self.level)
        self.width = max(len(line) for line in self.level)
        self.player_pos = src.util.find_player_in_state(self.level)
        self.total_boxes = sum(row.count("$") for row in self.level)

    def create_widgets(self):
        self.title_label = tk.Label(self.master, text="Sokoban", font=("Helvetica", 16))
        self.title_label.pack(pady=10)
        self.canvas = tk.Canvas(self.master, width=450, height=450)
        self.canvas.pack()
        self.reset_button = tk.Button(
            self.master, text="Reset", command=self.reset_level
        )
        self.reset_button.pack(pady=5)
        self.message_label = tk.Label(self.master, text="", font=("Helvetica", 14))
        self.message_label.pack(pady=5)

    def reset_level(self):
        self.level = [row.copy() for row in self.initial_level]
        self.player_pos = src.util.find_player_in_state(self.level)
        self.game_over = False
        self.draw_game()

        if self.policy:
            self.auto_play()

    def draw_game(self):
        self.canvas.delete("all")
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                x1 = XOFFSET + x * TILESIZE
                y1 = YOFFSET + y * TILESIZE
                x2, y2 = x1 + TILESIZE, y1 + TILESIZE
                if char == "#":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")
                elif char == ".":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                elif char == "$":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_rectangle(
                        x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="brown"
                    )
                elif char == "x": 
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_rectangle(
                        x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="green"
                    )
                elif char == "@":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="blue")
                elif char == "+": 
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="blue")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
        self.canvas.update()

    def bind_keys(self):
        if self.policy:
            self.master.bind("<Key>", self.key_pressed_policy)
        else:
            self.master.bind("<Key>", self.key_pressed_human)

    def key_pressed_policy(self, event):
        pass 

    def key_pressed_human(self, event):
        if self.game_over:
            return

        if event.keysym == "r":
            self.reset_level()
            return

        action = src.util.get_action(event.keysym)
        if action:
            self.player_pos, reward, moved_box = src.util.execute_action(self.level, action)
            self.draw_game()
            
            if reward == SUPERBONUS:
                print("Game won!")
                self.game_over = True
            elif reward == SUPERMALUS:
                print("Game lost!")
                self.game_over = True


    def auto_play(self):
        print("Playing level according to the learned policy...")
        self.number_of_actions = 0
        while not self.game_over:
            state = src.util.serialize_state(self.level)
            action = self.policy.get(state, None)
            if action is None:
                print("No action found in policy for the current state.")
                break
            action_vector = src.util.get_action(action)
            if action_vector is None:
                print(f"Invalid action '{action}' in policy for state.")
                break
            time.sleep(1)  
            self.player_pos, reward, moved_box = src.util.execute_action(self.level, action_vector)
            self.number_of_actions += 1
            self.draw_game()
            if reward == SUPERBONUS:
                print("Game won!")
                self.game_over = True
            elif reward == SUPERMALUS:
                print("Game lost!")
                self.game_over = True
        print("Finished playing level.")

    def print_metrics(self):
        print("Percentage of boxes placed: " + str(src.util.count_placed_boxes(self.level) / self.total_boxes * 100))
        print("Number of actions: " + str(self.number_of_actions))