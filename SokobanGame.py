import tkinter as tk
import time
import itertools


class Sokoban:
    def __init__(self, master, level_file):
        self.master = master
        self.master.title("Sokoban")
        self.level_file = level_file
        self.policy = None
        self.load_level()
        self.create_widgets()
        self.bind_keys()
        self.game_over = False
        self.generate_state_space()

        if self.policy:
            self.auto_play()

    def load_level(self):
        with open(self.level_file, "r") as file:
            lines = file.readlines()
        self.level = [list(line.rstrip("\n")) for line in lines]
        self.initial_level = [row.copy() for row in self.level]
        self.height = len(self.level)
        self.width = max(len(line) for line in self.level)
        self.find_player()
        self.total_boxes = sum(row.count("$") for row in self.level)

    def reset_level(self):
        self.level = [row.copy() for row in self.initial_level]
        self.find_player()
        self.game_over = False
        self.message_label.config(text="")
        self.draw_game()

        if self.policy:
            self.auto_play()

    def generate_state_space(self):
        state_space = set()
        wall_positions = set()
        goal_positions = set()
        box_positions = set()

        for y, row in enumerate(self.initial_level):
            for x, cell in enumerate(row):
                if cell == "#":
                    wall_positions.add((y, x))
                elif cell == ".":
                    goal_positions.add((y, x))
                elif cell == "$":
                    box_positions.add((y, x))
                elif cell == "@":
                    player_position = (y, x)

        non_wall_positions = [
            (y, x)
            for y, row in enumerate(self.initial_level)
            for x, cell in enumerate(row)
            if cell != "#"
        ]
        # Generate all possible combinations of box positions
        box_combinations = itertools.combinations(
            non_wall_positions, len(box_positions)
        )

        for box_comb in box_combinations:
            for player_pos in non_wall_positions:
                if player_pos not in box_comb:
                    level_state = []
                    for y, row in enumerate(self.initial_level):
                        new_row = []
                        for x, cell in enumerate(row):
                            pos = (y, x)
                            if pos in wall_positions:
                                new_row.append("#")
                            elif pos in goal_positions and pos in box_comb:
                                new_row.append("x")
                            elif pos in goal_positions and pos == player_pos:
                                new_row.append("+")
                            elif pos in goal_positions:
                                new_row.append(".")
                            elif pos == player_pos:
                                new_row.append("@")
                            elif pos in box_comb:
                                new_row.append("$")
                            else:
                                new_row.append(" ")
                        level_state.append(tuple(new_row))

                    state_space.add(tuple(level_state))
        self.state_space = state_space

    def find_player(self):
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                if char == "@" or char == "+":
                    self.player_pos = (x, y)
                    return

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
        self.draw_game()

    def bind_keys(self):
        if self.policy:
            self.master.bind("<Key>", self.key_pressed_policy)
        else:
            self.master.bind("<Key>", self.key_pressed_human)

    def key_pressed_policy(self, event):
        pass  # Ignore key presses when using a policy

    def key_pressed_human(self, event):
        if self.game_over:
            return

        if event.keysym == "r":
            self.reset_level()
            return

        action = self.get_action(event.keysym)
        if action:
            box_placed, box_stuck = self.execute_action(action)
            print("Was a box placed: " + str(box_placed))
            print("Was a box stuck: " + str(box_stuck))
            r = self.compute_reward(box_placed, box_stuck)
            print("Reward is " + str(r))
            self.draw_game()
            self.check_win()

    def get_action(self, key):
        key_mapping = {
            "w": (0, -1),
            "a": (-1, 0),
            "s": (0, 1),
            "d": (1, 0),
            "Up": (0, -1),
            "Left": (-1, 0),
            "Down": (0, 1),
            "Right": (1, 0),
        }
        return key_mapping.get(key, None)

    def execute_action(self, action):
        box_placed = False
        box_stuck = False

        dx, dy = action
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        if self.is_wall(nx, ny):
            return box_placed, box_stuck

        if self.is_box(nx, ny):
            if not self.move_box(nx, ny, dx, dy):
                return box_placed, box_stuck

            bx, by = nx + dx, ny + dy
            if self.is_box_stuck(bx, by):
                box_stuck = True
            if self.is_box_placed(bx, by):
                box_placed = True

        # Update player's current position
        if self.level[y][x] == "@":
            self.level[y][x] = " "
        elif self.level[y][x] == "+":  # Player was on a goal
            self.level[y][x] = "."

        # Update new player's position
        if self.level[ny][nx] == ".":
            self.level[ny][nx] = "+"
        else:
            self.level[ny][nx] = "@"

        self.player_pos = (nx, ny)

        return box_placed, box_stuck

    def is_wall(self, x, y):
        return self.level[y][x] == "#"

    def is_box(self, x, y):
        return self.level[y][x] in ["$", "x"]

    def is_goal(self, x, y):
        return self.level[y][x] == "."

    def move_box(self, x, y, dx, dy):
        nx, ny = x + dx, y + dy
        if self.level[ny][nx] in [" ", "."]:
            # Update box's current position
            if self.level[y][x] == "$":
                self.level[y][x] = " "
            elif self.level[y][x] == "x":
                self.level[y][x] = "."

            # Update box's new position
            if self.level[ny][nx] == ".":
                self.level[ny][nx] = "x"
            else:
                self.level[ny][nx] = "$"
            return True
        return False

    def draw_game(self):
        self.canvas.delete("all")
        size = 40  # Size of each tile
        offset_x = 25  # Offset to center the game
        offset_y = 25
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                x1 = offset_x + x * size
                y1 = offset_y + y * size
                x2, y2 = x1 + size, y1 + size
                if char == "#":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")
                elif char == ".":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                elif char == "$":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_rectangle(
                        x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="brown"
                    )
                elif char == "x":  # Box on goal
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_rectangle(
                        x1 + 10, y1 + 10, x2 - 10, y2 - 10, fill="green"
                    )
                elif char == "@":
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="blue")
                elif char == "+":  # Player on goal
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="blue")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
        self.canvas.update()

    def check_win(self):
        for row in self.level:
            if "$" in row:
                return False
        self.game_over = True
        self.message_label.config(text="You won!")
        return True

    def is_box_placed(self, x, y):
        return self.level[y][x] in ["x"]

    def is_box_stuck(self, x, y):
        # If the box is on a goal, it's not stuck
        if self.level[y][x] in ["x", "."]:
            return False

        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Check for corner configurations
        for i in range(4):
            dx1, dy1 = dirs[i]
            dx2, dy2 = dirs[(i + 1) % 4]
            if self.is_obstacle(x + dx1, y + dy1) and self.is_obstacle(
                x + dx2, y + dy2
            ):
                return True

        return False

    def is_obstacle(self, x, y):
        if x < 0 or x >= len(self.level[0]) or y < 0 or y >= len(self.level):
            return True  # Treat out-of-bounds as walls
        return self.level[y][x] in ["#", "$", "x"]

    def count_boxes_on_goals(self):
        count = 0
        for row in self.level:
            count += row.count("x")  # 'x' represents a box on a goal
        return count

    def compute_reward(self, box_placed, box_stuck):
        if box_stuck:
            return -99
        if box_placed:
            return 1
        return 0

    def serialize_state(self):
        return tuple(tuple(row) for row in self.level)

    def auto_play(self):
        # Automatically plays the game according to the given policy.
        while not self.game_over:
            state = self.serialize_state()
            if state in self.policy:
                action = self.policy[state]
                action_vector = self.get_action_from_direction(action)
                if action_vector is None:
                    print(f"Invalid action '{action}' in policy for state.")
                    break
                time.sleep(1)  # Wait between actions
                box_placed, box_stuck = self.execute_action(action_vector)
                print("Was a box placed: " + str(box_placed))
                print("Was a box stuck: " + str(box_stuck))
                r = self.compute_reward(box_placed, box_stuck)
                print("Reward is " + str(r))
                self.draw_game()
                self.check_win()
            else:
                print("No action found in policy for the current state.")
                break

    def get_action_from_direction(self, direction):
        action_mapping = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        return action_mapping.get(direction.lower(), None)
