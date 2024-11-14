import tkinter as tk
import time
import itertools
import random
import copy

TILESIZE = 40
XOFFSET = 25
YOFFSET = 25
MALUS = -1
BONUS = 1
MAXSTEPS = 100000

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
        self.action_space = ['up', 'left', 'down', 'right']

        # print("Generating state space...")
        # self.generate_state_space()

        print("Training policy...")
        self.mc_policy_evaluation(num_episodes=1000, discount_factor=0.9, epsilon=0.2, every_visit=True)

        print("Playing level...")
        self.auto_play()

    def load_level(self):
        with open(self.level_file, "r") as file:
            lines = file.readlines()
        self.level = [list(line.rstrip("\n")) for line in lines]
        self.initial_level = [row.copy() for row in self.level]
        self.height = len(self.level)
        self.width = max(len(line) for line in self.level)
        self.player_pos = self.find_player_in_state(self.level)
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
        self.player_pos = self.find_player_in_state(self.level)
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
        pass  # Ignore key presses when using a policy

    def key_pressed_human(self, event):
        if self.game_over:
            return

        if event.keysym == "r":
            self.reset_level()
            return

        action = self.get_action(event.keysym)
        if action:
            self.player_pos, reward = self.execute_action(self.level, action)
            # print("Reward is " + str(reward))
            self.draw_game()
            
            if self.check_win(self.level):
                print("Game won!")
                self.game_over = True
            elif reward == MALUS:
                print("Game lost!")
                self.game_over = True

    def find_player_in_state(self, state):
        """
        Finds the player's position in a given state.
        Returns (x, y) or None if not found.
        """
        for y, row in enumerate(state):
            for x, cell in enumerate(row):
                if cell in ('@', '+'):
                    return (x, y)
        return None

    def print_state(self, state):
        for row in state:
            print(row)

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

    def get_action(self, key):
        key_mapping = {
            "w": (0, -1),
            "a": (-1, 0),
            "s": (0, 1),
            "d": (1, 0),
            "up": (0, -1),
            "left": (-1, 0),
            "down": (0, 1),
            "right": (1, 0),
        }
        return key_mapping.get(key.lower(), None)

    def execute_action(self, state, action):
        dx, dy = action
        x, y = self.find_player_in_state(state)
        nx, ny = x + dx, y + dy
        
        reward = 0
        new_position = (x, y)

        if self.is_wall(state, nx, ny):
            return new_position, reward

        if self.is_box(state, nx, ny):
            if not self.move_box(state, nx, ny, dx, dy):
                return new_position, reward

            bx, by = nx + dx, ny + dy
            if self.is_box_stuck(state, bx, by):
                reward = MALUS
            elif self.is_box_placed(state, bx, by):
                reward = BONUS

        self.move_agent(state, x, y, nx, ny)
        new_position = (nx, ny)

        return new_position, reward

    def move_agent(self, state, x, y, nx, ny):
        # Update player's current position
        if state[y][x] == "@":
            state[y][x] = " "
        elif state[y][x] == "+":
            state[y][x] = "."

        # Update new player's position
        if state[ny][nx] == ".":
            state[ny][nx] = "+"
        else:
            state[ny][nx] = "@"

    def move_box(self, state, x, y, dx, dy):
        nx, ny = x + dx, y + dy
        if state[ny][nx] in [" ", "."]:
            # Update box's current position
            if state[y][x] == "$":
                state[y][x] = " "
            elif state[y][x] == "x":
                state[y][x] = "."

            # Update box's new position
            if state[ny][nx] == ".":
                state[ny][nx] = "x"
            else:
                state[ny][nx] = "$"
            return True
        return False

    def is_wall(self, state, x, y):
        return state[y][x] == "#"

    def is_box(self, state, x, y):
        return state[y][x] in ["$", "x"]

    def is_goal(self, state, x, y):
        return state[y][x] == "."

    def is_box_placed(self, state, x, y):
        return state[y][x] in ["x"]

    def is_position_free(self, state, x, y):
        return state[y][x] in [' ', '.']

    def is_box_stuck(self, state, x, y):
        # If the box is on a goal, it's not stuck
        if self.is_box_placed(state, x, y):
            return False

        dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # Check for corner configurations
        for i in range(4):
            dx1, dy1 = dirs[i]
            dx2, dy2 = dirs[(i + 1) % 4]
            if self.is_obstacle(state, x + dx1, y + dy1) and self.is_obstacle(state, x + dx2, y + dy2):
                return True

        return False

    def is_obstacle(self, state, x, y):
        if x < 0 or x >= len(state[0]) or y < 0 or y >= len(state):
            return True  # Treat out-of-bounds as walls
        return state[y][x] in ["#", "$", "x"]

    def check_win(self, state):
        for row in state:
            if '$' in row:
                return False
        return True
    
    def serialize_state(self, state):
        return tuple(tuple(row) for row in state)

    def mc_policy_evaluation(self, num_episodes=1000, discount_factor=0.9, epsilon=0.1, every_visit=False):        
        Q = {}
        returns_sum = {}
        returns_count = {}
        policy = {}
        
        for episode in range(num_episodes):
            print("Episode", episode + 1)
            current_state = copy.deepcopy(self.level)
            steps = 0
            episode_states_actions_rewards = []
            terminalState = False

            while not terminalState and steps < MAXSTEPS:
                steps += 1
                serialized_current_state = self.serialize_state(current_state)

                if random.random() < epsilon:
                    action = random.choice(self.action_space)
                else:
                    action = policy.get(serialized_current_state, random.choice(self.action_space))
                
                action_vector = self.get_action(action)
                new_pos, reward = self.execute_action(current_state, action_vector)
                episode_states_actions_rewards.append((serialized_current_state, action, reward))
                
                if self.check_win(current_state):
                    print("PUZZLE COMPLETED!")
                    terminalState = True
                elif reward == MALUS:
                    print("BOX STUCK!")
                    terminalState = True

            print("Number of steps: " + str(steps))

            # Calculate returns for the episode
            G = 0
            episode_length = len(episode_states_actions_rewards)
            visited_state_actions = set()

            for t in reversed(range(episode_length)):
                state, action, reward = episode_states_actions_rewards[t]
                G = discount_factor * G + reward

                # Check if this is the first occurrence of (state, action)
                if every_visit or (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))

                    if state not in Q:
                        Q[state] = {a: 0.0 for a in self.action_space}
                        returns_sum[state] = {a: 0.0 for a in self.action_space}
                        returns_count[state] = {a: 0 for a in self.action_space}
                    
                    returns_sum[state][action] += G
                    returns_count[state][action] += 1
                    Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                    best_action = max(Q[state], key=Q[state].get)
                    policy[state] = best_action

        self.policy = policy
        print("First Visit MC Policy Evaluation Completed.")

    def auto_play(self):
        """
        Automatically plays the game according to the learned policy.
        """
        while not self.game_over:
            state = self.serialize_state(self.level)
            action = self.policy.get(state, None)
            if action is None:
                print("No action found in policy for the current state.")
                break
            action_vector = self.get_action(action)
            if action_vector is None:
                print(f"Invalid action '{action}' in policy for state.")
                break
            time.sleep(1)  # Wait between actions for visualization
            self.player_pos, reward = self.execute_action(self.level, action_vector)
            # print("Reward is " + str(reward))
            self.draw_game()
            
            if self.check_win(self.level):
                print("Game won!")
                self.game_over = True
            elif reward == MALUS:
                print("Game lost!")
                self.game_over = True