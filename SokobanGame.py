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

    def get_action_from_direction(self, direction):
        action_mapping = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }
        return action_mapping.get(direction.lower(), None)

    def is_terminal(self, state):
        """
        Checks if the state is terminal (all boxes on goals).
        """
        for row in state:
            if '$' in row:
                return False
        return True

    def get_possible_actions(self, state):
        """
        Returns the list of possible actions from the given state.
        Actions are 'up', 'down', 'left', 'right'.
        """
        actions = []
        player_pos = self.find_player_in_state(state)
        if not player_pos:
            return actions

        x, y = player_pos
        action_mapping = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        for action, (dx, dy) in action_mapping.items():
            nx, ny = x + dx, y + dy
            if self.is_position_free(state, nx, ny):
                actions.append(action)
            elif self.is_box(state, nx, ny):
                bx, by = nx + dx, ny + dy
                if self.is_position_free(state, bx, by):
                    actions.append(action)
        return actions

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

    def is_position_free(self, state, x, y):
        """
        Checks if a position is free (empty or goal).
        """
        if not self.is_within_bounds(x, y, state):
            return False
        return state[y][x] in (' ', '.')

    def is_box(self, state, x, y):
        """
        Checks if a position has a box.
        """
        if not self.is_within_bounds(x, y, state):
            return False
        return state[y][x] in ('$', 'x')

    def is_within_bounds(self, x, y, state):
        """
        Checks if (x, y) is within the grid bounds.
        """
        return 0 <= y < len(state) and 0 <= x < len(state[0])

    def apply_action_to_state(self, state, action):
        """
        Applies an action to a given state and returns the new state and reward.
        """
        new_state = [list(row) for row in state]  # Deep copy
        player_pos = self.find_player_in_state(state)
        if not player_pos:
            return None, -99  # Invalid state

        x, y = player_pos
        action_mapping = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        if action not in action_mapping:
            return None, -99  # Invalid action

        dx, dy = action_mapping[action]
        nx, ny = x + dx, y + dy

        if not self.is_within_bounds(nx, ny, state):
            return None, -99  # Move into wall

        target_cell = state[ny][nx]

        if target_cell in (' ', '.'):
            # Move player
            if target_cell == '.':
                new_state[y][x] = '.' if state[y][x] == '+' else ' '
                new_state[ny][nx] = '+' if state[y][x] == '+' else '@'
            else:
                new_state[y][x] = '.' if state[y][x] == '+' else ' '
                new_state[ny][nx] = '@'
            return tuple(tuple(row) for row in new_state), 0  # No reward
        elif target_cell in ('$', 'x'):
            # Attempt to push the box
            bx, by = nx + dx, ny + dy
            if not self.is_within_bounds(bx, by, state):
                return None, -99  # Box push out of bounds
            box_target_cell = state[by][bx]
            if box_target_cell in (' ', '.'):
                # Move box
                if box_target_cell == '.':
                    new_state[by][bx] = 'x'
                else:
                    new_state[by][bx] = '$'

                # Update box's old position
                new_state[ny][nx] = '+' if state[ny][nx] == 'x' else '@'

                # Update player's old position
                new_state[y][x] = '.' if state[y][x] == '+' else ' '

                # Compute reward
                reward = 1 if new_state[by][bx] == 'x' else 0
                return tuple(tuple(row) for row in new_state), reward
            else:
                return None, -99  # Box push blocked
        else:
            return None, -99  # Invalid target cell

    def first_visit_mc_policy_evaluation(self, num_episodes=1000, discount_factor=0.9):
        """
        Performs First Visit Monte Carlo Policy Evaluation to estimate Q(s,a).
        Updates the policy to be greedy with respect to the estimated Q(s,a).
        
        Args:
            num_episodes (int): Number of episodes to sample.
            discount_factor (float): Discount factor for future rewards.
        """
        # Initialize Q(s,a) and returns_sum, returns_count
        Q = {}
        returns_sum = {}
        returns_count = {}
        
        # Initialize policy: random for all states
        for state in self.state_space:
            Q[state] = {}
            returns_sum[state] = {}
            returns_count[state] = {}
            possible_actions = self.get_possible_actions(state)
            for action in possible_actions:
                Q[state][action] = 0.0
                returns_sum[state][action] = 0.0
                returns_count[state][action] = 0

        # Initialize policy: random policy
        policy = {}
        for state in self.state_space:
            possible_actions = self.get_possible_actions(state)
            if possible_actions:
                policy[state] = random.choice(possible_actions)
            else:
                policy[state] = None  # No possible actions

        for episode in range(num_episodes):
            # Generate an episode following the current policy
            episode_states_actions_rewards = []
            current_state = self.serialize_state()
            while True:
                action = policy.get(current_state, None)
                if action is None:
                    break  # No possible actions, end episode
                next_state, reward = self.apply_action_to_state(current_state, action)
                if next_state is None:
                    # Invalid move, end episode with penalty
                    episode_states_actions_rewards.append((current_state, action, -99))
                    break
                episode_states_actions_rewards.append((current_state, action, reward))
                if self.is_terminal(next_state):
                    break
                current_state = next_state

            # Calculate returns for the episode
            G = 0
            episode_length = len(episode_states_actions_rewards)
            for t in reversed(range(episode_length)):
                state, action, reward = episode_states_actions_rewards[t]
                G = discount_factor * G + reward
                # Check if this is the first occurrence of (state, action)
                first_occurrence = True
                for i in range(t):
                    if (episode_states_actions_rewards[i][0] == state and 
                        episode_states_actions_rewards[i][1] == action):
                        first_occurrence = False
                        break
                if first_occurrence:
                    returns_sum[state][action] += G
                    returns_count[state][action] += 1
                    Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                    # Improve the policy to be greedy
                    best_action = max(Q[state], key=Q[state].get)
                    policy[state] = best_action

        # After all episodes, set the policy
        self.policy = policy
        print("First Visit MC Policy Evaluation Completed.")

    def every_visit_mc_policy_evaluation(self, num_episodes=1000, discount_factor=0.9):
        """
        Performs Every Visit Monte Carlo Policy Evaluation to estimate Q(s,a).
        Updates the policy to be greedy with respect to the estimated Q(s,a).
        
        Args:
            num_episodes (int): Number of episodes to sample.
            discount_factor (float): Discount factor for future rewards.
        """
        # Initialize Q(s,a) and returns_sum, returns_count
        Q = {}
        returns_sum = {}
        returns_count = {}
        
        # Initialize policy: random for all states
        for state in self.state_space:
            Q[state] = {}
            returns_sum[state] = {}
            returns_count[state] = {}
            possible_actions = self.get_possible_actions(state)
            for action in possible_actions:
                Q[state][action] = 0.0
                returns_sum[state][action] = 0.0
                returns_count[state][action] = 0

        # Initialize policy: random policy
        policy = {}
        for state in self.state_space:
            possible_actions = self.get_possible_actions(state)
            if possible_actions:
                policy[state] = random.choice(possible_actions)
            else:
                policy[state] = None  # No possible actions

        for episode in range(num_episodes):
            # Generate an episode following the current policy
            episode_states_actions_rewards = []
            current_state = self.serialize_state()
            while True:
                action = policy.get(current_state, None)
                if action is None:
                    break  # No possible actions, end episode
                next_state, reward = self.apply_action_to_state(current_state, action)
                if next_state is None:
                    # Invalid move, end episode with penalty
                    episode_states_actions_rewards.append((current_state, action, -99))
                    break
                episode_states_actions_rewards.append((current_state, action, reward))
                if self.is_terminal(next_state):
                    break
                current_state = next_state

            # Calculate returns for the episode
            G = 0
            episode_length = len(episode_states_actions_rewards)
            for t in reversed(range(episode_length)):
                state, action, reward = episode_states_actions_rewards[t]
                G = discount_factor * G + reward
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                # Improve the policy to be greedy
                best_action = max(Q[state], key=Q[state].get)
                policy[state] = best_action

        # After all episodes, set the policy
        self.policy = policy
        print("Every Visit MC Policy Evaluation Completed.")

    def first_visit_mc_policy_evaluation_complete(self, num_episodes=1000, discount_factor=0.9):
        """
        Wrapper to perform First Visit MC Policy Evaluation and update the policy.
        """
        self.first_visit_mc_policy_evaluation(num_episodes, discount_factor)

    def every_visit_mc_policy_evaluation_complete(self, num_episodes=1000, discount_factor=0.9):
        """
        Wrapper to perform Every Visit MC Policy Evaluation and update the policy.
        """
        self.every_visit_mc_policy_evaluation(num_episodes, discount_factor)

    def auto_play(self):
        """
        Automatically plays the game according to the learned policy.
        """
        while not self.game_over:
            state = self.serialize_state()
            action = self.policy.get(state, None)
            if action is None:
                print("No action found in policy for the current state.")
                break
            action_vector = self.get_action_from_direction(action)
            if action_vector is None:
                print(f"Invalid action '{action}' in policy for state.")
                break
            time.sleep(1)  # Wait between actions for visualization
            box_placed, box_stuck = self.execute_action(action_vector)
            print(f"Action: {action}, Reward: {self.compute_reward(box_placed, box_stuck)}")
            self.draw_game()
            self.check_win()
