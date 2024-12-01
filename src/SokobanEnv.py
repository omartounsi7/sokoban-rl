from src.constants import *
import src.util
import gym
import numpy as np
import tkinter as tk

class SokobanEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, level_file):
        super(SokobanEnv, self).__init__()
        self.level_file = level_file
        self.load_level()
        self.action_space = gym.spaces.Discrete(4)  # 0: Up, 1: Left, 2: Down, 3: Right
        self.observation_space = gym.spaces.Box(low=0, high=6, shape=(self.height * self.width,), dtype=np.int8)
        self.cell_types = {
            ' ': 0,   # Empty space
            '#': 1,   # Wall
            '.': 2,   # Goal
            '$': 3,   # Box
            '*': 4,   # Box on goal
            '@': 5,   # Player
            '+': 6    # Player on goal
        }
        self.reset()
        self.root = None
        self.canvas = None
        self.game_over = False
        self.policy = None
        self.stop_pressed = False
        self.reset_pressed = False
    
    def load_level(self):
        with open(self.level_file, "r") as file:
            lines = file.readlines()
        self.level = [list(line.rstrip("\n")) for line in lines]
        self.initial_level = [row.copy() for row in self.level]
        self.height = len(self.level)
        self.width = max(len(line) for line in self.level)
        self.player_pos = src.util.find_player_in_state(self.level)
        self.total_boxes = sum(row.count("$") + row.count("*") for row in self.level)
        
    def reset(self):
        self.level = [row.copy() for row in self.initial_level]
        self.player_pos = src.util.find_player_in_state(self.level)
        self.game_over = False
        self.steps = 0
        return self.get_observation()
    
    def get_observation(self):
        obs = np.zeros((self.height, self.width), dtype=np.float32) # dtype of reset and observation space mismatch
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                obs[y][x] = self.cell_types.get(cell, 0)
        return obs.flatten()

    def step(self, action):
        dx, dy = ACTIONMAP[action]
        
        reward = MALUS
        moved_box = False
        x, y = self.player_pos
        nx, ny = x + dx, y + dy
        
        if src.util.is_wall(self.level, nx, ny):
            return self.get_observation(), reward, self.game_over, False, {}
        
        if src.util.is_box(self.level, nx, ny):
            if not src.util.move_box(self.level, nx, ny, dx, dy):
                return self.get_observation(), reward, self.game_over, False, {}
            moved_box = True
        
        src.util.move_agent(self.level, x, y, nx, ny)
        self.player_pos = (nx, ny)
        reward = STEPREWARD
        
        if moved_box:
            bx, by = nx + dx, ny + dy
            if src.util.is_box_stuck(self.level, bx, by):
                reward = SUPERMALUS
                self.game_over = True
            elif src.util.check_win(self.level):
                reward = SUPERBONUS
                self.game_over = True
            elif src.util.is_box_placed(self.level, bx, by):
                reward = BONUS
        
        return self.get_observation(), reward, self.game_over, False, {}
    
    def render(self):
        if self.root is None:
            self.init_viewer()
        self.draw_game()
    
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
                elif char == "*":
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

    def init_viewer(self):
        self.root = tk.Tk()
        self.root.title("Sokoban")
        self.title_label = tk.Label(self.root, text="Sokoban", font=("Helvetica", 16))
        self.title_label.pack(pady=10)
        self.canvas = tk.Canvas(self.root, width=500, height=450)
        self.canvas.pack()
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_autoplay)
        self.stop_button.pack(pady=5)
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_autoplay)
        self.reset_button.pack(pady=5)
        self.stop_pressed = False
        self.reset_pressed = False
    
    def stop_autoplay(self):
        self.stop_pressed = True
    
    def reset_autoplay(self):
        self.reset_pressed = True
    
    def close(self):
        if self.root:
            self.root.destroy()
            self.root = None
    
    def autoplay(self, policy):
        print("Playing level according to the learned policy...")
        self.policy = policy
        self.stop_pressed = False
        self.reset_pressed = False
        self.obs = self.reset()
        self.done = False
        self.action_sequence = []
        self.autoplay_step()
    
    def autoplay_step(self):
        self.render()
        if self.stop_pressed:
            self.close()
            return
        elif self.reset_pressed:
            self.reset_pressed = False
            self.obs = self.reset()
            self.done = False
            self.action_sequence = []
        elif not self.done:
            state = tuple(self.obs)
            action = self.policy.get(state, self.action_space.sample())
            self.obs, reward, self.done, truncated, info = self.step(action)
            self.action_sequence.append(action)
            self.steps += 1
            if self.done:
                if reward == SUPERBONUS:
                    print("Game won!")
                elif reward == SUPERMALUS:
                    print("Game lost.")
                literal_policy = [ACTIONSPACE[action] for action in self.action_sequence]
                print("Number of actions: " + str(self.steps))
                print("Learned policy: " + str(literal_policy))
        self.root.after(WAITTIME, self.autoplay_step)