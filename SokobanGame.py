import tkinter as tk

class Sokoban:
    def __init__(self, master, level_file):
        self.master = master
        self.master.title("Sokoban")
        self.level_file = level_file
        self.load_level()
        self.create_widgets()
        self.bind_keys()
        self.game_over = False

    def load_level(self):
        with open(self.level_file, 'r') as file:
            lines = file.readlines()
        self.level = [list(line.rstrip('\n')) for line in lines]
        self.initial_level = [row.copy() for row in self.level]
        self.height = len(self.level)
        self.width = max(len(line) for line in self.level)
        self.find_player()
        self.total_boxes = sum(row.count('$') for row in self.level)

    def reset_level(self):
        self.level = [row.copy() for row in self.initial_level]
        self.find_player()
        self.game_over = False
        self.message_label.config(text="")
        self.draw_game()

    def find_player(self):
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                if char == '@' or char == '+':
                    self.player_pos = (x, y)
                    return

    def create_widgets(self):
        self.title_label = tk.Label(self.master, text="Sokoban", font=("Helvetica", 16))
        self.title_label.pack(pady=10)
        self.canvas = tk.Canvas(self.master, width=450, height=450)
        self.canvas.pack()
        self.reset_button = tk.Button(self.master, text="Reset", command=self.reset_level)
        self.reset_button.pack(pady=5)
        self.message_label = tk.Label(self.master, text="", font=("Helvetica", 14))
        self.message_label.pack(pady=5)
        self.draw_game()

    def bind_keys(self):
        self.master.bind("<Key>", self.key_pressed)

    def key_pressed(self, event):
        if self.game_over:
            return
        
        if event.keysym == 'r':
            self.reset_level()
            return

        action = self.get_action(event.keysym)
        if action:
            self.execute_action(action)
            self.draw_game()
            self.check_win()

    def get_action(self, key):
        key_mapping = {
            'w': (0, -1),
            'a': (-1, 0),
            's': (0, 1),
            'd': (1, 0),
            'Up': (0, -1),
            'Left': (-1, 0),
            'Down': (0, 1),
            'Right': (1, 0)
        }
        return key_mapping.get(key, None)

    def execute_action(self, action):
        dx, dy = action
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        if self.is_wall(nx, ny):
            return

        if self.is_box(nx, ny):
            if not self.move_box(nx, ny, dx, dy):
                return

        # Update player's current position
        if self.level[y][x] == '@':
            self.level[y][x] = ' '
        elif self.level[y][x] == '+': 
            self.level[y][x] = '.'

        # Update new player's position
        if self.level[ny][nx] == '.':
            self.level[ny][nx] = '+'
        else:
            self.level[ny][nx] = '@'

        self.player_pos = (nx, ny)

    def is_wall(self, x, y):
        return self.level[y][x] == '#'

    def is_box(self, x, y):
        return self.level[y][x] in ['$','x']

    # check if the position is our goal
    def is_goal(self, x, y):
        return self.level[y][x] == '.'
    
    # Check if the position (x, y) is surrounded by walls in two orthogonal directions
    def is_corner(self, x, y):
        return (self.is_wall(x - 1, y) and self.is_wall(x, y - 1)) or \
               (self.is_wall(x + 1, y) and self.is_wall(x, y - 1)) or \
               (self.is_wall(x - 1, y) and self.is_wall(x, y + 1)) or \
               (self.is_wall(x + 1, y) and self.is_wall(x, y + 1))
    
    def move_box(self, x, y, dx, dy):
        nx, ny = x + dx, y + dy
        if self.level[ny][nx] in [' ', '.']:
            # Update box's current position
            if self.level[y][x] == '$':
                self.level[y][x] = ' '
            elif self.level[y][x] == 'x':
                self.level[y][x] = '.'

            # Update box's new position
            if self.level[ny][nx] == '.':
                self.level[ny][nx] = 'x'
            else:
                self.level[ny][nx] = '$'
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
                if char == '#':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")
                elif char == '.':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                elif char == '$':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_rectangle(x1+10, y1+10, x2-10, y2-10, fill="brown")
                elif char == 'x':  # Box on goal
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_rectangle(x1+10, y1+10, x2-10, y2-10, fill="green")
                elif char == '@':
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="blue")
                elif char == '+':  # Player on goal
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightyellow")
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="blue")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

    def check_win(self):
        for row in self.level:
            if '$' in row:
                return False
        self.game_over = True
        self.message_label.config(text="You won!")
        return True
    
    # RL Part
    # function for checking if the box is on goal
    def is_box_on_goal(self, action):
        dx, dy = action
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        # Check if the action pushes a box to a goal position
        if self.is_box(nx, ny):
            box_next_x, box_next_y = nx + dx, ny + dy
            return self.is_goal(box_next_x, box_next_y)
        
        return False
    
    # function for checking if the box is stuck: box is in corner with 2 sides blocked, and is not a goal
    def is_box_stuck(self, action):
        dx, dy = action
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        # Check if the move would cause a box to get stuck
        if self.is_box(nx, ny):
            box_next_x, box_next_y = nx + dx, ny + dy
            return self.is_corner(box_next_x, box_next_y) and not self.is_goal(box_next_x, box_next_y)
        
        return False

    # function for computing reward. 
    # if box is correctly placed, gets a positive k reward
    # if box is stuck (game over), gets -99
    # for any other moves, let's say just moving around the map or just pushing a box somewhere, gets 0
    def compute_reward(self, action):
        k = self.total_boxes
        p_for_loss = -99
        neutral = 0

        if self.is_box_on_goal(action):
            return k
        
        if self.is_box_stuck(action):
            return p_for_loss
        
        return neutral
    